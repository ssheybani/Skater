import unittest

from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np

from skater.core.local_interpretation.text_interpreter import auto_feature_selection, \
    query_top_features_by_class, understand_estimator, vectorize_as_tf_idf, get_feature_names
from skater.util.text_ops import cleaner
from skater.util.dataops import convert_dataframe_to_dict
from skater.util.text_ops import preprocessor


class TestTextInterpreter(unittest.TestCase):
    """
    Module to evaluate and interpret text
    """

    def setUp(self):
        categories = [
            'rec.autos',
            'rec.motorcycles'
        ]
        remove = ('headers', 'footers', 'quotes')

        data_train = fetch_20newsgroups(subset='train',
                                        categories=categories, shuffle=True, random_state=0, remove=remove)

        self.X_train = data_train.data
        self.X_train = [preprocessor(t) for t in self.X_train]
        self.y_train = data_train.target

        data_test = fetch_20newsgroups(subset='test', categories=categories,
                                       shuffle=True, random_state=0,
                                       remove=remove)
        self.X_test = data_test.data
        self.y_test = data_test.target

        self.param_dict_vectorizer = {
            'sublinear_tf': True,
            'max_df': 0.5,
            'stop_words': 'english',
            'smooth_idf': True,
            'ngram_range': (1, 3),
            # 'max_features': 10 (didn't give the right result)
        }


    def test_cleaner(self):
        raw_text = "    11111this is just an example..., don't be surprised!!!0000000    "
        expected_result = "this is just an example , don t be surprised"
        result = cleaner(raw_text, to_lower=True, norm_num=False, char_to_strip="0|' '|1")
        self.assertEquals(result, expected_result)


    def test_utils(self):
        # converting of dataframe to dict
        data = np.array([['', 'feature_name', 'relevance_scores'],
                         ['R1', 'F1', 20],
                         ['R2', 'F2', 40]])
        test_df = pd.DataFrame(data=data[1:, 1:], index=data[1:, 0], columns=data[0, 1:])
        test_dict = convert_dataframe_to_dict("feature_name", "relevance_scores", test_df)
        self.assertEquals(test_dict['F1'], '20')


    def test_get_feature_names(self):
        vectorizer, X_train = vectorize_as_tf_idf(self.X_train, **self.param_dict_vectorizer)
        feature_names = get_feature_names(vectorizer_inst=vectorizer)
        self.assertEquals(len(feature_names), 126384)


    def test_understand_estimator(self):
        vectorizer, X_train = vectorize_as_tf_idf(self.X_train, **self.param_dict_vectorizer)
        feature_names = get_feature_names(vectorizer_inst=vectorizer)

        select_type_inst, X_new_train, selected_feature = auto_feature_selection(X_train, self.y_train,
                                                                                 feature_names=feature_names)
        self.assertEquals(len(selected_feature), 126384)

        clf = SGDClassifier(alpha=.0001, n_iter=10, penalty="elasticnet", loss='log', random_state=1)
        clf.fit(X_new_train, self.y_train)

        features_to_consider = query_top_features_by_class(X_new_train, self.y_train, selected_feature, class_index=0,
                                                           topk_features=len(selected_feature),
                                                           summarizer_type='mean', min_threshold=0.1)

        relevance_dict, relevance_df, _ = understand_estimator(clf, 0, features_to_consider, selected_feature,
                                                               top_k=len(features_to_consider), relevance_type='default')

        self.assertEquals(len(relevance_df.columns), 3)
        # validate the construction of the data-frame
        rows = [row for row in relevance_df.head(2).itertuples()]
        self.assertEquals(rows[0].features, "bike")
        self.assertEquals(rows[1].features, "bikes")
        self.assertEquals(rows[0].coef_scores_wts > 0, True)
        self.assertEquals(rows[1].relevance_wts > 0, True)

        relevance_dict, relevance_df, _ = understand_estimator(clf, 0, features_to_consider, selected_feature,
                                                               top_k=len(features_to_consider), relevance_type='SLRP')
        self.assertEquals(len(relevance_df.columns), 2)
        # validate the construction of the data-frame
        rows = [row for row in relevance_df.head(2).itertuples()]
        self.assertEquals(rows[0].features, "ll")
        self.assertEquals(rows[1].features, "bmw")
        self.assertEquals(rows[1].relevance_wts > 0, True)

        relevance_dict, relevance_df, _ = understand_estimator(clf, 1, features_to_consider, selected_feature,
                                                               top_k=len(features_to_consider), relevance_type='default')
        self.assertEquals("{:.2f}".format(relevance_df[relevance_df['features'] == 'sounds']
                                          .coef_scores_wts.values[0]), '0.55')


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTextInterpreter)
    unittest.TextTestRunner(verbosity=2).run(suite)
