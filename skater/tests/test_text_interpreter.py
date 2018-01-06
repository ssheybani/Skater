import unittest
from skater.util.text_ops import cleaner
from sklearn.datasets import fetch_20newsgroups

from skater.core.local_interpretation.text_interpreter import auto_feature_selection, query_top_features_in_doc, \
    query_top_features_overall, query_top_features_by_class, convert_dataframe_to_dict, understand_estimator, \
    relevance_wt_assigner, vectorize_as_tf_idf, get_feature_names

from skater.util.text_ops import preprocessor

class TestTextInterpreter(unittest.TestCase):
    """
    Module to evaluate text interpretation functionalities
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
        self.X_train = [preprocessor(t) for t in X_train]
        self.y_train = data_train.target

        data_test = fetch_20newsgroups(subset='test', categories=categories,
                                       shuffle=True, random_state=0,
                                       remove=remove)
        self.X_test = data_test.data
        self.y_test = data_test.target


    def test_cleaner(self):
        raw_text = "    11111this is just an example..., don't be surprised!!!0000000    "
        expected_result = "this is just an example , dont be surprised"
        result = cleaner(raw_text, to_lower=True, norm_num=False, char_to_strip="0|' '|1")
        self.assertEquals(result, expected_result)

    def test_understand_estimator(self):
        select_type_inst, X_new_train, selected_feature = auto_feature_selection(self.X_train, self.y_train,
                                                                                 feature_names=feature_names)




if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTextInterpreter)
    unittest.TextTestRunner(verbosity=2).run(suite)