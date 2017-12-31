# Reference:
# 1. https://github.com/cod3licious/textcatvis
# 2. https://buhrmann.github.io/tfidf-analysis.html

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from skater.util.text_ops import cleaner


def _handling_ngrams_wts(original_feat_dict):
    # Currently, when feature dictionary contains continuous sequences such as 2-gram/3-gram etc. sequences as features,
    # a short term solution is to further split them into additional keys.
    # e.g. {'stay ball dropped': 0,5} will be split into a new dict as {'stay':0.5, 'ball': 0.5, 'dropped': 0.5}
    # TODO: this is just a temporary solution for handling n-grams, figure out a better solution
    for k in list(original_feat_dict.keys()):
        additional_keys = k.split()
    for a_k in additional_keys:
        if a_k in original_feat_dict:
            original_feat_dict[a_k] += original_feat_dict[a_k]
        else:
            original_feat_dict[a_k] = original_feat_dict[k]
    new_dict = original_feat_dict
    return new_dict


def relevance_wt_assigner(raw_txt, wts_as_dict):
    # normalize score by absolute max value
    if isinstance(wts_as_dict, dict):
        feature_scores = _handling_ngrams_wts(wts_as_dict)
        max_wt = np.max(np.abs(list(feature_scores.values())))
        wts_as_dict = {word: feature_scores[word]/max_wt for word in feature_scores.keys()}
        # transform dict into list of tuples (word, relevance_wts)
        # TODO look into removing the below occurring side effect
        relevance_wts = []
        for word in raw_txt.split():
            # Clean up the raw word for irregularities
            word_cleaned_as_key = cleaner(word)
            if word_cleaned_as_key in wts_as_dict.keys():
                relevance_wts.append((word, wts_as_dict[word_cleaned_as_key]))
            else:
                relevance_wts.append((word, None))
    else:
        raise Exception('relevance wts currently needs to be as dict')
    return relevance_wts


def vectorize_as_tf_idf(data, **kwargs):
    """Term Frequency times Inverse Document Frequency"""
    # TODO: extend support to other forms of Vectorization schemes - Feature Hashing
    # Converting raw document to tf-idf feature matrix
    tfidf_vec = TfidfVectorizer(sublinear_tf=kwargs['sublinear_tf'], max_df=kwargs['max_df'],
                    stop_words=kwargs['stop_words'], smooth_idf=kwargs['smooth_idf'],
                                ngram_range=kwargs['ngram_range'])
    X = tfidf_vec.fit_transform(data)
    return tfidf_vec, X


def get_feature_names(vectorizer_inst):
    return vectorizer_inst.get_feature_names()


def _default_feature_selection(X, y, feature_names, top_k):
    y
    arg_sort = lambda r, k: np.argsort(r)[::-1][:k]
    top_k_index = arg_sort(X, top_k)
    top_features = [(feature_names[i], X[i]) for i in top_k_index]
    return None, None, top_features


def auto_feature_selection(X, y, feature_names, top_k='all'):
    """
    Feature selection currently is done using Chi2.
    Other solutions may be added shortly.
    """
    ch2 = SelectKBest(chi2, top_k)
    X_new = ch2.fit_transform(X, y)
    # retrieve the feature names post selection
    # Reference: https://stackoverflow.com/questions/14133348/show-feature-names-after-feature-selection
    features_scores = tuple(zip(feature_names, ch2.scores_))
    selected_feature = [features_scores[i] for i in ch2.get_support(indices=True)]
    return ch2, X_new, selected_feature


def _compute_top_features(X, y, features, feature_selection_type='default', top_k=25):
    """ Computes top 'k' features in a row.
    Parameters
    __________
    X: input data
    features:
    feature_selection_type:
     - 'default': uses vanila ranking of features based on Frequency Vectorization wts
     - 'ch2': chi-square statistical test of independence. It helps feature selection by getting rid of the features
        that are most likely independent of the categorization class or label (discarding irrelevant features)
        Reference:
        * https://nlp.stanford.edu/IR-book/html/htmledition/chi-square-feature-selection-1.html
        * Liu et al.'95: http://sci2s.ugr.es/keel/pdf/specific/congreso/liu1995.pdf
    top_k: number of top features to return

    Returns
    _______
    df : pandas.DataFrame

    """
    fs_choice_dict = {
        'default': _default_feature_selection,
        'chi2': auto_feature_selection
    }

    type_inst, new_x, top_features = fs_choice_dict[feature_selection_type](X, y, features, top_k)
    df = pd.DataFrame(top_features)
    df.columns = ['features', 'scores']
    return df


def query_top_features_in_doc(data, y, features, feature_selection_choice='default', top_k=25):
    """ Compute top features for each document in the corpus. The scores are dependent on the type of
    transformation applied e.g. TF-IDF

    Returns
    _______
    pandas.DataFrame with columns 'features', 'scores'
    """
    row = np.squeeze(data.toarray())
    return _compute_top_features(X=row, y=y, features=features, feature_selection_type=feature_selection_choice,
                                     top_k=top_k)


# Lamda for converting data-frame to a dictionary
convert_dataframe_to_dict = lambda key_column_name, value_column_name, df: \
    df.set_index(key_column_name).to_dict()[value_column_name]


def query_top_features_overall(data, y_true, feature_list, min_threshold=0.1, feature_selection='default',
                                      summarizer_type='mean', top_k=25):
    """
    """
    # TODO add summarizer type as a sub-argument
    # The use of summarizer to capture the tf-idf scores overall with use of vanilla aggregation is Experimental.
    # The idea here is to use such aggregation as a simple ranking criteria
    # On-going discussion: https://stackoverflow.com/questions/42269313/interpreting-the-sum-of-tf-idf-scores-of-words-across-documents
    # Always safe to compute globally responsible features using more theoretical statistical tests e.g. chi2
    if feature_selection is 'default':
        d = data.toarray()
        d[d < min_threshold] = 0
        summarizer_default = lambda x: np.sum(x, axis=0)
        summarizer_mean = lambda x: np.mean(x, axis=0)
        summarizer_median = lambda x: np.median(x, axis=0)
        summarizer_choice_dict = {
            'sum': summarizer_default,
            'mean': summarizer_mean,
            'median': summarizer_median
        }

        tfidf_summarized = summarizer_choice_dict[summarizer_type](d)
        temp_df = _compute_top_features(tfidf_summarized, y_true, feature_list, feature_selection, top_k)
        return temp_df
    else:
        temp_df = _compute_top_features(data, y_true, feature_list, 'chi2', top_k)
        return temp_df


def query_topk_tfidf_features_by_class(X, y, feature_names, class_index,
                                       summarizer_type='mean', topk_features=25, min_threshold=0.1):
    """
    """
    indexes = list(np.where(y==class_index))
    feature_df = query_top_features_overall(data=X[indexes[0]], y_true=y[indexes[0]], feature_list=feature_names,
                                            min_threshold=min_threshold, summarizer_type=summarizer_type,
                                            top_k=topk_features)
    return feature_df


def _single_layer_lrp(feature_coef_df, bias, features_by_class, top_k):
    """
    References
    ----------
    Franziska Horn, Leila Arras, Gregoire Montavon, Klaus-Robert Muller, Wojciech Samek. 2017
    Exploring text datasets by visualizing relevant words (https://arxiv.org/abs/1707.05261)
    """


    merged_df = pd.merge(feature_coef_df, features_by_class, on='features')
    merged_df['coef_wts'] = merged_df['coef_wts'].astype('float64')
    merged_df['scores'] = merged_df['scores'].astype('float64')
    merged_df['coef_score_wts'] = merged_df['coef_wts']*merged_df['scores'] + float(bias)

    # This is sorting is more of a precaution for corner cases, might be removed as the implementation matures
    top_feature_df = merged_df.nlargest(top_k, 'coef_scores_wts')[['features', 'coef_scores_wts']]
    top_feature_df_dict = convert_dataframe_to_dict('features', 'coef_scores_wts', top_feature_df)
    return top_feature_df_dict, top_feature_df, merged_df


def _based_on_learned_estimator(feature_coef_df, bias, top_k):
    feature_coef_df['coef_wts'] = feature_coef_df['coef_wts'].astype('float64')
    feature_coef_df['coef_wts_intercept'] = feature_coef_df['coef_wts'] + float(bias)
    top_feature_df = feature_coef_df.nlargest(top_k, 'coef_wts_intercept')

    top_feature_df_dict = convert_dataframe_to_dict('features', 'coef_wts_intercept', top_feature_df)
    return top_feature_df_dict, top_feature_df, feature_coef_df


def understand_estimator(estimator, class_label_index, tfidf_wts_by_class, feature_names,
                         top_k, relevance_type='default'):
    # Currently, support for sklearn based estimator
    # TODO: extend it for estimator from other frameworks - MLLib, H20, vw
    if ('coef_' in estimator.__dict__) is False:
        raise KeyError('the estimator does not support coef, try using LIME for local interpretation')

    # Currently, support for sklearn based estimator
    # TODO: extend it for estimator from other frameworks - MLLib, H20, vw
    # sort the coefficients in descending order, this will give access to the coefficient wts
    coef_array = np.squeeze(estimator.coef_[class_label_index])
    no_of_features = top_k
    _, _, feature_coef_list = _default_feature_selection(coef_array, feature_names, no_of_features)
    feature_coef_df = pd.DataFrame(feature_coef_list, columns=['features', 'coef_wts'])
    bias = estimator.intercept_[class_label_index]/no_of_features

    if relevance_type == 'default':
        top_feature_df_dict, top_feature_df, feature_coef_df = _based_on_learned_estimator(feature_coef_df, bias, no_of_features)
    elif relevance_type == 'SLRP':
        top_feature_df_dict, top_feature_df, feature_coef_df = _single_layer_lrp(feature_coef_df, bias,
                                                                                 tfidf_wts_by_class, top_k)
    return top_feature_df_dict, top_feature_df, feature_coef_df
