# Reference: https://github.com/cod3licious/textcatvis

import numpy as np
import pandas as pd
import re
from functools import map
from sklearn.feature_extraction.text import TfidfVectorizer


def cleaner(text, to_lower=True, norm_num=False, char_to_strip=' ', non_alphanumeric_exceptions=","):
    # if the to_lower flag is true, convert the text to lowercase
    text = text.lower() if to_lower else text
    # Removes unwanted http hyper links in text
    text = re.sub(r"http(s)?://\S*", " ", text)
    # In some cases, one may want to normalize numbers for better visualization
    text = re.sub(r"[0-9]", "1", text) if norm_num else text
    # remove non-alpha numeric characters [!, $, #, or %] and normalize whitespace
    text = re.sub(r"[^A-Za-z0-9-" + non_alphanumeric_exceptions + "]", " ", text)
    # replace leftover unwanted white space
    text = re.sub(r"\s+", " ", text)
    # remove trailing or leading white spaces
    text = text.strip(char_to_strip)
    return text


def relevance_wt_transformer(raw_txt, wts_as_dict):
    # normalize score by absolute max value
    if isinstance(wts_as_dict, dict):
        max_wt = np.max(np.abs(list(wts_as_dict.values())))
        wts_as_dict = {word: wts_as_dict[word]/max_wt for word in wts_as_dict}

        # Clean up the raw word for irregularities
        cleaned_words = list(map(cleaner, raw_txt.split()))
        # transform dict into list of tuples (word, relevance_wts)
        relevance_wts = [ (word, wts_as_dict[word]) if word in wts_as_dict else (word, None)
                          for word in cleaned_words ]
    return relevance_wts


def vectorize_as_tf_idf(data):
    tfidf_vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                    stop_words='english')
    return tfidf_vec.fit_transform(data)


def feature_names(vectorizer_inst):
    return vectorizer_inst.get_feature_names()


def _top_k_tfidf_features(each_row, features, top_k=25):
    """ Computes top 'k' tf-idf values in a row.

    Parameters
    __________
    each_row:
    features:
    top_k:

    Returns
    _______
    df : pandas.DataFrame

    """
    top_k_index = np.argsort(each_row)[::-1][:top_k]
    top_features = [(features[i], each_row[i]) for i in top_k_index]
    df = pd.DataFrame(top_features)
    df.columns = ['features', 'tf_idf']
    return df


def topk_tfidf_features_in_doc(data, features, top_k=20):
    """ Compute top tf-idf features for each document in the corpus

    Returns
    _______
    pandas.DataFrame with columns 'features', 'tf_idf'
    """
    row = np.squeeze(data.toarray())
    return _top_k_tfidf_features(row, features, top_k)


tf_idf_relevance_wts_dict = lambda df: df.set_index('features').to_dict()['tf_idf']


def _topk_tfidf_features_overall(data, features, min_tfidf=0.1, top_n=25):
    """Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids.
    """
    d = data.toarray()

    d[d < min_tfidf] = 0
    tfidf_means = np.mean(d, axis=0)
    return _top_k_tfidf_features(tfidf_means, features, top_n)


def topk_tfidf_features_by_class(Xtr, y, features, class_index, min_tfidf=0.1, top_n=25, ):
    """
    """
    labels = np.unique(y)
    ids_by_class = list(map(lambda label: np.where(y==label), labels))
    feature_df = _topk_tfidf_features_overall(Xtr, features, ids_by_class[class_index], min_tfidf=min_tfidf, top_n=top_n)
    feature_df.label = ids_by_class[class_index]
    return feature_df