# Reference: https://github.com/cod3licious/textcatvis

import numpy as np
import re

def cleaner(text, to_lower=True, norm_num=False):
    # if the to_lower flag is true, convert the text to lowercase
    text = text.lower() if to_lower else text
    # Removes unwanted http hyper links in text
    text = re.sub(r"http(s)?://\S*", " ", text)
    # In some cases, one may want to normalize numbers for better visualization
    text = re.sub(r"[0-9]", "1", text) if norm_num else text
    # remove non-alpha numeric characters [!, $, #, or %] and normalize whitespace
    text = re.sub(r"[^A-Za-z0-9-]+", " ", text)
    # replace leftover unwanted white space
    text = re.sub(r"\s+", " ", text)
    # remove trailing or leading white spaces
    text = text.strip()
    return text

def relevance_wt_transformer(raw_txt, wts_as_dict):
    # normalize score by absolute max value
    if isinstance(wts_as_dict, dict):
        max_wt = np.max(np.abs(list(wts_as_dict.values())))
        wts_as_dict = {word: wts_as_dict[word]/max_wt for word in wts_as_dict}
        # transform dict into word list with scores
        relevance_wts = []
        for word in re.findall(r'[\w-]+', raw_txt, re.UNICODE):
            # Clean up the raw text
            word_pp = cleaner(word)
            if word_pp in wts_as_dict:
                relevance_wts.append((word, wts_as_dict[word_pp]))
            else:
                relevance_wts.append((word, None))
    return relevance_wts


def show_in_notebook():
    from IPython.core.display import display, HTML
    return HTML('./rendered.html')