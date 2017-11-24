from matplotlib.cm import get_cmap
from skater.core.local_interpretation.text_interpreter import relevance_wt_transformer
from wordcloud import (WordCloud, get_single_color_func)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import codecs


def build_html(text, feature_relevance_wts, font_size='10pt', file_name='rendered', metainf='',
                    pos_clr_name='Blues', neg_clr_name='Reds', highlight_oov=False):
    """
    Reference: http://matplotlib.org/examples/color/colormaps_reference.html
    Based on the original text and relevance scores, generate a html doc highlighting positive / negative words
    Inputs:
        - text: the raw text in which the words should be highlighted
        - scores: a dictionary with {word: score} or a list with tuples [(word, score)]
        - file_name: the name (path) of the file
        - metainf: an optional string which will be added at the top of the file (e.g. true class of the document)
        - highlight_oov: if True, out-of-vocabulary words will be highlighted in yellow (default False)
    Saves the visualization in 'file_name.html' (you probably want to make this a whole path to not clutter your main directory...)
    """
    # color-maps
    cmap_pos = get_cmap(pos_clr_name)
    cmap_neg = get_cmap(neg_clr_name)
    norm = mpl.colors.Normalize(0., 1.)

    html_str = u'<body><div style="white-space: pre-wrap; font-size: {}; font-family: Verdana;">'.format(font_size)
    html_str += '%s\n\n' % metainf if metainf else html_str

    rest_text = text
    relevance_wts = relevance_wt_transformer(text, feature_relevance_wts)
    for word, wts in relevance_wts:
        # was anything before the identified word? add it unchanged to the html
        html_str += rest_text[:rest_text.find(word)]
        # cut off the identified word
        rest_text = rest_text[rest_text.find(word) + len(word):]
        # get the colorcode of the word
        rgbac = (1., 1., 0.)  # for unknown words
        alpha = 0.3 if highlight_oov else 0.

        if wts is not None:
            if wts < 0:
                rgbac = cmap_neg(norm(-wts))
            else:
                rgbac = cmap_pos(norm(wts))
            alpha = 0.5
        html_str += u'<span style="background-color: rgba(%i, %i, %i, %.1f)">%s</span>' \
                   % (round(255 * rgbac[0]), round(255 * rgbac[1]), round(255 * rgbac[2]), alpha, word)
    # after the last word, add the rest of the text
    html_str += rest_text
    html_str += u'</div></body>'
    with codecs.open('%s.html' % file_name, 'w', encoding='utf8') as f:
        f.write(html_str)


class _GroupedColorFunc(object):
    # Reference: https://github.com/amueller/word_cloud/blob/master/examples/colored_by_group.py
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.
       Uses wordcloud.get_single_color_func
       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.
       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)


    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                    color_func for (color_func, words) in self.color_func_to_words
                    if word in words)
        except StopIteration:
            color_func = self.default_color_func
        return color_func


    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)


def generate_word_cloud(relevant_feature_wts, pos_clr_name='blue',
                        neg_clr_name='red', threshold=0.1, mask_filename=None):
    # Prepare a color map to word aggregated on threshold
    color_mapping = {pos_clr_name: [], neg_clr_name: []}
    color_map_append = lambda key, value: color_mapping[key].append(value)

    for word, wt in relevant_feature_wts.items():
        color_map_append(neg_clr_name, word) if wt < threshold else color_map_append(pos_clr_name, word)
        # this is a temporary fix as there seems to be a bug in the wordcloud implementation used.
        # If there are continuous set of 0 as weights, then one get 'float division by zero' or
        # 'cannot convert float NaN to integer'. Below mentioned code is a work around for now
        # TODO: Figure out a better fix
        if wt == 0:
            relevant_feature_wts[word] = 0.000000001

    default_color = 'yellow'
    grouped_color_func = _GroupedColorFunc(color_mapping, default_color)

    mask_file = np.array(Image.open(mask_filename)) if mask_filename is not None else mask_filename
    # TODO extend support for Word Cloud params
    wc = WordCloud(background_color="white", random_state=0, max_words=len(relevant_feature_wts), width=900,
                   height=600, mask=mask_file, color_func=grouped_color_func)
    wc.generate_from_frequencies(relevant_feature_wts)
    plt.imshow(wc)
    plt.axis("off")


def show_in_notebook(file_name='rendered'):
    from IPython.core.display import display, HTML
    return HTML('./{}.html'.format(file_name))