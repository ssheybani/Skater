import matplotlib as mpl
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

def display_in_html(text, scores, fname='testfile', metainf='', pos_clr_name='Blues',
         neg_clr_name='Reds', highlight_oov=False):
    """
    Reference: http://matplotlib.org/examples/color/colormaps_reference.html
    Based on the original text and relevance scores, generate a html doc highlighting positive / negative words
    Inputs:
        - text: the raw text in which the words should be highlighted
        - scores: a dictionary with {word: score} or a list with tuples [(word, score)]
        - fname: the name (path) of the file
        - metainf: an optional string which will be added at the top of the file (e.g. true class of the document)
        - highlight_oov: if True, out-of-vocabulary words will be highlighted in yellow (default False)
    Saves the visualization in 'fname.html' (you probably want to make this a whole path to not clutter your main directory...)
    """
    # color-maps
    cmap_pos = get_cmap(pos_clr_name)
    cmap_neg = get_cmap(neg_clr_name)
    norm = mpl.colors.Normalize(0., 1.)

    htmlstr = u'<body><div style="white-space: pre-wrap; font-family: monospace;">'
    if metainf:
        htmlstr += '%s\n\n' % metainf
    resttext = text
    for word, score in scores:
        # was anything before the identified word? add it unchanged to the html
        htmlstr += resttext[:resttext.find(word)]
        # cut off the identified word
        resttext = resttext[resttext.find(word) + len(word):]
        # get the colorcode of the word
        rgbac = (1., 1., 0.)  # for unknown words
        if highlight_oov:
            alpha = 0.3
        else:
            alpha = 0.
        if score is not None:
            if score < 0:
                rgbac = cmap_neg(norm(-score))
            else:
                rgbac = cmap_pos(norm(score))
            alpha = 0.5
        htmlstr += u'<span style="background-color: rgba(%i, %i, %i, %.1f)">%s</span>' \
                   % (round(255 * rgbac[0]), round(255 * rgbac[1]), round(255 * rgbac[2]), alpha, word)
    # after the last word, add the rest of the text
    htmlstr += resttext
    htmlstr += u'</div></body>'
    with codecs.open('%s.html' % fname, 'w', encoding='utf8') as f:
        f.write(htmlstr)


def show_in_notebook():
    from IPython.core.display import display, HTML
    return HTML('./rendered.html')