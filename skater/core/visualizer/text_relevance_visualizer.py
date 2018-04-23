from matplotlib.cm import get_cmap
import matplotlib as mpl
from matplotlib.patches import Patch
from PIL import Image
from wordcloud import (WordCloud, get_single_color_func)
import numpy as np
import pandas as pd
import codecs

from skater.data.datamanager import DataManager as DM
from skater.util.exceptions import MatplotlibUnavailableError
from skater.util.logger import build_logger
from skater.util.logger import _INFO
from skater.core.local_interpretation.text_interpreter import relevance_wt_assigner
from skater.util.dataops import convert_dataframe_to_dict
from skater.util.text_ops import generate_word_list

logger = build_logger(_INFO, __name__)


def __set_plot_feature_relevance_keyword(**plot_kw):
    plot_name = plot_kw['plot_name'] if 'plot_name' in plot_kw.keys() else 'feature_relevance.png'
    top_k = plot_kw['top_k'] if 'top_k' in plot_kw.keys() else 10
    color_map = plot_kw['color_map'] if 'color_map' in plot_kw.keys() else ('Red', 'Blue')
    fig_size = plot_kw['fig_size'] if 'fig_size' in plot_kw.keys() else (20, 10)
    font_name = plot_kw['font_name'] if 'font_name' in plot_kw.keys() else "Avenir Black"
    txt_font_size = plot_kw['txt_font_size'] if 'txt_font_size' in plot_kw.keys() else '14'
    return plot_name, top_k, color_map, fig_size, font_name, txt_font_size


# Reference: https://stackoverflow.com/questions/30618002/static-variable-in-a-function-with-python-decorator
def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("plot_counter", 0)
def build_visual_explainer(text, relevance_scores, font_size='12pt', file_name='rendered', title='',
                           pos_clr_name='Reds', neg_clr_name='Blues', highlight_oov=False,
                           enable_plot=False, **plot_kw):
    """
    Build a visualizer explainer highlighting the positive(color code: Red(default)) and
    negative(color code: Blue(default)) effect of the input features

    Parameters
    ----------
        - text: the raw text in which the words should be highlighted
        - relevance_scores: a dictionary with {word: score} or a list with tuples [(word, score)]
        - file_name: the name (path) of the file
        - metainf: an optional string which will be added at the top of the file (e.g. true class of the document)
        - highlight_oov: if True, out-of-vocabulary words will be highlighted in yellow (default False)
    Saves the visualization in 'file_name.html' (you probably want to make this a whole path to not clutter your main directory...)

    References
    ----------
    * http://matplotlib.org/examples/color/colormaps_reference.html
    * https://github.com/cod3licious/textcatvis
    """
    # TODO: Add support for better visualization and plotting frameworks- e.g bokeh, plotly

    # Process the raw text to a word list
    _words = generate_word_list(text, ' ')
    features_df = pd.DataFrame({'features': _words})
    scores_df = pd.DataFrame({'relevance_scores': relevance_scores.tolist()})
    # Merge the data-frame column-wise, assert if the length of the word list and relevance
    # score data-frame does not match. This is important as currently the join is done on the index
    assert len(_words) == scores_df.shape[0]
    words_scores_df = features_df.join(scores_df)
    # assign column names to df containing 'words | relevance score'
    words_scores_df.columns = ['features', 'relevance_scores']
    words_scores_dict = convert_dataframe_to_dict('features', 'relevance_scores', words_scores_df)

    # generate plot displaying feature relevance rank ordered
    f_name = plot_feature_relevance(words_scores_df, **plot_kw) if enable_plot else None
    logger.info("Relevance plot name: {}".format(f_name))

    # color-maps
    cmap_pos = get_cmap(pos_clr_name)
    cmap_neg = get_cmap(neg_clr_name)
    # color mapping for non-vocabulary words
    rgbac = (0.1, 0.1, 1.0)
    # adjust opacity for non-vocabulary words
    alpha = 0.2 if highlight_oov else 0.
    norm = mpl.colors.Normalize(0., 1.)

    # build the html structure
    html_str = u'<body><h3>{}</h3>'\
               u'<div class="row" style=background-color:#F5F5F5' \
               u'white-space: pre-wrap; ' \
               u'font-size: {}; ' \
               u'font-family: Avenir Black>'.format(title, font_size)

    rest_text = text
    relevance_wts = relevance_wt_assigner(text, words_scores_dict)
    for word, wts in relevance_wts:
        html_str += rest_text[:rest_text.find(word)]
        # cut off the identified word
        rest_text = rest_text[rest_text.find(word) + len(word):]
        if wts is not None:
            rgbac = cmap_neg(norm(-wts)) if wts < 0 else cmap_pos(norm(wts))
            # adjusting opacity for in-dictionary words
            alpha = 0.5
        html_str += u'<span style="background-color: rgba(%i, %i, %i, %.1f)">%s</span>' \
                    % (round(255 * rgbac[0]), round(255 * rgbac[1]), round(255 * rgbac[2]), alpha, word)
    # after the last word, add the rest of the text
    html_str += rest_text
    html_str += u'</div>'

    # Embed the feature relevance scores as a plot
    build_visual_explainer.plot_counter += 1
    html_str += u'<div align="center"><img src="./{}?{}"</div>'.\
        format(f_name, build_visual_explainer.plot_counter) if f_name is not None else ''
    html_str += u'</body>'
    file_name_with_extension = '{}.html'.format(file_name)
    with codecs.open(file_name_with_extension, 'w', encoding='utf8') as f:
        f.write(html_str)
        logger.info("Visual Explainer built, "
                    "use show_in_notebook to render in Jupyter style Notebooks: {}".format(file_name_with_extension))


class _GroupedColorFunc(object):
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

       References
       ----------
       https://github.com/amueller/word_cloud/blob/master/examples/colored_by_group.py

    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)


    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(color_func for (color_func, words) in self.color_func_to_words if word in words)
        except StopIteration:
            color_func = self.default_color_func
        return color_func


    def __call__(self, word, **plot_kw):
        return self.get_color_func(word)(word, **plot_kw)


def generate_word_cloud(relevant_feature_wts, pos_clr_name='blue',
                        neg_clr_name='red', threshold=0.1, save_to_file=True, mask_filename=None):
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

    im = Image.open(mask_filename)
    mask_file = np.array(im) if mask_filename is not None else mask_filename
    # TODO extend better support for Word Cloud params
    wc = WordCloud(background_color="white", random_state=0, max_words=len(relevant_feature_wts),
                   mask=mask_file, color_func=grouped_color_func)
    wc.generate_from_frequencies(relevant_feature_wts)
    wc.to_file('word_cloud.png') if save_to_file else None
    return wc


def plot_feature_relevance(feature_relevance_scores, **plot_kw):
    """ Plotting function for visualizing the feature contribution rank ordered wrt +ve/-ve effect

    Parameters
    ----------
    feature_relevance_scores:
    plot_kw:

    Returns
    -------
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on the system."))

    # Validate the input data format. Currently supported types include (pandas.DataFrame)
    DM._check_input(feature_relevance_scores)
    # set the params needed for the plot
    f_name, top_k, color_map, fig_size, font_name, txt_font_size = __set_plot_feature_relevance_keyword(**plot_kw)

    # set the colors
    pos_color = color_map[0]
    neg_color = color_map[1]

    df = feature_relevance_scores.sort_values(by='relevance_scores', ascending=False)
    df['positive'] = df['relevance_scores'] > 0

    # Setting the style globally for the plot
    # For other style check the reference here `plt.style.available`
    plt.clf()
    plt.style.use('bmh')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # filter the top k features wrt each signed category (Positive / Negative ) features
    df_filtered = df.groupby('positive').head(top_k)
    color_list = df_filtered['positive'].map({True: pos_color, False: neg_color})
    df_filtered['relevance_scores'].plot(ax=ax, kind='barh', color=color_list, figsize=fig_size, width=0.85)
    custom_lines = [Patch(facecolor=pos_color, edgecolor='r'), Patch(facecolor=neg_color, edgecolor='b')]

    ax.legend(custom_lines, ['positive', 'negative'], loc='best')
    ax.yaxis.set_visible(False)
    # Display the feature names on the plot
    for index, f_x in enumerate(df_filtered['features']):
        ax.text(0, index + 0.5, f_x, ha='right', fontsize=txt_font_size, fontname=font_name)
    plt.savefig(f_name)
    logger.info("Rank order feature relevance based on input created and saved as {}".format(f_name))
    return f_name


def _render_html(file_name):
    from IPython.core.display import display, HTML
    return HTML(file_name)


def _render_image(file_name):
    from IPython.display import Image
    return Image(file_name)


def show_in_notebook(file_name_with_type='rendered.html'):
    """ Display generated artifacts(e.g. .png, .html, .jpeg/.jpg) in interactive Jupyter style Notebook

    Parameters
    -----------
    file_name_with_type:


    Return
    ------
    """
    file_type = file_name_with_type.split('/')[-1].split('.')[-1]
    choice_dict = {
        'html': _render_html,
        'png': _render_image,
        'jpeg': _render_image,
        'jpg': _render_image
    }
    select_type = lambda choice_type: choice_dict[choice_type]
    return select_type(file_type)(file_name_with_type)
