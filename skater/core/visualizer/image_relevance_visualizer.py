# -*- coding: UTF-8 -*-

from skimage.filters import roberts, sobel
import numpy as np
from skater.util.exceptions import MatplotlibUnavailableError, MatplotlibDisplayError, KerasUnavailableError
from skater.util.image_ops import normalize
from sklearn.preprocessing import MinMaxScaler

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise MatplotlibUnavailableError("Matplotlib is required but unavailable on your system.")
except RuntimeError:
    raise (MatplotlibDisplayError("Matplotlib unable to open display"))

try:
    from keras.models import Model
except ImportError:
    raise KerasUnavailableError("Keras binaries are not installed")


# helper function to enable or disable matplotlib access
_enable_axis = lambda ax, flag: ax.axis("off") if flag is True else ax.axis("on")


def visualize(relevance_score, original_input_img=None, edge_detector_type='sobel', cmap='bwr', axis=plt,
              percentile=100, alpha_edges=0.8, alpha_bgcolor=1, disable_axis=True):

    dx, dy = 0.01, 0.01
    xx = np.arange(0.0, relevance_score.shape[1], dx)
    yy = np.arange(0.0, relevance_score.shape[0], dy)

    x_min, x_max, y_min, y_max = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = x_min, x_max, y_min, y_max
    xi_cmap = plt.cm.gray
    xi_cmap.set_bad(alpha=0)

    edges = _edge_detection(original_input_img, edge_detector_type) if original_input_img is not None else None

    # draw the edges of the image before overlaying rest of the image
    if edges is not None:
        axis.imshow(edges, extent=extent, interpolation='nearest', cmap=xi_cmap, alpha=alpha_edges)

    abs_max = np.percentile(np.abs(relevance_score), percentile)
    abs_min = abs_max

    relevance_score = relevance_score[:, :, 0] if len(relevance_score.shape) == 3 else relevance_score
    # Plot the image with relevance scores
    axis.imshow(relevance_score, extent=extent, interpolation='nearest', cmap=cmap,
                vmin=-abs_min, vmax=abs_max, alpha=alpha_bgcolor)
    _enable_axis(axis, disable_axis)
    return axis


def _edge_detection(original_input_img=None, edge_detector_alg='sobel'):
    # Normalize the input image to (0,1)
    xi = normalize(original_input_img)
    xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
    # Applying edge detection ( Roberts or Sobel edge detection )
    # Reference: http://scikit-image.org/docs/0.11.x/auto_examples/plot_edge_filter.html
    edge_detector = {'robert': roberts, 'sobel': sobel}
    return edge_detector[edge_detector_alg](xi_greyscale)


def visualize_feature_maps(model_inst, X, layer_name=None,
                           precomputed_feature_map=None, n_filters=16,
                           plt_height=20, plt_width=20, fig_bg_color='darkgrey', **plot_kwargs):
    # reference: https://matplotlib.org/2.0.0/examples/color/named_colors.html
    framework_type = 'keras' if 'keras' in str(type(model_inst)) else None
    model_class = Model(inputs=model_inst.input, outputs=model_inst.get_layer(layer_name).output) \
        if framework_type == 'keras' else None
    feature_maps = model_class.predict(X)[0] if model_class is not None else precomputed_feature_map

    if feature_maps is None:
        raise Exception("All option to compute feature map failed")

    if len(feature_maps.shape) != 3:
        raise Exception("In-correct shape of the feature map used. "
                        "Feature map should be of the form (height, weight, depth) "
                        "in that order to get accurate results")

    _, _, depth = feature_maps.shape if n_filters is 'all' else (0, 0, n_filters)
    n_plts_per_row = np.rint(np.sqrt(depth))
    n_rows, n_cols = n_plts_per_row, n_plts_per_row
    fig = plt.figure(figsize=(plt_height, plt_width))
    fig.set_facecolor(fig_bg_color)
    for index in range(depth):
        ax = plt.subplot(n_rows, n_cols, index + 1, **plot_kwargs)
        # normalize the weights to be in the range (0, 1)
        fm = MinMaxScaler().fit_transform(feature_maps[:, :, index])
        ax.imshow(fm, cmap='bwr')
        ax.set_title('filter {}'.format(index + 1))
    return plt, fig
