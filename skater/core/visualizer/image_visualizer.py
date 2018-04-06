from skimage.filters import roberts, sobel
import numpy as np
import matplotlib.pyplot as plt

from skater.util.image_ops import normalize


def visualize(relevance_score, original_input_img=None, edge_detector_alg='robert', cmap='bwr', axis=plt,
              percentile=100, alpha=0.8):
    # Normalize the input image to (0,1)
    xi = normalize(original_input_img)

    dx, dy = 0.01, 0.01
    xx = np.arange(0.0, relevance_score.shape[1], dx)
    yy = np.arange(0.0, relevance_score.shape[0], dy)

    x_min, x_max, y_min, y_max = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = x_min, x_max, y_min, y_max
    xi_cmap = plt.cm.gray
    xi_cmap.set_bad(alpha=0)

    if xi is not None:
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        # Applying edge detection ( Roberts or Sobel edge detection )
        # Reference: http://scikit-image.org/docs/0.11.x/auto_examples/plot_edge_filter.html
        edge_detector = {'robert': roberts, 'sobel': sobel}
        edges = edge_detector[edge_detector_alg](xi_greyscale)

    abs_max = np.percentile(np.abs(relevance_score), percentile)
    abs_min = abs_max

    relevance_score = relevance_score[:, :, 0] if len(relevance_score.shape) == 3 else relevance_score
    axis.imshow(relevance_score, extent=extent, interpolation='nearest', cmap=cmap, vmin=-abs_min, vmax=abs_max)

    # draw the edges of the image before overlaying the learned relevance
    if edges is not None:
        axis.imshow(edges, extent=extent, interpolation='nearest', cmap=xi_cmap, alpha=alpha)
    axis.axis('off')
    return axis