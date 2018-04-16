# -*- coding: UTF-8 -*-

import numpy as np
import skimage
import skimage.io

from .exceptions import MatplotlibUnavailableError
from skater.util.logger import build_logger
from skater.util.logger import _INFO


logger = build_logger(_INFO, __name__)


def load_image(path, img_height, img_width):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from the center
    short_edge = np.min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]

    # Re-size the image to required dimension
    resized_img = skimage.transform.resize(crop_img, (img_width, img_height))
    return resized_img


def add_noise(image, noise_typ='gaussian', random_state=None):
    """ Helper function to add different forms of noise

    Parameters
    -----------
    image: numpy.ndarray
        input image
    noise_typ: string ( default 'gaussian' )
        - 'gaussian'  Gaussian-distributed additive noise.
        - 'localvar'  Gaussian-distributed additive noise, with specified
                      local variance at each point of `image`
        - 'poisson'   Poisson-distributed noise generated from the data.
        - 'salt'      Replaces random pixels with 1.
        - 'pepper'    Replaces random pixels with 0 (for unsigned images) or
                      -1 (for signed images).
        - 's&p'       Replaces random pixels with either 1 or `low_val`, where
                      `low_val` is 0 for unsigned images or -1 for signed
                      images.
        - 'speckle'   Multiplicative noise using out = image + n*image, where
                      n is uniform noise with specified mean & variance.
    random_state: int
        seed for setting repeatable state before generating noise

    Returns
    -------
    numpy.ndarray

    References
    ----------
    .. [1] http://scikit-image.org/docs/stable/api/skimage.util.html#random-noise
    """
    return skimage.util.random_noise(image, mode=noise_typ, seed=random_state)


def image_transformation():
    # https://www.kaggle.com/tomahim/image-manipulation-augmentation-with-skimage
    pass


# Helper functions for filtering based on conditional type
greater_than = lambda X, value: np.where(X > value)
less_than = lambda X, value: np.where(X < value)
in_between = lambda X, min_value, max_value: np.where((X >= min_value) & (X <= max_value))


def remove_pixels(X, num_of_pixel, filtered_pixel=None):
    try:
        if len(filtered_pixel) > 0 & isinstance(filtered_pixel, tuple):
            f_pixels = filtered_pixel
            logger.info("Number of pixels matching the condition : {}".format(len(f_pixels[0])))
            logger.info("Number of pixels specified to be replaced : {}".format(num_of_pixel))

            if len(f_pixels) == 3:
                # uniformly random
                h = np.random.choice(f_pixels[0], num_of_pixel)
                w = np.random.choice(f_pixels[1], num_of_pixel)
                c = np.random.choice(f_pixels[2], num_of_pixel)

                # for the selected pixels, set the pixel intensity to 0
                for h_i, w_i, c_i in zip(h, w, c):
                    X[h_i, w_i, c_i] = 0
            elif len(f_pixels) == 2:
                # uniformly random
                h = np.random.choice(f_pixels[0], num_of_pixel)
                w = np.random.choice(f_pixels[1], num_of_pixel)

                # for the selected pixels, set the pixel intensity to 0
                for h_i, w_i, c_i in zip(h, w):
                    X[h_i, w_i] = 0
            else:
                logger.info("Ambiguity in the shape of the input image : {}".format(X.shape))
    except:
        raise ValueError("No matching pixel for the specified condition")
    return X


def normalize(X):
    """ Normalize image of the shape (H, W, D) in the range of 0 and 1
    """
    return np.array((X - np.min(X)) / (np.max(X) - np.min(X)))


def show_image(X, axis=None, cmap=None, bins=None, stats=False):
    try:
        import matplotlib.pyplot as plt
        axis = plt if axis is None else axis
    except ImportError:
            raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))
    axis.imshow(X, cmap=cmap)
    bins = X.shape[1] if None else bins
    axis.hist(X.ravel(), bins=bins, histtype='step') if stats else None
