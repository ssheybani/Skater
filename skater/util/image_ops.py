# -*- coding: UTF-8 -*-

import numpy as np
import skimage
import skimage.io
from .exceptions import MatplotlibUnavailableError


def load_image(path, img_height, img_width):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()

    # Crop image from the center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]

    # Re-size the image to required dimension
    resized_img = skimage.transform.resize(crop_img, (img_width, img_height))
    return resized_img


def noisy(image, noise_typ='gaussian', random_state=None):
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


def normalize(X):
    """ Normalize image of the shape (H, W, D) in the range of 0 and 1
    """
    return np.array((X - np.min(X)) / (np.max(X) - np.min(X)))


def show_image(X):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
            raise (MatplotlibUnavailableError("Matplotlib is required but unavailable on your system."))
    plt.imshow(X)
