"""Circular convolution and involution."""
import numpy as np


def generate(size, norm=False):
    """Generates normally distributed random vectors of size."""
    if isinstance(size, (int, float)):
        size = (size,)
    if len(size) == 1:
        size = (1, size[0])
    vecs = np.random.normal(0, size=size, scale=1/size[1])
    if norm:
        vecs /= np.linalg.norm(vecs, axis=1)[:, None]
    return vecs


def addition(p):
    return np.sum(p, 0)


def circular_convolution(x, y):
    """A fast version of the circular convolution."""
    # Stolen from:
    # http://www.indiana.edu/~clcl/holoword/Site/__files/holoword.py
    z = np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)).real
    if np.ndim(z) == 1:
        z = z[None, :]
    return z


def involution(x):
    """Involution operator."""
    if np.ndim(x) == 1:
        x = x[None, :]
    return np.concatenate([x[:, None, 0], x[:, -1:0:-1]], 1)


def circular_correlation(x, y):
    """Circular correlation is the inverse of circular convolution."""
    return circular_convolution(involution(x), y)


def decode(x, y):
    """Simple renaming."""
    return circular_correlation(x, y)
