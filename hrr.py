import numpy as np

def bind(a, v):
    # circular convolution of vectors a and v
    # from https://stackoverflow.com/questions/35474078/python-1d-array-circular-convolution
    return np.real(np.fft.ifft( np.fft.fft(a)*np.fft.fft(v) ))

def unbind(m, a):
    # circular correlation via Plate's involution
    # returns noisy v currently bound to a in memory m
    a_inv = np.roll(a[::-1], 1)
    return bind(a_inv, m)


