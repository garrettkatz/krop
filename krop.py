import numpy as np

def matrix(K):
    H = np.array([[1]])
    thetas = np.linspace(0, 2*np.pi, K+2)[1:-1]
    for theta in thetas:
        c, s = np.cos(theta), np.sin(theta)
        H = np.block([[c*H, s*H], [s*H, -c*H]])

    assert np.allclose(H.T @ H, np.eye(len(H)))
    return H, thetas

def reconstruct(thetas, idx):
    v = np.array([1.])
    for k, t in enumerate(thetas):
        kth_bit = (idx >> k) & 1
        if kth_bit == 0:
            v = np.concatenate([np.cos(t) * v,  np.sin(t) * v])
        else:
            v = np.concatenate([np.sin(t) * v, -np.cos(t) * v])
    return v

def cleanup(thetas, u):

    # decompose into sub-problems
    X = u.copy()
    for d, theta in enumerate(reversed(thetas)):
        c, s = np.cos(theta), np.sin(theta)
        X = X.reshape((2**(d+1), -1))
        x_top, x_bot = X[::2], X[1::2]
        X[::2], X[1::2] = c*x_top + s*x_bot, s*x_top - c*x_bot

    # return i*
    return np.argmax(X.flat)


