import itertools as it
import numpy as np

K = 4
H = np.array([[1]])
for _ in range(K):
    H = np.block([[H, H], [H, -H]])

print(H)

# mapping of pair-wise products to new indices
mapping = np.empty(H.shape, dtype=int)
for (i,j) in it.product(range(len(H)), repeat=2):
    mapping[i,j] = (H == (H[i] * H[j])).all(axis=1).argmax()

print(mapping)

xor = np.arange(2**K)[:,None] ^ np.arange(2**K)
print(xor)
assert (xor == mapping).all()


