import pickle as pk
import itertools as it
import matplotlib.pyplot as pt
import numpy as np
import sympy as sp
from sympy.physics.quantum.tensorproduct import TensorProduct

do_sampling = True
sampling = 40
K = 4

# symbolic

th_s = sp.symbols(" ".join([f"t{k}" for k in range(K)]))
if K == 1: th_s = (th_s,)
print(th_s)

H_s = sp.Matrix([[1]])
for k in range(K):
    H_s = TensorProduct(H_s,
        sp.Matrix([[sp.cos(th_s[k]), sp.sin(th_s[k])], [sp.sin(th_s[k]), -sp.cos(th_s[k])]]))

print(" *** H *** \n")
print(repr(H_s))

print("\n *** dots *** \n")
idx = np.arange(2**K)
dots_s = []
for r in range(1, 2**K):
    Hr = H_s[np.roll(idx, r).tolist(), :]
    dots_s.append( H_s @ Hr )

for r, dots in enumerate(dots_s):
    print(r)
    print(repr(dots.reshape(2**(2*K), 1)))

# numerical
if do_sampling:

    H_n = np.empty((sampling,)*K + (2**K, 2**K))
    th_n = np.linspace(0, 2*np.pi, sampling)
    outer_j = -1
    for js in it.product(range(sampling), repeat=K):
        if js[0] != outer_j:
            outer_j = js[0]
            print(f"{outer_j} of {sampling}...")
        H_js = np.array([[1.]])
        for j in js:
            H_js = np.kron(H_js, 
                np.array([[np.cos(th_n[j]), np.sin(th_n[j])], [np.sin(th_n[j]), -np.cos(th_n[j])]]))
        H_n[js] = H_js
    
    dots_n = np.full(H_n.shape, -np.inf)
    for r in range(1, 2**K):
        print(r)
        Hr = H_n[..., np.roll(idx, r).tolist(), :]
        # dots_n[r-1] =  H_n @ Hr # too much space needed
        dots_n = np.maximum(dots_n, np.fabs(H_n @ Hr))

    with open(f"hand_thetas_{K}_{sampling}.pkl", "wb") as f: pk.dump(dots_n, f)

with open(f"hand_thetas_{K}_{sampling}.pkl", "rb") as f: dots_n = pk.load(f)

# worst = np.fabs(dots_n).max(axis=(0, -2, -1))
worst = dots_n.max(axis=(-2, -1))
best = worst.min()
jb = np.unravel_index(worst.argmin(), worst.shape)
print(jb, worst[jb], worst.min())

if K == 2:

    pt.imshow(worst)
    pt.colorbar()
    pt.plot(*jb[::-1], 'ro')
    pt.show()


# idx = np.arange(2**K)
# dots_s = []
# dots_n = np.empty((sampling, sampling, 2**K - 1, 2**K, 2**K))
# th_n = np.linspace(0, 2*np.pi, sampling)
# for r in range(2**K-1):

#     # symbolic
#     Hr = H_s[:, np.roll(idx, r+1).tolist()]
#     dots.append( H_s @ Hr.T )

#     # numerical
#     dots_n[

# H1 = sp.Matrix([[sp.cos(t1), sp.sin(t1)], [sp.sin(t1), -sp.cos(t1)]])
# H2 = sp.Matrix([[sp.cos(t2), sp.sin(t2)], [sp.sin(t2), -sp.cos(t2)]])
# H = TensorProduct(H1, H2)

# theta = np.linspace(0, 2*np.pi, 100)

# tsc = 2*np.sin(theta)*np.cos(theta)
# sdf = np.sin(theta)**2 - np.cos(theta)**2

# print(2*np.sin(np.pi/8)*np.cos(np.pi/8))
# print(np.sin(np.pi/8)**2 - np.cos(np.pi/8)**2)

# # pt.plot(theta, tsc)
# # pt.plot(theta, sdf)
# # pt.plot(theta, -tsc)
# # pt.plot(theta, -sdf)
# # pt.plot(theta, np.maximum(np.maximum(tsc, sdf), np.maximum(-tsc, -sdf)), 'r--')
# # pt.show()

# theta = np.pi/8
# H = np.array([[np.cos(theta), np.sin(theta)],[np.sin(theta),-np.cos(theta)]])
# print(H)

# # v = H[0] + np.roll(H[1], 1)
# # v = H[0] + np.roll(H[0], 1)
# # v = H[1] + np.roll(H[1], 1)
# v = H[1] + np.roll(H[0], 1)
# print(v)
# print(H @ v)
# print(H @ np.roll(v, -1))

# t1, t2 = sp.symbols("t1 t2")
# H1 = sp.Matrix([[sp.cos(t1), sp.sin(t1)], [sp.sin(t1), -sp.cos(t1)]])
# H2 = sp.Matrix([[sp.cos(t2), sp.sin(t2)], [sp.sin(t2), -sp.cos(t2)]])
# H = TensorProduct(H1, H2)

# print(repr(H))
# print(repr(H.subs(t1, 0).subs(t2, 0)))

# Hr = H[:,[1,2,3,0]]

# print(repr(H))
# # print(repr(Hr))
# print(repr(Hr.T))
# print(repr((H @ Hr.T)[:,0]))

# rolls = [
#     [1,2,3,0],
#     [2,3,0,1],
#     [3,0,1,2],
# ]

# sampling = 100
# thetas = np.linspace(0, 2*np.pi, sampling)

# numerical = np.empty((sampling, sampling, len(rolls), 4, 4))

# if do_search:

#     for r, roll in enumerate(rolls):
#         Hr = H[:,roll]
#         HHrT = H @ Hr.T
#         for (i,j) in it.product(range(sampling), repeat=2):
#             print(r,i,j)
#             numerical[i,j,r] = HHrT.subs(t1, thetas[i]).subs(t2, thetas[j])

#     with open(f"hand_thetas_2_{sampling}.pkl", "wb") as f: pk.dump(numerical, f)

# with open(f"hand_thetas_2_{sampling}.pkl", "rb") as f: numerical = pk.load(f)

# # worst = np.fabs(numerical).max(axis=(2,3,4))

# # valid_idxs = [0, 1, 2, 3] # only use these for address/value
# # numerical = numerical[:,:,:,valid_idxs,[valid_idxs]]
# valid_idxs = [0, 2] # only use these for address/value
# numerical = numerical[:,:,:,::2,::2]

# worst = np.fabs(numerical).max(axis=(2,3,4))
# (ib, jb) = np.unravel_index(worst.argmin(), worst.shape)
# print(ib, jb, worst[ib,jb], worst.min())

# print("normalized:")
# print(thetas[ib] / (2*np.pi), thetas[jb] / (2*np.pi))

# pt.imshow(worst)
# pt.colorbar()
# pt.plot(jb, ib, 'ro')
# pt.show()

# for r in range(3):
#     print(r)
#     print(numerical[ib,jb,r])

# # check
# t1, t2 = thetas[[ib, jb]]

# H = np.kron(
#     np.array([[np.cos(t1), np.sin(t1)], [np.sin(t1), -np.cos(t1)]]),
#     np.array([[np.cos(t2), np.sin(t2)], [np.sin(t2), -np.cos(t2)]])
# )

# print(H)

# correct = []
# for rep in range(30):

#     num_mem = 2
#     # a_idx = np.random.choice(range(4), size=num_mem, replace=False)
#     a_idx = np.random.choice(valid_idxs, size=num_mem, replace=False)
#     v_idx = np.random.choice(valid_idxs, size=num_mem)
#     mem = 0
#     for (a_i, v_i) in zip(a_idx, v_idx):
#         mem += np.roll(H[v_i], a_i)
    
#     all_right = True
#     for (a_i, v_i) in zip(a_idx, v_idx):
#         if v_i != (H @ np.roll(mem, -a_i)).argmax(): all_right = False

#     correct.append(all_right)

# print(f"success rate = {np.mean(correct)}")

