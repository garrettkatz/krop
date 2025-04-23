import pickle as pk
import itertools as it
import matplotlib.pyplot as pt
import numpy as np
import sympy as sp
from sympy.physics.quantum.tensorproduct import TensorProduct

do_sampling = False
sampling = 30
K = 3

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
th_n = np.linspace(0, 2*np.pi, sampling)
if do_sampling:

    H_n = np.empty((sampling,)*K + (2**K, 2**K))
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
    
    dots_n = np.full((2**K-1,) + H_n.shape, -np.inf)
    # dots_n = np.full(H_n.shape, -np.inf)
    for r in range(1, 2**K):
        print(r)
        Hr = H_n[..., np.roll(idx, r).tolist(), :]
        dots_n[r-1] =  H_n @ Hr # uses more space needed
        # dots_n = np.maximum(dots_n, np.fabs(H_n @ Hr)) # to save space

    with open(f"hand_thetas_{K}_{sampling}.pkl", "wb") as f: pk.dump(dots_n, f)

with open(f"hand_thetas_{K}_{sampling}.pkl", "rb") as f: dots_n = pk.load(f)

# K = 2, looking for M capacity with optimal choice of rolls and embeddings
# dots_n[roll, th0, th1, i, j] is dot between H[i] and roll(H[j])
# need |H[i] @ roll(H[j])| < 1/(2*(M-1)) for optimal choice of roll and i,j

M = 3

found_feas = False
for r in it.combinations(range(2**K - 1), r=M-1):
    for idx in it.combinations(range(2**K), r=M):
        idx = np.array(idx)
        dots_nr = np.fabs(dots_n)[list(r)].max(axis=0)
        dots_ridx = dots_nr[...,idx[:,None],idx]
        # print(r, idx, dots_ridx.shape)
        # input('.')
        # dots_ridx = dots_nr[...,:,idx] # cleanup will check with all if you let it
        worst = dots_ridx.max(axis=(-2, -1))
        best = worst.min()
        if best < 0.5 / (M-1):
            jb = np.unravel_index(worst.argmin(), worst.shape)
            valid_idxs = idx
            valid_roll = [0] + [_+1 for _ in r]
            print(f"roll {valid_roll} idx {valid_idxs} jb {jb} best {best}")
            print(f"thetas {th_n[list(jb)]}")

            # pt.imshow(dots_n[(r,)+jb])
            print(dots_n.shape, dots_ridx.shape, worst.shape)
            pt.imshow(worst[jb[:-2]])
            pt.plot(jb[-1], jb[-2], 'ro')
            pt.colorbar()
            pt.show()

            found_feas = True

        if found_feas: break
    if found_feas: break


# check
th = th_n[list(jb)]
H = np.array([[1.]])
for th_j in th:
    H = np.kron(H, 
        np.array([[np.cos(th_j), np.sin(th_j)], [np.sin(th_j), -np.cos(th_j)]]))

print(H)

correct = []
for rep in range(30):

    v_idx = np.random.choice(valid_idxs, size=M)
    mem = 0
    for (r, v_i) in enumerate(v_idx):
        mem += np.roll(H[v_i], valid_roll[r])

    print('v_idx', v_idx)
    print('mem', mem)

    all_right = True
    for (r, v_i) in enumerate(v_idx):
        idx_i = (H[valid_idxs,:] @ np.roll(mem, -valid_roll[r])).argmax()
        if v_i != valid_idxs[idx_i]:
            all_right = False
            print('break roll v_i v_i\'', valid_roll[r], v_i, valid_idxs[idx_i])
            break

    print(all_right)
    input('.')

    correct.append(all_right)

print(f"success rate = {np.mean(correct)}")

