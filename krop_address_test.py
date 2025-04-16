import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
from matplotlib import rcParams
import hrr, krop

do_check = True
K_N_min = 2
K_N_max_direct = 12
K_N_max = 15
num_reps = 30

if do_check:

    success = {}

    for K_N in range(K_N_min, K_N_max+1):
        for K_M in range(min(K_N_max_direct-1, K_N-1)):
            N = 2**K_N
            M = 2**K_M
            print(f"{N=}, {M=}")

            success[K_N, K_M] = []

            for rep in range(num_reps):

                # sample the address and value indices
                add_idx = np.random.choice(range(N), size=M, replace=False)
                val_idx = np.random.randint(N, size=M)

                # sample the thetas
                add_thetas = np.random.uniform(0, 2*np.pi, K_N)
                val_thetas = np.random.uniform(0, 2*np.pi, K_N)
                        
                # write memory
                mem = np.zeros(N)
                for i in range(M):
                    a = krop.reconstruct(add_thetas, add_idx[i])
                    v = krop.reconstruct(val_thetas, val_idx[i])
                    mem += hrr.bind(a, v)

                # read memory and clean
                idx_clean = np.empty(M, dtype=int)
                for i in range(M):
                    a = krop.reconstruct(add_thetas, add_idx[i])
                    u = hrr.unbind(mem, a)
                    idx_clean[i] = krop.cleanup(val_thetas, u)

                # check success
                success[K_N, K_M].append( (val_idx == idx_clean).all() )

    with open("krop_address.pkl","wb") as f: pk.dump(success, f)

with open("krop_address.pkl","rb") as f: success = pk.load(f)

K_Ms = (1, 6)

rcParams["font.family"] = "serif"
rcParams["text.usetex"] = True
rcParams["font.size"] = 12

fig, axs = pt.subplots(1, len(K_Ms)+1, figsize=(8,3), constrained_layout=True)

# representative samples
for m, K_M in enumerate(K_Ms):
    M = 2**K_M

    x = 2**np.array([K_N for K_N in range(K_N_min, K_N_max+1) if (K_N, K_M) in success])
    y = [np.mean(success[K_N, K_M]) for K_N in range(K_N_min, K_N_max+1) if (K_N, K_M) in success]
    axs[m].plot(x, y, color='k', linestyle='-',marker='o', mfc="none", mec="k")

    axs[m].set_title(f"$M = {M}$")
    if m == 0: axs[m].set_ylabel("Success Rate")
    # pt.xlabel("Vector dimension")
    axs[m].set_xscale("log", base=2)
    axs[m].set_ylim([-.1, 1.1])
    if m > 0: axs[m].set_yticks([])
    # if m == 0: axs[m].legend()

# overall capacity
capacities = {}
for (K_N, K_M), data in success.items():
    if K_N not in capacities and len(data) > 0:
        capacities[K_N] = 0
    if len(data) > 0 and np.all(data):
        capacities[K_N] = max(capacities[K_N], 2**K_M)

K_Ns = sorted([K_N for K_N in capacities.keys()])
Ms = sorted([capacities[K_N] for K_N in K_Ns])
axs[-1].plot(2**np.array(K_Ns), Ms, color='k', linestyle='-', marker='o', mfc="none", mec="k")

axs[-1].set_ylabel("Capacity")
axs[-1].set_xscale("log",base=2)
axs[-1].set_yscale("log",base=2)

fig.supxlabel("$N$")
pt.savefig("krop_address_capacity.eps")
pt.show()

