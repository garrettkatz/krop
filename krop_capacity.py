import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
from matplotlib import rcParams
import hrr, krop

do_check = False
K_N_min = 2
K_N_max_direct = 12
K_N_max = 15
num_reps = 30
kinds = ("normal", "binary", "sylvester", "krop")

if do_check:

    success = {kind: {} for kind in kinds}

    for K_N in range(K_N_min, K_N_max+1):
        for K_M in range(min(K_N_max_direct-1, K_N-1)):
            N = 2**K_N
            M = 2**K_M
            print(f"{N=}, {M=}")

            for kind in kinds:
                success[kind][K_N, K_M] = []

            for rep in range(num_reps):

                # sample the keys
                A = np.random.randn(M, N) / N**.5

                # sample the value indices
                idx = np.random.randint(N, size=M)

                # direct methods
                if K_N <= K_N_max_direct:

                    # normal and binary
                    V = {
                        "normal": np.random.randn(N,N) / N**.5,
                        "binary": np.random.choice([-1,1], size=(N,N)) / N**.5,
                    }

                    for kind in V.keys():

                        # write memory
                        mem = np.zeros(N)
                        for i in range(M):
                            mem += hrr.bind(A[i], V[kind][idx[i]])
    
                        # read memory and clean in batches
                        reads = np.stack([hrr.unbind(mem, a) for a in A], axis=-1)
                        idx_clean = (V[kind] @ reads).argmax(axis=0)
    
                        # check success
                        success[kind][K_N, K_M].append( (idx == idx_clean).all() )

                # krop methods
                thetas = {
                    "sylvester": np.ones(K_N) * np.pi/4,
                    "krop": np.linspace(0, 2*np.pi, K_N+2)[1:-1],
                }
                        
                for kind in thetas.keys():

                    # write memory
                    mem = np.zeros(N)
                    for i in range(M):
                        v = krop.reconstruct(thetas[kind], idx[i])
                        mem += hrr.bind(A[i], v)

                    # read memory and clean
                    idx_clean = np.empty(M, dtype=int)
                    for i in range(M):
                        u = hrr.unbind(mem, A[i])
                        idx_clean[i] = krop.cleanup(thetas[kind], u)

                    # check success
                    success[kind][K_N, K_M].append( (idx == idx_clean).all() )

    with open("krop_capacity.pkl","wb") as f: pk.dump(success, f)

with open("krop_capacity.pkl","rb") as f: success = pk.load(f)

K_Ms = (1, 6)

rcParams["font.family"] = "serif"
rcParams["text.usetex"] = True
rcParams["font.size"] = 12
markers = {
    "normal": "+",
    "binary": "x",
    "sylvester": "s",
    "krop": "o",
}

fig, axs = pt.subplots(1, len(K_Ms)+1, figsize=(8,3), constrained_layout=True)

# representative samples
for m, K_M in enumerate(K_Ms):
    M = 2**K_M

    for kind in kinds:
        print(f"kind {kind}...")
        if kind in ("normal", "binary"):
            x = 2**np.array([K_N for K_N in range(K_N_min, K_N_max_direct+1) if (K_N, K_M) in success[kind]])
            y = [np.mean(success[kind][K_N, K_M]) for K_N in range(K_N_min, K_N_max_direct+1) if (K_N, K_M) in success[kind]]
        else:
            x = 2**np.array([K_N for K_N in range(K_N_min, K_N_max+1) if (K_N, K_M) in success[kind]])
            y = [np.mean(success[kind][K_N, K_M]) for K_N in range(K_N_min, K_N_max+1) if (K_N, K_M) in success[kind]]
        axs[m].plot(x, y, color='k', linestyle='-', marker=markers[kind], mfc="none", mec="k", label=kind)

    axs[m].set_title(f"$M = {M}$")
    if m == 0: axs[m].set_ylabel("Success Rate")
    # pt.xlabel("Vector dimension")
    axs[m].set_xscale("log", base=2)
    axs[m].set_ylim([-.1, 1.1])
    if m > 0: axs[m].set_yticks([])
    if m == 0: axs[m].legend()

# overall capacity
for kind in kinds:
    capacities = {}
    for (K_N, K_M), data in success[kind].items():
        if K_N not in capacities and len(data) > 0:
            capacities[K_N] = 0
        if len(data) > 0 and np.all(data):
            capacities[K_N] = max(capacities[K_N], 2**K_M)

    K_Ns = sorted([K_N for K_N in capacities.keys()])
    Ms = sorted([capacities[K_N] for K_N in K_Ns])
    axs[-1].plot(2**np.array(K_Ns), Ms, color='k', linestyle='-', marker=markers[kind], mfc="none", mec="k", label=kind)

axs[-1].set_ylabel("Capacity")
axs[-1].set_xscale("log",base=2)
axs[-1].set_yscale("log",base=2)

fig.supxlabel("$N$")
pt.savefig("krop_success_capacity.eps")
pt.show()



# fig, axs = pt.subplots(1, len(K_Ms), figsize=(5,3), constrained_layout=True)
# for m, K_M in enumerate(K_Ms):
#     M = 2**K_M

#     for kind in kinds:
#         print(f"kind {kind}...")
#         if kind in ("normal", "binary"):
#             x = 2**np.array([K_N for K_N in range(K_N_min, K_N_max_direct+1) if (K_N, K_M) in success[kind]])
#             y = [np.mean(success[kind][K_N, K_M]) for K_N in range(K_N_min, K_N_max_direct+1) if (K_N, K_M) in success[kind]]
#         else:
#             x = 2**np.array([K_N for K_N in range(K_N_min, K_N_max+1) if (K_N, K_M) in success[kind]])
#             y = [np.mean(success[kind][K_N, K_M]) for K_N in range(K_N_min, K_N_max+1) if (K_N, K_M) in success[kind]]
#         axs[m].plot(x, y, color='k', linestyle='-', marker=markers[kind], mfc="none", mec="k", label=kind)

#     axs[m].set_title(f"$M = {M}$")
#     if m == 0: axs[m].set_ylabel("Success Rate")
#     # pt.xlabel("Vector dimension")
#     axs[m].set_xscale("log", base=2)
#     axs[m].set_ylim([-.1, 1.1])
#     if m > 0: axs[m].set_yticks([])
#     if m == 0: axs[m].legend()

# fig.supxlabel("$N$")
# pt.savefig("krop_success.eps")
# pt.show()

# pt.figure(figsize=(3.5,3))
# for kind in kinds:
#     capacities = {}
#     for (K_N, K_M), data in success[kind].items():
#         if K_N not in capacities and len(data) > 0:
#             capacities[K_N] = 0
#         if len(data) > 0 and np.all(data):
#             capacities[K_N] = max(capacities[K_N], 2**K_M)

#     K_Ns = sorted([K_N for K_N in capacities.keys()])
#     Ms = sorted([capacities[K_N] for K_N in K_Ns])
#     pt.plot(2**np.array(K_Ns), Ms, color='k', linestyle='-', marker=markers[kind], mfc="none", mec="k", label=kind)

# pt.xlabel("$N$")
# pt.ylabel("Capacity")
# pt.xscale("log",base=2)
# pt.yscale("log",base=2)
# pt.legend()
# pt.tight_layout()
# pt.savefig("krop_capacity.eps")
# pt.show()
