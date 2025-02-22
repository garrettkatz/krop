import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
from matplotlib import rcParams
import hrr, krop

do_check = True
K_N_min = 2
K_N_max_direct = 12
K_N_max = 15
K_M_max = 10
num_reps = 10
T = 30
kinds = ("krop", "sign", "none")

if do_check:

    retrieval_rate = {kind: {} for kind in kinds}

    for K_N in range(K_N_min, K_N_max+1):

        thetas = np.linspace(0, 2*np.pi, K_N+2)[1:-1]

        for K_M in range(min(K_M_max+1, K_N-1)):
            N = 2**K_N
            M = 2**K_M
            print(f"{N=}, {M=}")

            for rep in range(num_reps):

                for kind in kinds:
                    retrieval_rate[kind][K_N, K_M, rep] = []

                A = np.random.randn(M,N) / N**.5
                if K_N <= K_N_max_direct:
                    V = {
                        "none": np.random.randn(N,N) / N**.5,
                        "sign": np.random.choice([-1, 1], size=(N,N)) / N**.5,
                    }

                # random initial contents
                mem = {kind: np.zeros(N) for kind in kinds}
                ref_memory = np.random.randint(N, size=M)
                for m, i in enumerate(ref_memory):
                    mem["krop"] += hrr.bind(A[m], krop.reconstruct(thetas, i))
                    if K_N <= K_N_max_direct:
                        mem["sign"] += hrr.bind(A[m], V["sign"][i])
                        mem["none"] += hrr.bind(A[m], V["none"][i])

                # overwrites and retrievals
                for t in range(T):

                    # overwrites
                    m = np.random.randint(M)
                    i = np.random.randint(N)
                    ref_memory[m] = i
                
                    # krop
                    u = hrr.unbind(mem["krop"], A[m])
                    v_old = krop.reconstruct(thetas, krop.cleanup(thetas, u))
                    v_new = krop.reconstruct(thetas, i)
                    mem["krop"] += hrr.bind(A[m], v_new) - hrr.bind(A[m], v_old)

                    if K_N <= K_N_max_direct:

                        # sign
                        u = hrr.unbind(mem["sign"], A[m])
                        v_old = np.sign(u) / N**.5
                        mem["sign"] += hrr.bind(A[m], V["sign"][i]) - hrr.bind(A[m], v_old)
                    
                        # none
                        v_old = hrr.unbind(mem["none"], A[m])
                        mem["none"] += hrr.bind(A[m], V["none"][i]) - hrr.bind(A[m], v_old)

                    # retrievals

                    # krop
                    retrieved = []
                    for m, i in enumerate(ref_memory):
                        u = hrr.unbind(mem["krop"], A[m])
                        j = krop.cleanup(thetas, u)
                        retrieved.append( i == j )
                    retrieval_rate["krop"][K_N, K_M, rep].append(np.mean(retrieved))

                    # sign and none
                    if K_N <= K_N_max_direct:

                        for kind in ("sign", "none"):
                            reads = np.stack([hrr.unbind(mem[kind], a) for a in A], axis=-1)
                            idx_clean = (V[kind] @ reads).argmax(axis=0)
                            retrieval_rate[kind][K_N, K_M, rep].append(np.mean(ref_memory == idx_clean))

    with open("krop_mutable.pkl","wb") as f: pk.dump(retrieval_rate, f)

with open("krop_mutable.pkl","rb") as f: retrieval_rate = pk.load(f)

rcParams["font.family"] = "serif"
rcParams["text.usetex"] = True
rcParams["font.size"] = 12
markers = {
    "none": "+",
    "sign": "x",
    "krop": "o",
}

K_Ms, K_Ns = (3,3,8), (8,10,K_N_max)
fig, axs = pt.subplots(1, len(K_Ms), figsize=(8,2.5), constrained_layout=True)
for m, (K_M, K_N) in enumerate(zip(K_Ms, K_Ns)):
    N = 2**K_N
    M = 2**K_M
    rates = {
        kind: np.array([retrieval_rate[kind][K_N, K_M, rep] for rep in range(num_reps) if (K_N, K_M, rep) in retrieval_rate[kind]])
        for kind in kinds}

    print(N,M)
    print(rates)
    
    for kind in kinds:
        axs[m].plot(rates[kind].mean(axis=0), 'k-' + markers[kind], mfc="none", label=kind)
    if m == 0:
        axs[m].set_ylabel("Retrieval Rate")
        axs[m].legend(loc="right")
    else:
        axs[m].set_yticks([])
    axs[m].set_ylim([-.1,1.1])
    axs[m].set_title(f"M={M}, N={N}")
fig.supxlabel("Time-step")
pt.savefig("krop_mutable_sample.eps")
pt.show()

# K_Ms = np.linspace(0, K_N_max_direct-2, 4).round().astype(int)
# K_Ms = tuple(range(5,9))
K_Ms = tuple(range(0,K_N_max-1))
K_Ns = np.arange(K_N_min, K_N_max+1)

fig, axs = pt.subplots(1, len(K_Ms), figsize=(8,2.5), constrained_layout=True)
for m, K_M in enumerate(K_Ms):
    M = 2**K_M
    rates = {
        kind: np.array([
            np.mean([retrieval_rate[kind].get((K_N, K_M, rep), np.nan) for rep in range(num_reps)])
            for K_N in K_Ns])
        for kind in kinds}

    for kind in kinds:
        axs[m].plot(2**np.array(K_Ns), rates[kind], 'k-' + markers[kind], mfc="none", label=kind)
    axs[m].set_xscale("log", base=2)
    axs[m].set_ylim([-.1,1.1])
    axs[m].set_title(f"$M = {M}$")
    if m == 0:
        axs[m].set_ylabel("Time-Averaged Retrieval Rate")
        axs[m].legend()
    else:
        axs[m].set_yticks([])
fig.supxlabel("$N$")
pt.savefig("krop_mutable.eps")
pt.show()

