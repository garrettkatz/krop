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
T = 30
kinds = ("krop", "sign", "none")

if do_check:

    success_rate = {kind: {} for kind in kinds}

    for K_N in range(K_N_min, K_N_max+1):

        thetas = np.linspace(0, 2*np.pi, K_N+2)[1:-1]

        for K_M in range(min(K_N_max_direct-1, K_N-1)):
            N = 2**K_N
            M = 2**K_M
            print(f"{N=}, {M=}")

            for rep in range(num_reps):

                for kind in kinds:
                    success_rate[kind][K_N, K_M, rep] = []

                A = np.random.randn(M,N) / N**.5
                if K_N <= K_N_max_direct:
                    V = {
                        "none": np.random.randn(N,N) / N**.5,
                        "sign": np.random.choice([-1, 1], size=(N,N)) / N**.5,
                    }

                ref_memory = np.zeros(M, dtype=int)
                mem = {kind: np.zeros(N) for kind in kinds}

                # initial contents
                for m in range(M):
                    mem["krop"] += hrr.bind(A[m], krop.reconstruct(thetas, 0))
                    if K_N <= K_N_max_direct:
                        mem["sign"] += hrr.bind(A[m], V["sign"][0])
                        mem["none"] += hrr.bind(A[m], V["none"][0])

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
                    success = []
                    for m, i in enumerate(ref_memory):
                        u = hrr.unbind(mem["krop"], A[m])
                        j = krop.cleanup(thetas, u)
                        success.append( i == j )
                    success_rate["krop"][K_N, K_M, rep].append(np.mean(success))

                    # sign and none
                    if K_N <= K_N_max_direct:

                        for kind in ("sign", "none"):
                            success = []
                            for m, i in enumerate(ref_memory):
                                u = hrr.unbind(mem[kind], A[m])
                                j = (V[kind] @ u).argmax()
                                success.append( i == j )
                            success_rate[kind][K_N, K_M, rep].append(np.mean(success))

    with open("krop_mutable.pkl","wb") as f: pk.dump(success_rate, f)

with open("krop_mutable.pkl","rb") as f: success_rate = pk.load(f)

rcParams["font.family"] = "serif"
rcParams["text.usetex"] = True
rcParams["font.size"] = 12
markers = {
    "none": "+",
    "sign": "x",
    "krop": "o",
}

K_M, K_N = 3, 8
N = 2**K_N
M = 2**K_M
rates = {
    kind: np.array([success_rate[kind][K_N, K_M, rep] for rep in range(num_reps)])
    for kind in kinds}

pt.figure(figsize=(5,3))
for kind in kinds:
    pt.plot(rates[kind].mean(axis=0), 'k-' + markers[kind], mfc="none", label=kind)
pt.xlabel("Time-step")
pt.ylabel("Retrieval Rate")
pt.legend(loc="right")
pt.tight_layout()
pt.savefig("krop_mutable_sample.eps")
pt.show()

K_Ms = np.linspace(0, K_N_max_direct-2, 4).round().astype(int)
K_Ns = np.arange(K_N_min, K_N_max+1)

fig, axs = pt.subplots(1, len(K_Ms), figsize=(10,3), constrained_layout=True)
for m, K_M in enumerate(K_Ms):
    M = 2**K_M
    rates = {
        kind: np.array([
            np.mean([success_rate[kind].get((K_N, K_M, rep), np.nan) for rep in range(num_reps)])
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
fig.supxlabel("$N$")
pt.savefig("krop_mutable.eps")
pt.show()

