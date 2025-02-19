# big test for correctness and timing
from time import perf_counter
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import rcParams
import krop

# config for big check
do_check = True
num_reps = 30
K_max = 15

if do_check:

    runtimes = {"direct": {}, "krop": {}}

    for K in range(1, K_max+1):
        N = 2**K

        # build krop matrix
        print(f"{K=} build...")
        H, thetas = krop.matrix(K)

        # time the cleanup
        print(f"{K=} timing...")
        runtimes["direct"][K] = []
        runtimes["krop"][K] = []

        for rep in range(num_reps):
            u = np.random.randn(N)

            start = perf_counter()
            v_direct = H[(H.T @ u).argmax()]
            duration = perf_counter() - start
            runtimes["direct"][K].append(duration)

            start = perf_counter()
            i = krop.cleanup(thetas, u)
            v_krop = krop.reconstruct(thetas, i)
            duration = perf_counter() - start
            runtimes["krop"][K].append(duration)
            
            assert np.allclose(v_direct, v_krop)

    with open("krop_timing.pkl","wb") as f: pk.dump(runtimes, f)

with open("krop_timing.pkl","rb") as f: runtimes = pk.load(f)

rcParams["font.family"] = "serif"
rcParams["font.size"] = 12
rcParams["text.usetex"] = True

pt.figure(figsize=(8,2.5))
markers = {"krop": "x", "direct": "o"}
means = {"krop": [], "direct": []}
for key, data in runtimes.items():
    for i, (K, results) in enumerate(data.items()):
        pt.plot(K + 0.05*np.random.randn(len(results)), results, markers[key], mec=(.75,)*3, mfc="none", linestyle="none")
        means[key].append(np.mean(results))

for key, data in means.items():
    pt.plot(1+np.arange(K_max), data, markers[key] + "k-", mfc="none", label=key)

pt.legend()
pt.xlabel("$K = \\mathrm{log}_2 N$")
pt.ylabel("Runtime (seconds)")
pt.yscale("log")
pt.tight_layout()
pt.savefig("krop_timing.eps")
pt.show()



