import itertools as it
import numpy as np
np.set_printoptions(linewidth=1000)

# K = 2
# N = 2**K
N = 8
M = 3

vertices = np.array(list(it.product((+1,-1), repeat=N)))

print(vertices)

# find all size-M subsets with off-diagonal products at parity N/2
equis = {}
for idx in it.combinations(range(len(vertices)), r=M):

    print('idx, V, prods')
    print(idx)

    print(vertices[list(idx)])

    prods = np.stack([
        vertices[i] * vertices[j]
        for (i,j) in it.combinations(idx, r=2)])

    print(prods)

    if (prods.sum(axis=1) == 0).all():
        equis[idx] = prods

print(f"{len(equis)} subsets at parity")
input('..')

# check if any pairs of subsets have their quaducts all at parity N/2
solns = {}
for (ab, uv) in it.combinations(equis.keys(), r=2):
    print(ab, uv)
    quads = np.stack([x*y for (x,y) in it.product(equis[ab], equis[uv])])
    if (quads.sum(axis=1) == 0).all():
        solns[ab, uv] = quads
        print('soln: ab, uv, V[ab], V[uv], P[ab], P[uv], quads')
        print(ab)
        print(uv)
        print(vertices[list(ab)])
        print(vertices[list(uv)])
        print(equis[ab])
        print(equis[uv])
        print(quads)
        input("soln found!...")

print(f"{len(solns)} solutions found")
