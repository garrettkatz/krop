import itertools as it
import numpy as np
np.set_printoptions(linewidth=1000)
# import cvxpy as cp
# from pyscipopt import Model, quicksum

K = 3
N = 2**K
M = 3 # 2**(K-1)
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

V = H[:M]
vertices = np.array([v for v in it.product((+1,-1), repeat=N)])

# recursive approach
def rec(sprout):
    if len(sprout) == M: return sprout
    best = sprout
    for v in range(sprout[-1]+1, 2**N):
        if len(sprout) < 3:
            print(" "*len(sprout) + str(sprout + (v,)))
        Ab = vertices[list(sprout)] * vertices[v]
        dots = V @ Ab.T
        if not (dots == 0).all(): continue
        result = rec(sprout + (v,))
        if len(result) > len(best): best = result
        if len(result) == M: break
    return best
    
for v in range(2**N - M):
    print(f"{v} of {2**N - M}...")
    sprout = rec((v,))
    if len(sprout) == M: break


# found = False
# sprouts = {(i,): True for i in range(2**N)} # sprout: feasible
# for m in range(2, M+1):
#     print(f" sprouting m = {m} of {M}, {len(sprouts)} sprouts...")
#     new_sprouts = {}
#     for s, sprout in enumerate(sprouts):
#         print(f"{s} of {len(sprouts)}: growing sprout {sprout}")
#         for v in range(sprout[-1]+1, 2**N):
#             new_sprout = sprout + (v,)
#             new_sprouts[new_sprout] = True

#             for (i,j) in it.combinations(new_sprout, r=2):
#                 dots = V @ (vertices[i] * vertices[j])
#                 if not (np.fabs(dots) == 0).all():
#                     new_sprouts[new_sprout] = False
#                     break

#             if m == M and new_sprouts[new_sprout]:
#                 found = True
#                 break
        
#         if found: break

#     sprouts = new_sprouts

# sprouts = {(0, 15, 102): True} # M = 3


# all_v = frozenset(range(2**N))
# sprouts = {frozenset([i]): True for i in range(2**N)} # sprout: feasible

# found = False
# for m in range(2, M+1):
#     print(f" sprouting m = {m} of {M}, {len(sprouts)} sprouts...")
#     new_sprouts = {}
#     for s, sprout in enumerate(sprouts):
#         print(f"{s} of {len(sprouts)}: growing sprout {sprout}")
#         for v in all_v - sprout:
#             new_sprout = sprout | frozenset([v])
#             if new_sprout in new_sprouts: continue
#             new_sprouts[new_sprout] = True

#             # A = vertices[sorted(new_sprout)]
#             # A2 = (A[:,None,:] * A[None,:,:]).reshape(-1, N)
#             # dots = V @ A2.T
#             # if not (np.fabs(dots) == 0).all():
#             #     new_sprouts[new_sprout] = False

#             # A2 = np.stack([vertices[i] * vertices[j]
#             #     for (i,j) in it.combinations(new_sprout, r=2)])
#             # dots = V @ A2.T
#             # if not (np.fabs(dots) == 0).all():
#             #     new_sprouts[new_sprout] = False

#             for (i,j) in it.combinations(new_sprout, r=2):
#                 dots = V @ (vertices[i] * vertices[j])
#                 if not (np.fabs(dots) == 0).all():
#                     new_sprouts[new_sprout] = False
#                     break

#             if m == M and new_sprouts[new_sprout]:
#                 found = True
#                 break
        
#         if found: break

#     sprouts = new_sprouts

# # sprouts = {frozenset([0, 15, 102]): True} # M = 3

# sprouts = [sprout for (sprout, feasible) in sprouts.items() if feasible]
# print(f"{len(sprouts)} feasible sprouts")

# sprout = sprouts[0]
# print(f"first sprout = {sprout}")

print(f"best sprout = {sprout}")

# hand-check
A = vertices[sorted(sprout)]
print("A:")
print(A)
print("A**2:")
A2 = np.stack([a * b for (a,b) in it.combinations_with_replacement(A, r=2)])
print(A2)
# print((A[:,None,:] * A[None,:,:]).reshape(-1, N))
print("V:")
print(V)

for (a,b) in it.combinations(A, r=2):
    for v in V:
        assert (a*b) @ v == 0

print("check passed.")

# A2 = H[2**(K-1):]
# A = np.where(A2 == 1, 0-1j, 0+1j)
# B = np.where(A2 == 1, 0-1j, 0+1j)
# AB = np.real(A[:,None,:] * B[None,:,:]).astype(int)
# # print(np.real(AB.reshape(-1, 2**K)).astype(int))
# for ab in AB: print(ab)
# print(A2)

# # brute force search for M=3 addresses and values
# M = 5
# input(f"M={M} requires N = M log2 M = {M * np.log2(M)}, you have N = 2**{K} = {2**K}...")
# found = False
# # for a in map(np.array, it.combinations(range(2**K), r=M)):
# #     print(a)
# #     for v in map(np.array, it.combinations(range(2**K), r=M)):
# # combos = list(it.combinations(range(2**K), r=M))
# # for i, a in enumerate(map(np.array, combos)):
# for a in map(np.array, it.combinations(range(2**K), r=M)):
#     print(a)
#     leftover = set(range(2**K)) - set(a) | set([0])
#     # for j in range(i+1, len(combos)):
#         # v = np.array(combos[j])
#     for v in map(np.array, it.combinations(sorted(leftover), r=M)):
#         Ma = mapping[a[:,None], a]
#         Mv = mapping[v[:,None], v]
#         inter = set(Ma.flat) & set(Mv.flat)
#         if inter != set([0]): continue
#         found = True
#         print("a,v:", a, v)
#         print("Ma;Mv:")
#         print(Ma)
#         print(Mv)
#         print("inter:")
#         print(inter)

#         if found: break
#     if found: break

# con = np.arange(2**K)[:,None] & np.arange(2**K)
# print(con)

# mx = np.empty(H.shape, dtype=int)
# for (i,j) in it.product(range(len(H)), repeat=2):
#     mx[i,j] = (H == (H[i] * H[j])).all(axis=1).argmax()

# for r in range(1,2**(K-1)):
#     Hr = np.roll(H, -r)
#     print(Hr)
    
#     abHr = H[:2**(K-1),None,:] * Hr[None,:,:] # A2 x V x row
#     abHrHrT = abHr @ Hr.T # A2 x V x V
#     mx = np.fabs(abHrHrT).max(axis=0) # V x V
    
#     print(mx)    
#     idx = np.array([0,3,4])
#     print(mx[idx[:,None], idx])
    
# odds = H[1::2, :]
# print('odds:')
# print(odds)
# print('even times odd')
# print(mapping[1::2,::2])

# # prods = (odds[None,:,:] * odds[:,None,:])
# prods = (H[None,1::2,:] * H[::2,None,:])
# # change diagonals
# for i in range(len(prods)):
#     prods[i,i,:] = 1
# print("prods[:,:,n]:")
# for n in range(2**K):
#     print(n)
#     print(prods[:,:,n])
#     U, S, Vh = np.linalg.svd(prods[:,:,n])
#     print(S)
#     print(U[:,0], Vh[0,:])

# # log sum exp
# N = 2**K
# M = 2**(K-1)
# V = H[:2**(K-1)]
# print(V)

# alpha = cp.Variable((M,N), complex=True)
# beta = cp.Variable((M,N), complex=True)
# eta = np.where(V == +1, 0, np.pi * 0+1j)
# print(eta)

# constraints = [(alpha + beta == 0)]
# for v in range(len(V)):
#     for i,j in it.product(range(M), repeat=2):
#         if i == j: continue
#         # logdot = cp.log_sum_exp(alpha[i] + beta[j] + eta[v])
#         # constraints.append(logdot <= np.log(+(N - 1)/M))
#         # constraints.append(logdot >= np.log(-(N - 1)/M))
#         dot = cp.sum(cp.exp(alpha[i] + beta[j] + eta[v]))
#         constraints.append(dot <= +(N - 1)/M)
#         constraints.append(dot >= -(N - 1)/M)

# # try to get real-valued solutions
# objective = cp.Minimize(cp.sum_squares(cp.imag(alpha)) + cp.sum_squares(cp.imag(beta)))
# prob = cp.Problem(objective, constraints)
# prob.solve()

# # using solvers for bilinear constraints
# N = 2**K
# M = 5 #2**(K-1)
# V = H[:M]

# model = Model()

# t = model.addVar(vtype="C", name="t")
# a = np.empty((M,N), dtype=object)
# b = np.empty((M,N), dtype=object)
# for (i,n) in it.product(range(M), range(N)):
#     a[i,n] = model.addVar(vtype="C", name=f"a[{i},{n}]", lb=-2, ub=2)
#     b[i,n] = model.addVar(vtype="C", name=f"b[{i},{n}]", lb=-2, ub=2)

# # # try constraining the problem more?
# # b = a

# id_cons = np.empty((M,N), dtype=object)
# for (i,n) in it.product(range(M), range(N)):
#     id_cons[i,n] = model.addCons(a[i,n] * b[i,n] == 1, name=f"id_cons[{i},{j}]")

# orth_cons = np.empty((M,M,M), dtype=object)
# for (i,j,v) in it.product(range(M), repeat=3):
#     if i == j: continue
#     orth_cons[i,j,v] = model.addCons(
#         quicksum(a[i,n] * b[j,n] * V[v,n] for n in range(N)) == 0,
#         name=f"orth_cons[{i},{j},{v}]")

# obj_cons = model.addCons(
#     t >= quicksum(v**2 for v in a.flat) + quicksum(v**2 for v in b.flat),
#     name="objective")

# # model.setObjective(t, sense="minimize")
# model.optimize()
# sol = model.getBestSol()
# a_s = np.array([[sol[a[i,n]] for n in range(N)] for i in range(M)])
# b_s = np.array([[sol[b[i,n]] for n in range(N)] for i in range(M)])
# print("a,b soln:")
# print(a_s)
# print(b_s)

# print("a,b rounded:")
# a_s = a_s.round().astype(int)
# b_s = b_s.round().astype(int)
# print(a_s)
# print(b_s)

# # x = model.addVar("x")
# # y = model.addVar("y", vtype="INTEGER")
# # model.setObjective(x + y)
# # model.addCons(2*x - y*y >= 0)
# # model.optimize()
# # sol = model.getBestSol()
# # print("x: {}".format(sol[x]))
# # print("y: {}".format(sol[y]))

# # # hand check
# # a_s = b_s = H[[0,2]] # K == 2

# for i in range(M):
#     assert (a_s[i]*b_s[i] == 1).all()

# for (i,j,v) in it.product(range(M), repeat=3):
#     if i == j: continue
#     print(i,j,v)
#     print(a_s[i])
#     print(b_s[j])
#     print(V[v])
#     assert (a_s[i] * b_s[j] * V[v]).sum() == 0

# print(f"N={N}, M={M}: it worked.")

