import itertools as it
import numpy as np

K = 3
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
    
odds = H[1::2, :]
print('odds:')
print(odds)
print('even times odd')
print(mapping[1::2,::2])

# prods = (odds[None,:,:] * odds[:,None,:])
prods = (H[None,1::2,:] * H[::2,None,:])
# change diagonals
for i in range(len(prods)):
    prods[i,i,:] = 1
print("prods[:,:,n]:")
for n in range(2**K):
    print(n)
    print(prods[:,:,n])
    U, S, Vh = np.linalg.svd(prods[:,:,n])
    print(S)
    print(U[:,0], Vh[0,:])
