import matplotlib.pyplot as pt
import numpy as np
import sympy as sp

theta = np.linspace(0, 2*np.pi, 100)

tsc = 2*np.sin(theta)*np.cos(theta)
sdf = np.sin(theta)**2 - np.cos(theta)**2

print(2*np.sin(np.pi/8)*np.cos(np.pi/8))
print(np.sin(np.pi/8)**2 - np.cos(np.pi/8)**2)

# pt.plot(theta, tsc)
# pt.plot(theta, sdf)
# pt.plot(theta, -tsc)
# pt.plot(theta, -sdf)
# pt.plot(theta, np.maximum(np.maximum(tsc, sdf), np.maximum(-tsc, -sdf)), 'r--')
# pt.show()

theta = np.pi/8
H = np.array([[np.cos(theta), np.sin(theta)],[np.sin(theta),-np.cos(theta)]])
print(H)

# v = H[0] + np.roll(H[1], 1)
# v = H[0] + np.roll(H[0], 1)
# v = H[1] + np.roll(H[1], 1)
v = H[1] + np.roll(H[0], 1)
print(v)
print(H @ v)
print(H @ np.roll(v, -1))

t1, t2 = sp.symbols("t1 t2")
M1 = sp.Matrix([[sp.cos(t1), sp.sin(t1)], [sp.sin(t1), -sp.cos(t1)]])
M2 = sp.Matrix([[sp.cos(t2), sp.sin(t2)], [sp.sin(t2), -sp.cos(t2)]])
print(repr(M1 @ M2))
from sympy.physics.quantum.tensorproduct import TensorProduct
print(repr(TensorProduct(M1, M2)))
print(repr(TensorProduct(M1, M2).subs(t1, 0).subs(t2, 0)))

