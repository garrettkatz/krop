import matplotlib.pyplot as pt
import itertools as it
import torch as tr
tr.set_default_dtype(tr.float64)

# torch versions of key functions

def bind(a, v):
    # circular convolution of vectors a and v
    # need to double-check if this is actually equivalent to HRR circular convolution
    idx = tr.arange(len(a))[:,None] - tr.arange(len(a))
    return a[idx] @ v

def unbind(m, a):
    # circular correlation via Plate's involution
    # returns noisy v currently bound to a in memory m
    a_inv = tr.roll(tr.flip(a, dims=(0,)), 1)
    return bind(a_inv, m)

def make_matrix(thetas):
    H = tr.tensor([[1.]])
    for k in range(len(thetas)):
        H = tr.kron(H, 
            tr.stack([
                tr.stack([tr.cos(thetas[k]),  tr.sin(thetas[k])]),
                tr.stack([tr.sin(thetas[k]), -tr.cos(thetas[k])]),]))
    return H

if __name__ == "__main__":

    # empirical optima?
    H_a = make_matrix(tr.tensor([1.5708, 3.1416]))
    H_v = make_matrix(tr.tensor([1.5480, 4.6729]))
    pt.subplot(1,2,1)
    pt.imshow(H_a)
    pt.colorbar()
    pt.subplot(1,2,2)
    pt.imshow(H_v)
    pt.colorbar()
    pt.show()

    thetas = tr.linspace(0, 2*tr.pi, 4)[1:-1]
    thetas.requires_grad_(True)
    H = make_matrix(thetas)
    print(H.detach().numpy().round(3))
    print((H @ H).detach().numpy().round(3))

    # test gradient flow
    H.sum().backward()
    print(thetas.grad.numpy().round(3))
    thetas.grad[:] = 0.

    pt.imshow(H.detach())
    pt.colorbar()
    pt.show()

    # test optimize
    for itr in range(100):
        H = make_matrix(thetas)
        loss = H.sum()
        loss.backward()
        thetas.data = thetas.data - 0.01*thetas.grad
        thetas.grad[:] = 0.
        print(f"{itr}: {loss:.3f}")

    print(H.detach().numpy().round(3))
    print(thetas.detach().numpy().round(3))
    input('.')

    # optimize address and value codebooks for reliable recall
    lr = 0.001
    thetas_a = (2 * tr.pi * tr.rand(2)).requires_grad_(True)
    thetas_v = (2 * tr.pi * tr.rand(2)).requires_grad_(True)
    loss_curve = []
    for itr in range(10000):
        # remake codebooks with current thetas
        H_a = make_matrix(thetas_a)
        H_v = make_matrix(thetas_v)

        # accumulate loss and gradient over each possible pair
        # (could sub-sample to scale up)
        loss = 0.
        for (i,j) in it.product(range(len(H_a)), repeat=2):
            # bind address and value
            a, v = H_a[i], H_v[j]
            M = bind(a, v)
    
            # penalize bad unbind and cleanup
            u = unbind(M, a)
            dots = tr.softmax(H_v @ u, dim=0)
            loss += tr.nn.functional.cross_entropy(dots, tr.tensor(j))

        # update thetas
        loss.backward()
        thetas_a.data = thetas_a.data - lr * thetas_a.grad
        thetas_v.data = thetas_v.data - lr * thetas_v.grad
        thetas_a.grad[:] = 0
        thetas_v.grad[:] = 0

        # status update
        loss_curve.append(loss.item())
        if itr % 100 == 0: print(f"{itr}: {loss:.3f}")
            

    print("thetas_a,v:")
    print(thetas_a.data)
    print(thetas_v.data)

    pt.plot(loss_curve)
    pt.show()


