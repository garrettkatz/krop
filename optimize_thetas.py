import numpy as np
import matplotlib.pyplot as pt
import itertools as it
import torch as tr

# important to minimize round-off error
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

    thetas = tr.linspace(0, 2*tr.pi, 6)[1:-1]
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
    num_memories = 2
    lr = 0.001
    thetas_a = (2 * tr.pi * tr.rand(4)).requires_grad_(True)
    thetas_v = (2 * tr.pi * tr.rand(4)).requires_grad_(True)
    loss_curve, success_curve = [], []
    for itr in range(300):
        # remake codebooks with current thetas
        H_a = make_matrix(thetas_a)
        H_v = make_matrix(thetas_v)

        # accumulate loss and gradient over sample
        success = []
        loss = 0.
        for sample in range(100):
            a_idx = np.random.choice(range(len(H_a)), size=num_memories, replace=False)
            v_idx = np.random.choice(range(len(H_v)), size=num_memories)

            # bind addresses and values
            M = tr.zeros(len(H_a))
            for (i_a, i_v) in zip(a_idx, v_idx):
                a, v = H_a[i_a], H_v[i_v]
                M = M + bind(a, v)

            # unbind noisy values
            U = []
            for (i_a, i_v) in zip(a_idx, v_idx):
                a = H_a[i_a]
                u = unbind(M, a)
                U.append(u)
            U = tr.stack(U)

            # calculate loss and success rate
            dots = U @ H_v
            targets = tr.tensor(v_idx)
            loss += tr.nn.functional.cross_entropy(dots, targets)
            success.append((dots.argmax(dim=-1) == targets).all())

        # update thetas
        loss.backward()
        thetas_a.data = thetas_a.data - lr * thetas_a.grad
        thetas_v.data = thetas_v.data - lr * thetas_v.grad
        thetas_a.grad[:] = 0
        thetas_v.grad[:] = 0

        # status update
        loss_curve.append(loss.item())
        success_curve.append(np.mean(success))
        if itr % 100 == 0: print(f"{itr}: loss = {loss:.3f}, success rate = {int(success_curve[-1]*100)}%")
            

    print("thetas_a,v:")
    print((thetas_a.data % (2*tr.pi)) / (2*tr.pi))
    print((thetas_v.data % (2*tr.pi)) / (2*tr.pi))

    pt.plot(loss_curve)

    pt.figure()
    pt.subplot(1,2,1)
    pt.imshow(H_a.detach())
    pt.title("Addresses")
    pt.colorbar()
    pt.subplot(1,2,2)
    pt.imshow(H_v.detach())
    pt.title("Values")
    pt.colorbar()
    pt.show()


