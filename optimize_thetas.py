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

    # # empirical optima?
    # H_a = make_matrix(tr.tensor([1.5708, 3.1416]))
    # H_v = make_matrix(tr.tensor([1.5480, 4.6729]))
    # pt.subplot(1,2,1)
    # pt.imshow(H_a)
    # pt.colorbar()
    # pt.subplot(1,2,2)
    # pt.imshow(H_v)
    # pt.colorbar()
    # pt.show()

    # thetas = tr.linspace(0, 2*tr.pi, 6)[1:-1]
    # thetas.requires_grad_(True)
    # H = make_matrix(thetas)
    # print(H.detach().numpy().round(3))
    # print((H @ H).detach().numpy().round(3))

    # # test gradient flow
    # H.sum().backward()
    # print(thetas.grad.numpy().round(3))
    # thetas.grad[:] = 0.

    # pt.imshow(H.detach())
    # pt.colorbar()
    # pt.show()

    # optimize address and value codebooks for reliable recall
    K = 6
    num_memories = 2
    lr = 0.005
    thetas_a = (2 * tr.pi * tr.rand(K)).requires_grad_(True)
    thetas_v = (2 * tr.pi * tr.rand(K)).requires_grad_(True)
    # thetas_a = tr.linspace(0, 2 * tr.pi, K+2)[1:-1].requires_grad_(True)
    # thetas_v = tr.linspace(0, 2 * tr.pi, K+2)[1:-1].requires_grad_(True)
    optimizer = tr.optim.Adam([thetas_a, thetas_v], lr=lr)
    loss_curve, success_curve = [], []
    for itr in range(1000):
        # remake codebooks with current thetas
        H_a = make_matrix(thetas_a)
        H_v = make_matrix(thetas_v)

        # accumulate loss and gradient over possible memory states
        success = []
        loss = 0.
        # a_idx_gen = it.combinations(range(len(H_a)), r=num_memories)
        # v_idx_gen = it.product(range(len(H_v)), repeat=num_memories)
        # print(len(H_a)*(len(H_a)-1) / 2 * len(H_v)**2)
        # for (a_idx, v_idx) in it.product(a_idx_gen, v_idx_gen):
        # for sample in range(30):
        #     a_idx = np.random.choice(range(len(H_a)), size=num_memories, replace=False)
        #     v_idx = np.random.choice(range(len(H_v)), size=num_memories)
        a_idx = np.arange(2) # use first two addresses
        for combo, v_idx in enumerate(it.product(range(len(H_v)), repeat=num_memories)):
            # print(f" {combo} of {len(H_v)**num_memories}")

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
        optimizer.step()
        optimizer.zero_grad()

        # status update
        loss_curve.append(loss.item())
        success_curve.append(np.mean(success))
        if itr % 1 == 0:
            print(f"{itr}: loss = {loss:.3f}, success rate = {int(success_curve[-1]*100)}%, " +\
                f"{thetas_a.data % (2*tr.pi)/ (2*tr.pi)}, " +\
                f"{thetas_v.data % (2*tr.pi)/ (2*tr.pi)}")
            

    print("thetas_a,v:")
    print((thetas_a.data % (2*tr.pi)) / (2*tr.pi))
    print((thetas_v.data % (2*tr.pi)) / (2*tr.pi))

    pt.subplot(1,2,1)
    pt.plot(loss_curve)
    pt.ylabel("cross entropy loss")
    pt.subplot(1,2,2)
    pt.plot(success_curve)
    pt.ylabel("success rate")
    pt.gcf().supxlabel("Iterations")

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


