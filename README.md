This repository contains code to reproduce results in the paper:

[Ruipeng Liu, Garrett E Katz, Qinru Qiu, Simon Khan. Linearithmic Clean-up for Vector-Symbolic Key-Value Memory with Kroneker Rotation Products.  In 19th International Conference on Neurosymbolic Learning and Reasoning, 2025, PMLR.](https://openreview.net/forum?id=MxZZKQfjg5)

The key files are:

- `hrr.py`: Implements the binding and unbinding operations of HRRs

- `krop.py`: Implements the core algorithms to construct and cleanup KROP embeddings

- `krop_timing.py`: Run this script to regenerate Figure 1 from the paper

- `krop_capacity.py`: Run this script to regenerate Figure 2 from the paper

- `krop_mutable.py`: Run this script to regenrate Figures 3 and 4 from the paper

Other scripts are experimental, not included in the paper and can be ignored.

