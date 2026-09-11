# Environment

## Hardware used for the committed results

- 2x Intel Xeon Gold 6248, 40 physical / 80 logical cores, 96 GiB RAM
- 3x NVIDIA RTX 4500 Ada Generation (23.6 GiB) + 1x RTX A5500 (23.5 GiB)
- GPU used for perceptron inference only; all other computation is
  processor-bound

## Software

```
Python        3.11.16
PyTorch       2.5.1  (CUDA 12.4, cuDNN 9.1)
scikit-learn  1.9.0
XGBoost       3.2.0
NumPy         2.4.6   (OpenBLAS 0.3.31)
pandas        3.0.5
SciPy         1.17.1
statsmodels   0.15.0
POT           0.9.7
```

The BLAS backend is recorded because it affects bitwise reproducibility of
the covariance decompositions used in correlation alignment.

## OS

Ubuntu 22.04.5 LTS, kernel 7.0.14-5-pve (Proxmox virtualised guest).
