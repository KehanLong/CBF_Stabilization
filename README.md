# CBF-based Stabilization for Nonlinear Systems

This repository contains the implementations for the paper "Stabilization of Nonlinear Systems through Control Barrier Functions", accepted to IEEE CDC 2024. The main idea is to use a Control Barrier Function (CBF), along with a weak candidate (quadratic) Control Lyapunov Function (CLF) to stabilize general nonlinear systems.


## Dependencies

To run the code, you will need to install the following packages:

- Python 3.9: [Official Python Website](https://www.python.org/downloads/release/python-390/)
- NumPy 1.26.0: [NumPy Documentation](https://numpy.org/doc/stable/)
- CVXPY 1.4.1: [CVXPY Documentation](https://www.cvxpy.org/index.html)
- Matplotlib 3.5.3: [Matplotlib Documentation](https://matplotlib.org/stable/index.html)

You can install the required packages using the following command:
```
pip install python==3.9 numpy==1.26.0 cvxpy==1.4.1 matplotlib==3.5.3

```

## How to Run

To reproduce the provided examples, follow these steps:

1. Clone the repository:
```git clone https://github.com/KehanLong/CBF_Stabilization.git```

2. Navigate to the cloned repo:
``` cd CBF_Stabilization```

3. Run the provided examples:
```python warm_up_CLF.py``` ; ```python unicycle_CLF.py``` ; ```python polynomial_systems_CLF.py``` ; ```python inverted_pendulum_CLF.py``` .

 
