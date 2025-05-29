
# 🌀 MOCK Dynamics Learning in MATLAB

This repository implements the **MOCK (Multivariate Occupation Kernel)** algorithm for learning continuous vector fields from trajectory data using MATLAB. The implementation demonstrates the method on several well-known dynamical systems.

## ✨ Features

- Kernel-based vector field estimation
- Regularized regression using occupation kernel integrals
- Snapshot-based trajectory reshaping for training
- ODE-based inference from learned dynamics
- Modular implementation with support for custom systems

## 📈 Included Dynamical Systems

### 🔁 1. Rotation Dynamics
```math
\dot{x} = -y, \quad \dot{y} = x

###🌪️ 2. Lorenz System (2D Projection)

\dot{x} = \sigma(y - x), \quad \dot{y} = x(\rho - z) - y

###🔂 3. Van der Pol Oscillator

\dot{x} = y, \quad \dot{y} = \mu(1 - x^2)y - x

###🔄 4. Rotational + Radial Limit Cycle Dynamics

\dot{x} = -y + 0.1x(1 - x^2 - y^2), \quad \dot{y} = x + 0.1y(1 - x^2 - y^2)

##📂 Project Structure

mock-dynamics/
├── mock.m
├── helpers/
│   ├── solve_alpha_rbf.m
│   ├── get_segments.m
│   ├── rbf_kernel.m
│   └── generate_vector_field.m
└── README.md

##🚀 How to Run

Each demo is independent. Simply open MATLAB and run one of them


##📦 Requirements
	•	MATLAB R2020 or later
	•	No external toolboxes required

##📜 License

MIT License © Alice Chen

Let me know if you want a version with embedded figures or GitHub Actions integration!
