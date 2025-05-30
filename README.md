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
- dx/dt = -y  
- dy/dt = x
![Rotation Dynamics](1_1.png)

### 🌪️ 2. Lorenz System (2D Projection)
- dx/dt = sigma * (y - x)  
- dy/dt = x * (rho - z) - y

### 🔂 3. Van der Pol Oscillator
- dx/dt = y  
- dy/dt = mu * (1 - x^2) * y - x
![Van der Pol](1_3.png)

### 🔄 4. Rotational + Radial Limit Cycle Dynamics
- dx/dt = -y + 0.1 * x * (1 - x^2 - y^2)  
- dy/dt =  x + 0.1 * y * (1 - x^2 - y^2)
![Rotational+Radial Limit Cycle](1_2.png)

## 🚀 How to Run

Each demo is independent. Simply open MATLAB and run one of them


## 📦 Requirements
	•	MATLAB R2020 or later
	•	No external toolboxes required

## 📜 License

MIT License © Alice Chen


