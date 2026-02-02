# Runge–Kutta Method Comparison (RK4 vs SciPy RK45)

## Description
This project numerically solves the first-order ordinary differential equation

y' = −y + ln(x)

using:
- A **manually implemented classical 4th-order Runge–Kutta method (RK4)**, and
- SciPy’s built-in **RK45 solver** for comparison.

The solutions are evaluated over the same interval and compared visually and numerically.

---

## Methods
- **Manual RK4** implementation using a fixed step size  
- **SciPy `solve_ivp` (RK45)** with interpolation to match the RK4 grid  
- **Absolute error** computed between RK4 and RK45 solutions

---

## Features
- Solves the ODE using **1000 steps**
- Generates **three figures**:
  1. RK4 solution
  2. SciPy RK45 solution
  3. Comparison plot (RK4 vs RK45)
- Prints:
  - Maximum absolute error
  - Final values from both methods

---

## Requirements
This program requires Python and the following libraries:

- `numpy`
- `matplotlib`
- `scipy`

You can install them using:
```bash
pip install numpy matplotlib scipy
