# Optimal bit allocation

Numerical methods for distributing a budget of `B` bits across `n` layers so as
to minimize the total quantization distortion.

## Problem

Each layer `i` has variance `sigma_i^2`. Allocating `b_i` bits to it gives a
distortion `sigma_i^2 * 2^(-2 b_i)`. We solve

```
minimize    J(b) = sum_i sigma_i^2 * 2^(-2 b_i)
subject to  G(b) = sum_i b_i - B = 0      (budget, multiplier mu)
            F_i(b) = -b_i <= 0            (no negative bits, multipliers lambda_i >= 0)
```

with Lagrangian `L(b, mu, lambda) = J(b) + mu * G(b) + lambda . F(b)`.

When no layer ends up with 0 bits, the KKT conditions give the closed form

```
b_i = B/n + 1/2 * log2(sigma_i^2 / sigma_bar^2),   sigma_bar^2 = geometric mean of the sigma_i^2
```

## Methods

All methods live in the `NumericalMethod` class in
[`numerical_method.py`](numerical_method.py).

| Method | Role |
| --- | --- |
| `get_solution()` | Closed-form KKT solution (raises if a layer would get negative bits) |
| `J_function(b)`, `J_derivative(b)` | Objective and its gradient |
| `G_function(b)`, `G_derivative(b)` | Budget constraint and its gradient `(1, ..., 1)` |
| `F_function(b)`, `F_derivative(b)` | Non-negativity constraints and their Jacobian `-I` |
| `projection(v, B)` | Orthogonal projection onto `K = {b >= 0, sum(b) = B}` |
| `projected_gradient_descent(b0, ...)` | Projected gradient descent on `K` |
| `lagrangian_derivative(b, mu, lambda_)` | Gradient of `L` w.r.t. `b` (should vanish at the optimum) |
| `minimize_lagrangian(mu, lambda_)` | Exact `argmin_b L(b, mu, lambda)` |
| `update_multipliers(b, mu, lambda_, rho)` | Dual ascent step on `(mu, lambda)` |
| `uzawa(mu0, lambda0, rho, ...)` | Uzawa algorithm |
| `display_followed_path(history, ...)` | Convergence plots (plus the path on `K` when `n = 3`) |

### Projected gradient

```
b^(k+1) = P_K( b^(k) - eta * grad J(b^(k)) )
```

### Uzawa

Each iteration alternates between:

1. **Primal step:** minimize the Lagrangian in `b` for fixed multipliers. `L`
   is separable and strictly convex, so setting `dL/db_i = 0` gives
   ```
   b_i = -1/2 * log2( (mu - lambda_i) / (2 ln2 sigma_i^2) )
   ```
2. **Dual step:** projected gradient ascent on the multipliers
   ```
   mu     <- mu + rho * G(b)
   lambda <- max(0, lambda + rho * F(b))
   ```

The multipliers are of the order of `1e-5` for the example below, so `rho`
must be small (around `1e-6` to `1e-5`); larger values make the iterates
oscillate.

## Usage

```python
import numpy as np
from numerical_method import NumericalMethod

sigmas = np.sqrt([0.00255, 0.00781, 0.01562])
m = NumericalMethod(B=12, sigmas=sigmas)

b_star = m.get_solution()

b_pg, history_pg = m.projected_gradient_descent(b0=[0, 0, 12], learning_rate=50.0)
b_uz, mu, lam, history_uz = m.uzawa(rho=1e-5)

m.display_followed_path(history_uz, b_star=b_star, title="Uzawa")
```

[`Notebook.ipynb`](Notebook.ipynb) runs both algorithms on this example and
plots their convergence.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Jupyter (for the notebook)
