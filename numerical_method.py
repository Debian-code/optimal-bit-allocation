import numpy as np
import matplotlib.pyplot as plt 

class NumericalMethod:
    def __init__(self,B:int,sigmas:np.ndarray):
        self.B = B
        self.sigmas = sigmas
    
    def theta_barre(self) -> float:
        variances = np.asarray(self.sigmas, dtype=float) ** 2
        return 2.0 ** np.mean(np.log2(variances))

    # --- Constraints ---------------------------------------------------------
    # Equality   : G(b) = sum(b) - B = 0           (multiplier mu, real)
    # Inequality : F_i(b) = -b_i <= 0, one per layer (multiplier lambda_i >= 0)
    # Lagrangian : L(b, mu, lambda) = J(b) + mu * G(b) + lambda . F(b)

    def G_function(self, b) -> float:
        """Equality (budget) constraint: G(b) = sum(b) - B = 0."""
        return np.sum(np.asarray(b, dtype=float)) - self.B

    def G_derivative(self, b) -> np.ndarray:
        """Gradient of G: (1, ..., 1)."""
        return np.ones(len(self.sigmas))

    def F_function(self, b) -> np.ndarray:
        """Inequality constraints: F_i(b) = -b_i <= 0, one per layer."""
        return -np.asarray(b, dtype=float)

    def F_derivative(self, b) -> np.ndarray:
        """Jacobian of F: -I (row i is the gradient of F_i)."""
        return -np.eye(len(self.sigmas))
    
    def get_solution(self) -> np.ndarray:
        """
        KKT solution when every layer receives bits:
            b_i = B/n + 1/2 * log2(sigma_i^2 / sigma_bar^2)
        Only valid if all resulting b_i are >= 0.
        """
        variances = np.asarray(self.sigmas, dtype=float) ** 2
        n = len(variances)
        b = self.B / n + 0.5 * np.log2(variances / self.theta_barre())
        if np.any(b < 0):
            raise ValueError(
                "Some layers should receive 0 bits: the closed form without "
                "active constraints does not apply, use uzawa or "
                "projected_gradient_descent instead."
            )
        return b

    def J_function(self, b) -> float:
        sigmas = np.asarray(self.sigmas, dtype=float)
        b = np.asarray(b, dtype=float)
        return np.sum(sigmas**2 * 2.0**(-2 * b))

    def J_derivative(self, b) -> np.ndarray:
        sigmas = np.asarray(self.sigmas, dtype=float)
        b = np.asarray(b, dtype=float)
        return -2 * np.log(2) * sigmas**2 * 2.0**(-2 * b)

    def projection(self, v, B) -> np.ndarray:
        """Orthogonal projection onto K = {b >= 0, sum(b) = B}."""
        v = np.asarray(v, dtype=float)
        u = np.sort(v)[::-1]                     
        cumsum = np.cumsum(u) - B
        k = np.arange(1, len(v) + 1)
        k_max = k[u - cumsum / k > 0][-1]        # largest valid k
        tau = cumsum[k_max - 1] / k_max
        return np.maximum(v - tau, 0.0)

    def projected_gradient_descent(self, b0, learning_rate=0.1,
                                   num_iterations=1000, tol=1e-10):
        b = self.projection(b0, self.B)              
        history = [b.copy()]
        for _ in range(num_iterations):
            b_new = self.projection(b - learning_rate * self.J_derivative(b), self.B)
            history.append(b_new.copy())
            if np.linalg.norm(b_new - b) < tol:  
                b = b_new
                break
            b = b_new
        return b, np.array(history)


    def display_followed_path(self, history, b_star=None, title="Projected gradient"):
        """
        Plot the path followed by the algorithm.
        - history: array (n_iterations, n), row k is b^(k)
        - b_star : reference solution; defaults to the analytical solution
        If n = 3, also draws the level sets of J on K with the path on top.
        """
        history = np.asarray(history, dtype=float)
        variances = np.asarray(self.sigmas, dtype=float) ** 2
        n_iter, n = history.shape
        B = self.B

        # Reference solution: closed form (all layers active)
        if b_star is None:
            b_star = self.get_solution()
        b_star = np.asarray(b_star, dtype=float)

        J_star = self.J_function(b_star)
        J_k = np.array([self.J_function(b) for b in history])
        err = np.linalg.norm(history - b_star, axis=1)

        n_panels = 4 if n == 3 else 3
        fig, axs = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
        axs = list(axs)

        # --- Path on K (only in dimension 3) ---
        if n == 3:
            ax = axs.pop(0)
            b1, b2 = np.meshgrid(np.linspace(0, B, 300), np.linspace(0, B, 300))
            b3 = B - b1 - b2
            J_grid = (variances[0] * 2.0**(-2 * b1)
                    + variances[1] * 2.0**(-2 * b2)
                    + variances[2] * 2.0**(-2 * b3))
            J_grid = np.where(b3 >= 0, J_grid, np.nan)
            cs = ax.contourf(b1, b2, np.log10(J_grid), levels=30, cmap="viridis")
            fig.colorbar(cs, ax=ax, label=r"$\log_{10} J$")
            ax.plot(history[:, 0], history[:, 1], "o-", color="white", ms=2, lw=1, label="path")
            ax.plot(*history[0, :2], "s", color="red", label="start")
            ax.plot(*b_star[:2], "*", color="orange", ms=14, label=r"$b^*$")
            ax.set_xlabel("$b_1$"); ax.set_ylabel("$b_2$")
            ax.set_title(r"Path on $K$ (with $b_3 = B - b_1 - b_2$)")
            ax.legend(loc="upper right")

        # --- Convergence (valid for any n) ---
        axs[0].semilogy(np.maximum(J_k - J_star, 1e-16))
        axs[0].set_title(r"$J(b^{(k)}) - J(b^*)$"); axs[0].set_xlabel("iteration")

        axs[1].semilogy(np.maximum(err, 1e-16))
        axs[1].set_title(r"$\|b^{(k)} - b^*\|$"); axs[1].set_xlabel("iteration")

        for i in range(n):
            axs[2].plot(history[:, i], label=f"$b_{{{i+1}}}$")
        axs[2].set_title("Bits per layer"); axs[2].set_xlabel("iteration")
        if n <= 10:
            axs[2].legend()

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # --- Uzawa ---------------------------------------------------------------

    def lagrangian_derivative(self, b, mu, lambda_) -> np.ndarray:
        """Gradient of L w.r.t. b: J'(b) + mu * G'(b) + F'(b)^T lambda."""
        return (self.J_derivative(b)
                + mu * self.G_derivative(b)
                + self.F_derivative(b).T @ lambda_)

    def minimize_lagrangian(self, mu, lambda_, eps=1e-15) -> np.ndarray:
        """
        b = argmin_b L(b, mu, lambda), unconstrained.
        L is separable in b, and dL/db_i = 0 gives:
            2 ln2 sigma_i^2 2^(-2 b_i) = mu - lambda_i
            b_i = -1/2 * log2((mu - lambda_i) / (2 ln2 sigma_i^2))
        If mu - lambda_i <= 0, L is unbounded below in b_i: we clip it to eps,
        which gives a very large b_i that the next mu update will correct.
        """
        variances = np.asarray(self.sigmas, dtype=float) ** 2
        c = np.maximum(mu - np.asarray(lambda_, dtype=float), eps)
        return -0.5 * np.log2(c / (2 * np.log(2) * variances))

    def update_multipliers(self, b, mu, lambda_, rho):
        """
        Projected gradient ascent on the dual:
            mu     <- mu + rho * G(b)                  (equality: no projection)
            lambda <- max(0, lambda + rho * F(b))      (inequalities: projection onto R+)
        """
        mu_new = mu + rho * self.G_function(b)
        lambda_new = np.maximum(0.0, lambda_ + rho * self.F_function(b))
        return mu_new, lambda_new

    def uzawa(self, mu0=None, lambda0=None, rho=1e-6,
              num_iterations=10000, tol=1e-10):
        """
        Uzawa algorithm. Returns b, mu, lambda and the history of b
        (compatible with display_followed_path).
        By default, mu0 = marginal price of a bit at the uniform allocation.
        """
        n = len(self.sigmas)
        if mu0 is None:
            mu0 = np.mean(-self.J_derivative(np.full(n, self.B / n)))
        if mu0 <= 0:
            raise ValueError("mu0 must be strictly positive.")
        mu = float(mu0)
        lambda_ = np.zeros(n) if lambda0 is None else np.asarray(lambda0, dtype=float)

        b = self.minimize_lagrangian(mu, lambda_)
        history = [b.copy()]
        for _ in range(num_iterations):
            mu, lambda_ = self.update_multipliers(b, mu, lambda_, rho)
            b_new = self.minimize_lagrangian(mu, lambda_)
            history.append(b_new.copy())
            converged = (np.linalg.norm(b_new - b) < tol
                         and abs(self.G_function(b_new)) < tol)
            b = b_new
            if converged:
                break
        return b, mu, lambda_, np.array(history)
