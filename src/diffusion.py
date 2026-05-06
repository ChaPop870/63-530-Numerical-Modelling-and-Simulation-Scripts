# %%
import matplotlib.pyplot as plt
import numpy as np

n = 101
x = np.linspace(0, 1, n)
dx = x[1] - x[0]
s = 10 * np.sin(np.pi * x)

def diffusion_matrix(n: int) -> np.ndarray:
    """Defines the n x n transformation matrix necessary to solve the diffusion problem."""
    H = np.zeros((n, n))
    H += np.diag(-1 * np.ones(n - 1), k=1)
    H += np.diag(2 * np.ones(n))
    H += np.diag(-1 * np.ones(n - 1), k=-1)
    H[0, :3] = np.array([-1.5, 2.0, -0.5])
    H[-1, :] = 0
    H[-1, -1] = 1.0
    return H

H = diffusion_matrix(n)

u = dx**2 * np.linalg.inv(H) @ s

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(x, s)
ax2.plot(x, u)
plt.show()