import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 1.0                # Length of the domain (cube side)
N = 50                 # Number of spatial grid points per dimension
T = 0.1                # Final time
dt = 1e-4              # Time step size
alpha = 1              # Coefficient for mixed derivative term
steps = int(T / dt)    # Number of time steps

# Discretization
dx = L / (N - 1)
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
z = np.linspace(0, L, N)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Initialize u and time derivatives
u = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)  # Initial condition
u_t = np.zeros_like(u)  # First time derivative
u_tt = np.zeros_like(u)  # Second time derivative
u_prev = np.copy(u)
u_next = np.copy(u)

# Helper functions for Laplacian and biharmonic operators
def laplacian(u):
    return (
        (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
         np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
         np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2) - 6 * u) / dx**2
    )

def biharmonic(u):
    return laplacian(laplacian(u))

# Time-stepping loop
frames = []
for step in range(steps):
    lap_u = laplacian(u)
    biharm_u = biharmonic(u)
    
    # Fourth-order time derivative approximation
    u_next = (
        2 * u
        - u_prev
        + dt**2 * lap_u
        - dt**4 * biharm_u
    )
    
    # Apply boundary conditions (u=0 at boundaries)
    u_next[0, :, :] = u_next[-1, :, :] = 0
    u_next[:, 0, :] = u_next[:, -1, :] = 0
    u_next[:, :, 0] = u_next[:, :, -1] = 0
    
    # Update
    u_prev = np.copy(u)
    u = np.copy(u_next)
    
    # Extract a 2D slice (e.g., middle of the domain in z-direction)
    mid_idx = N // 2
    u_slice = u[:, :, mid_idx]
    frames.append(u_slice)

# Create animation
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.contourf(X[:, :, mid_idx], Y[:, :, mid_idx], frames[0], levels=50, cmap='viridis')
fig.colorbar(cax, ax=ax, label='u(x, y, z_mid)')
ax.set_title(f"Solution at t={0:.2f}, z={z[mid_idx]:.2f}")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(i):
    ax.clear()
    cax = ax.contourf(X[:, :, mid_idx], Y[:, :, mid_idx], frames[i], levels=50, cmap='viridis')
    ax.set_title(f"Solution at t={i * dt:.2f}, z={z[mid_idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return cax

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
ani.save('wave_equation_solution.gif', writer='pillow', fps=20)
plt.show()