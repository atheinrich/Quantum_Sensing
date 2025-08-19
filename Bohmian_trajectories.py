##############################################################################################
# Bohmian mechanics
#
# Calculates Bohmian trajectories and checks compatibility with Born interpretation by 
# calculating probability distributions over many trajectories. This all assumes solutions
# and does not solve the Hamiltonian.
#
# Eventually, this should be replaced by something that actually solves a Hamiltonian.
# The idea is to use this interpretation to study noise-enhanced stability.
##############################################################################################

##############################################################################################
# Import
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from scipy.integrate import solve_ivp
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

##############################################################################################
# Parameters
# Constants
hbar = 1.0
m = 1.0
omega = 1.0

##############################################################################################
# Functions
# Harmonic oscillator basis state
def psi_n(x, n):
    Hn = hermite(n)
    norm = (1.0 / np.sqrt(2.0**n * np.math.factorial(n))) * (m*omega/np.pi/hbar)**0.25
    return norm * np.exp(-m*omega*x**2/(2*hbar)) * Hn(np.sqrt(m*omega/hbar)*x)

def E_n(n):
    return hbar * omega * (n + 0.5)

# Superposition wavefunction
def psi(x, t, c0=1/np.sqrt(2), c1=1/np.sqrt(2)):
    return (c0 * psi_n(x, 0) * np.exp(-1j * E_n(0) * t / hbar) +
            c1 * psi_n(x, 1) * np.exp(-1j * E_n(1) * t / hbar))

# ODE for trajectory
def trajectory_rhs(t, x, dx=1e-5):
    
    # Finite difference for dS/dx = hbar * Im(ψ* dψ/dx)/|ψ|^2
    psi_val = psi(x, t)
    psi_plus = psi(x + dx, t)
    psi_minus = psi(x - dx, t)
    dpsi_dx = (psi_plus - psi_minus) / (2 * dx)
    phase = hbar * np.imag(np.conj(psi_val) * dpsi_dx) / (np.abs(psi_val)**2 + 1e-14)

    return phase / m

##############################################################################################
# Plotting
# Times and initial setup
t_max = 10
t_eval = np.linspace(0, t_max, 40)
time_snapshots = [0, 5, 10]

# Initial positions for trajectories (few)
x0_few = np.linspace(-2, 2, 9)
trajectories_few = []
for x0 in tqdm(x0_few, desc='phase space'):
    sol = solve_ivp(trajectory_rhs, [t_eval[0], t_eval[-1]], [x0], t_eval=t_eval, method='RK45')
    trajectories_few.append(sol.y[0])

# Sample many initial positions from |ψ(x,0)|²
x_grid = np.linspace(-5, 5, 100)
pdf_0 = np.abs(psi(x_grid, 0))**2
pdf_0 /= np.trapz(pdf_0, x_grid)
cdf = np.cumsum(pdf_0)
cdf /= cdf[-1]
from scipy.interpolate import interp1d
inverse_cdf = interp1d(cdf, x_grid)

N_particles = 100
x0_many = inverse_cdf(np.random.rand(N_particles))
trajectories_many = np.zeros((N_particles, len(t_eval)))
for i in tqdm(range(N_particles), desc='distributions'):
    x0 = x0_many[i]
    sol = solve_ivp(trajectory_rhs, [t_eval[0], t_eval[-1]], [x0], t_eval=t_eval, method='RK45')
    trajectories_many[i, :] = sol.y[0]

# Prepare the plot
fig = plt.figure(figsize=(14, 8))
gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

# Left plot: trajectories
ax1 = fig.add_subplot(gs[0])
for i, x_traj in enumerate(trajectories_few):
    ax1.plot(t_eval, x_traj, label=f"$x_0={x0_few[i]:.1f}$")
ax1.set_xlabel("Time $t$")
ax1.set_ylabel("Position $x(t)$")
ax1.grid(True)

# Right plot: 4 vertical subplots for distributions at different times
gs_right = GridSpec(3, 1, height_ratios=[1,1,1], hspace=0.3, left=0.7, right=0.95, top=0.95, bottom=0.05)

x_plot = np.linspace(-3, 3, 500)

for i, t_snap in enumerate(time_snapshots):
    ax = fig.add_subplot(gs_right[i])
    idx = np.argmin(np.abs(t_eval - t_snap))
    x_vals_t = trajectories_many[:, idx]
    ax.hist(x_vals_t, bins=50, density=True, alpha=0.6, label="Bohmian ensemble")
    psi_sq = np.abs(psi(x_plot, t_snap))**2
    ax.plot(x_plot, psi_sq, 'k--', label=r"$|\psi(x,t)|^2$")
    ax.set_xlim(-4,4)
    ax.set_ylabel("Density")
    ax.set_title(f"$t = {t_snap:.1f}$")
    ax.grid(True)
    if i == 3: ax.set_xlabel("Position $x$")
    if i == 0: ax.legend(loc='upper right', fontsize=8)

plt.show()

##############################################################################################