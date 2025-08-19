#####################################################################################
# Rigid pendulum with noise
#
# This script studies the effects of noise on the dynamics of a classical pendulum
# initialized near an unstable equilibrium.
#
# This was used once in my activity log and should be later compared to 
# pendulum_quantum.
#####################################################################################

#####################################################################################
# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from tqdm import tqdm

#####################################################################################
# Parameters
## System
### Static parameters
g = 9.81                       # gravitational acceleration
L = 1.0                        # length
m = 1.0                        # mass
T = 2 * np.pi * np.sqrt(L / g) # period

### Dynamic parameters
t_max    = 3 * T
n_steps  = 1000
dt       = t_max / n_steps
t_array  = np.linspace(0, t_max, n_steps+1)

## Initial conditions
### With noise, with perturbation
θ_0     = np.pi
ω_0     = 0
θ_pert  = 0.001
ω_pert  = 0.001

## Noise: None, 'Brownian', 'GMR', or 'CIR'
np.random.seed(3)
noise = 'CIR'

## Other
action_angle = False
threshold_angle = np.pi * (1 + 0.1)

#####################################################################################
# Functions
def evolve_system(n_trials, θ_pert, ω_pert, noise, σ=0):
    """ Define EOM and generate data with the Euler–Maruyama algorithm.
        
        Parameters
        ----------
        n_trials : int; number of rows; different Brownian trajectory for each
        θ_pert   : int; percentage of initial condition
        ω_pert   : int; percentage of initial condition
        noise    : str or None; options are [None, 'Brownian', 'GMR', 'CIR']
        σ        : float; scaling parameter for noise
        
        Returns
        -------
        θ_array : 2D array; rows for trials, columns for time steps
        ω_array : 2D array; rows for trials, columns for time steps """
    
    # Initialize arrays
    θ_array       = np.zeros((n_trials, n_steps+1))
    ω_array       = np.zeros((n_trials, n_steps+1))
    θ_array[:, 0] = θ_0 * (1 + θ_pert)
    ω_array[:, 0] = ω_0 * (1 + ω_pert)
    dW            = np.sqrt(dt) * np.random.randn(n_trials, n_steps)
    
    # Evolve stepwise
    for i in range(n_steps):
        
        # Update angle
        θ_array[:, i+1] = θ_array[:, i] + ω_array[:, i] * dt
        
        # Assign names for clarity
        prev_θ      = θ_array[:, i]
        prev_ω      = ω_array[:, i]
        
        # Update angular velocity
        if not noise:             C = 0
        elif noise == 'Brownian': C = σ
        elif noise == 'GMR':      C = σ * ω_array[:, i].copy()
        elif noise == 'CIR':      C = σ * np.sqrt(np.abs(ω_array[:, i].copy()))
        
        ω_array[:, i+1] = prev_ω - (g/L)*np.sin(prev_θ)*dt + C*dW[:, i]
    
    return θ_array, ω_array

def lifetime(θ_array):
    escape_times = []
    for i in range(θ_array.shape[0]):
        for j in range(len(θ_array[i])):
            if abs(θ_array[i][j]) > threshold_angle:
                escape_times.append(t_array[j])
                break
    return np.mean(escape_times)

def diffusion(θ_array, ω_array):
    """ Calculates variance of total energy.
        
        Parameters
        ----------
        θ_array : 2D array; rows for trials, columns for time steps
        ω_array : 2D array; rows for trials, columns for time steps
        
        Returns
        -------
        E_var          :  
        diffusion_rate :  """
    
    E = (1/2) * ω_array**2 + (1 - np.cos(θ_array))
    E_var = np.var(E, axis=0)
    diffusion_rate, _, _, _, _ = linregress(t_array, E_var)
    return diffusion_rate

def Lyapunov_exponents(θ_1, ω_1, θ_2, ω_2):
    """ Compute finite-time Lyapunov exponents for trajectory pairs.
        
        Parameters
        ----------
        θ_i, ω_i : 2D arrays; trajectories for θ_i.shape[0] trials

        Returns
        -------
        λ_array : 1D array; finite-time Lyapunov exponent for each trajectory """
    
    # Initial distance between paired trajectories
    δθ0 = θ_2[:, 0] - θ_1[:, 0]
    δω0 = ω_2[:, 0] - ω_1[:, 0]
    δ0 = np.sqrt(δθ0**2 + δω0**2)
    
    # Final distance
    δθT = θ_2[:, -1] - θ_1[:, -1]
    δωT = ω_2[:, -1] - ω_1[:, -1]
    δT = np.sqrt(δθT**2 + δωT**2)
    
    # Total integration time
    T = θ_1.shape[1] * dt
    
    # Avoid divide-by-zero
    δ0 = np.where(δ0 == 0, 1e-12, δ0)
    
    # Finite-time Lyapunov exponent
    λ_array = (1 / T) * np.log(δT / δ0)
    
    return λ_array

#####################################################################################
# Generate data
## Calculate trajectories
θ_noise, ω_noise = evolve_system(
    n_trials = 10,
    θ_pert   = θ_pert,
    ω_pert   = 0,
    noise    = noise,
    σ        = 1)
θ_comp, ω_comp = evolve_system(
    n_trials = 1,
    θ_pert   = θ_pert,
    ω_pert   = 0,
    noise    = None)
θ_eq, ω_eq = evolve_system(
    n_trials = 1,
    θ_pert   = 0,
    ω_pert   = 0,
    noise    = None)

## Calculate mean lifetime and diffusion rate
τ_list, D_list, λ_list  = [], [], []
σ_array = np.linspace(0, 2, 201)
for σ in tqdm(σ_array, desc=f"{'calculating lifetimes':<35}"):
    
    # Evolve system; one with initial perturbation, one without
    θ_1, ω_1 = evolve_system(
        n_trials = 201,
        θ_pert   = θ_pert,
        ω_pert   = ω_pert,
        noise    = noise,
        σ        = σ)
    #θ_2, ω_2 = evolve_system(
    #    n_trials = 101,
    #    θ_pert   = 0,
    #    ω_pert   = 0,
    #    noise    = noise,
    #    σ        = σ)
    
    # Make calculations
    τ_list.append(lifetime(θ_1))
    #D_list.append(diffusion(θ_1, ω_1))
    #λ_list.append(np.mean(Lyapunov_exponents(θ_1, ω_1, θ_2, ω_2)))

# Smooth and normalize data
τ_list = gaussian_filter1d(τ_list, sigma=10)
τ_list = τ_list[1:-1]
#D_list = gaussian_filter1d(D_list, sigma=10)
#D_list = D_list[1:-1] / max(D_list)
#λ_list = gaussian_filter1d(λ_list, sigma=10)
#λ_list = λ_list[1:-1] / max(λ_list)
σ_list = list(σ_array / np.sqrt(L))[1:-1]

#####################################################################################
# Initialize plots
## Set up GridSpec
fig = plt.figure(figsize=(12, 5))
row, col = 2, 4
gs = GridSpec.GridSpec(row, col, height_ratios=[1, 1])

## Add subplots
ax0 = fig.add_subplot(gs[:, 0], polar=False)
if action_angle:
    ax1 = fig.add_subplot(gs[:, 1])
    ax2 = fig.add_subplot(gs[:, 2])
else:
    ax1 = fig.add_subplot(gs[0:4, 1:3])
ax3 = fig.add_subplot(gs[:, 3])

#####################################################################################
# Plot spatial trajectory
## Without noise, without perturbation
if ax0.name == 'polar': θ_eq_vals = θ_eq[0]
else:                   θ_eq_vals = θ_eq[0]/np.pi
ax0.plot(
    θ_eq_vals,
    t_array,
    linewidth = 2,
    linestyle = '--',
    color     = 'gray')

## With noise, with perturbation
for i in range(θ_noise.shape[0]):
    if ax0.name == 'polar': x_vals = θ_noise[i]
    else:                   x_vals = θ_noise[i]/np.pi
    ax0.plot(
        x_vals,
        t_array,
        linewidth = 2)

## Without noise, with perturbation
if ax0.name == 'polar': x_pert_vals = θ_comp[0]
else:                   x_pert_vals = θ_comp[0]/np.pi
ax0.plot(
    x_pert_vals,
    t_array,
    linewidth = 3,
    color     = 'k')

## Initial conditions
ax0.plot(x_pert_vals[0], t_array[0], 'ko')

#####################################################################################
# Plot phase space trajectory
## With noise, with perturbation
for i in range(θ_noise.shape[0]):
    x_vals = θ_noise[i]/np.pi
    y_vals = ω_noise[i]/np.pi
    ax1.plot(
        x_vals,
        y_vals)

## Without noise, without perturbation
x_eq_vals = θ_eq[0]/np.pi
y_eq_vals = ω_eq[0]/np.pi
ax1.plot(
    x_eq_vals,
    y_eq_vals,
    linewidth = 2,
    linestyle = '--',
    color     = 'gray')

## Without noise, with perturbation
x_pert_vals = θ_comp[0]/np.pi
y_pert_vals = ω_comp[0]/np.pi
ax1.plot(
    x_pert_vals,
    y_pert_vals,
    linewidth = 3,
    color     = 'k')

## Initial conditions
ax1.plot(x_pert_vals[0], y_pert_vals[0], 'ko')

#####################################################################################
# Plot volatility analysis
## Mean lifetime
ax3.plot(
    σ_list,
    τ_list,
    linewidth = 3,
    color     = 'k')

## Diffusion rate
#ax3.plot(
#    σ_list,
#    D_list,
#    linewidth = 1,
#    color     = 'b')

## Mean Lyapunov exponent
#ax3.plot(
#    σ_list,
#    λ_list,
#    linewidth = 1,
#    color     = 'r')

#####################################################################################
# Customize plots and show
## Angular position over time
ax0.set_title("Angular Position")
if ax0.name == 'polar':
    ax0.set_xlabel(r"$θ(t)$ [$π$]")
    ax0.set_rticks([])
else:
    ax0.set_xlabel(r"$θ$ [$π$]")
    ax0.set_ylabel(r"$t$ [s]")
ax0.set_xlim((-2, 6))
ax0.grid(True)

## Phase space (θ, ω)
ax1.set_title("Phase Space")
ax1.set_xlabel(r"$θ$ [$π$]")
ax1.set_ylabel(r"$\dot{θ}$ [$π$]")
ax1.set_xlim((-2, 6))
ax1.grid(True)

## Mean lifetime (σ, τ)
ax3.set_title("Mean Lifetime")
ax3.set_xlabel(r"$σ$ [1]")
ax3.set_ylabel(r"$τ$ [s]")
ax3.grid(True)

## Render plots
fig.tight_layout()
plt.show()

#####################################################################################