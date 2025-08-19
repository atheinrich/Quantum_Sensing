#####################################################################################
# Quantum pendulum
#
# Calculated in the position (θ) basis. I use a Vandermonde matrix to construct the
# DFT because my domain is periodic.
#
# This is used to investigate noise-enhanced stability and Bloch-type wavefunctions.
# It should absorb the Bohm code.
#####################################################################################

#####################################################################################
# Imports
## Solving the Hamiltonian
import numpy as np
from scipy.linalg import eigh

## Bohmian mechanics
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

## Utility
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

#####################################################################################
# Parameters
## Schrödinger mechanics
### Static parameters
m = 1.0        # mass
g = 10       # gravitational acceleration
L = 4.0        # length
ℏ = 1.0
I = m * L**2   # moment of inertia

### Dynamic parameters
n_steps  = 512 # number of θ values in the domain
n_trials = 1   # size of parameter space
g_array  = np.linspace(10, 11, n_trials)
θ = np.linspace(0, 2*np.pi, n_steps, endpoint=False) # periodic grid for position
N = np.fft.fftfreq(n_steps, d=(θ[1]-θ[0])/(2*np.pi)) # periodic grid for momentum

## Bohmian mechanics
###

#####################################################################################
# Schrödinger mechanics
## Build and solve Hamiltonian
def build_kinetic():
    # Kinetic energy in position basis
    K_J = np.diag((ℏ*N)**2 / (2*I))
    K_θ = J_to_θ_basis(n_steps, K_J)
    return K_θ

def build_potential(g):
    # Potential energy in position basis
    U   = m*g*L * (1 - np.cos(θ))
    U_θ = np.diag(U)
    return U_θ

def build_Hamiltonian(K_θ, U_θ):
    # Hamiltonian in position basis
    H_θ = K_θ + U_θ
    return H_θ

def J_to_θ_basis(n_steps, K_J):
    """ Converts kinetic energy operator from the momentum basis to position. 
        ⟨θ|K|θ'⟩ = (1/2π) ∑_n ((ℏn)²/(2I) * e^(in(θ-θ')),      K = F⁻¹ K_J F
        Constructs DFT as Vandermonde matrix, then computes similarity transform. """
    
    z     = np.exp(-2j * np.pi * np.arange(n_steps) / n_steps)
    F     = np.vander(z, n_steps, increasing=True) / np.sqrt(n_steps)
    F_dag = np.conjugate(F).T
    K     = F_dag @ K_J @ F
    return np.real(K)

def find_eigenstates(H_list):
    """ Calculates eigenvalues and eigenvectors for the given matrix.
        For some reason, eigh provides the same eigenvectors as QuTiP, but eig does not.
        
        Returns
        -------
        states : list of arrays; sets the standard representation """
    
    eigenvalues, eigenvectors = [], []
    for i in tqdm(range(len(H_list)), desc=f"{'finding eigenstates':<35}"):
        eigenvalue, eigenvector = np.linalg.eigh(H_list[i])
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
    states = [np.array(eigenvalues), np.array(eigenvectors)]
    return states

#####################################################################################
# Bohmian mechanics
## Prepare initial state
def E_Gaussian(states, E_0, σ_E):
        """ Generates superposition around a given energy. """
        
        # Compute weights
        weights = np.exp(-(states[0][0] - E_0)**2 / (2 * σ_E**2))
        plt.plot(states[0][0], weights, 'o')
        plt.xlabel('Energy')
        plt.ylabel('Weight')
        plt.title('Weights applied to eigenstates')
        plt.grid(True)
        plt.show()
        
        # Form wavepacket
        ψ = np.sum(weights[:, np.newaxis] * states[1][0].T, axis=0)
        ψ = rtc(ψ / np.linalg.norm(ψ))
        return ψ

def wrapped_Gaussian(θ, σ):
    """ Generates a Gaussian without reference to eigenstates. """
    
    d = np.minimum(np.abs(θ - np.pi), 2*np.pi - np.abs(θ - np.pi))
    ψ = np.exp(-0.5 * (d / σ)**2)
    ψ = rtc(ψ / np.linalg.norm(ψ))
    return ψ

## Trajectories
def Bohmian_velocity(θ, ψ, dθ, ℏ=1.0, m=1.0):
    """Compute Bohmian velocity field on a periodic domain."""
    dψ_dθ = np.roll(ψ, -1) - np.roll(ψ, 1)  # central difference
    dψ_dθ /= (2 * dθ)
    v = ℏ * np.imag(np.conj(ψ) * dψ_dθ) / (np.abs(ψ)**2 + 1e-14)
    return v / m

def trajectory_rhs_periodic(t, θ, θ_vals, ψ_func, dθ, ℏ=1.0, m=1.0):
    """RHS of Bohmian trajectory ODE for periodic domain."""
    ψ = ψ_func(θ_vals, t)
    v_field = Bohmian_velocity(θ_vals, ψ, dθ, ℏ, m)
    velocity_interp = interp1d(θ_vals, v_field, kind='cubic', fill_value="wrap", bounds_error=False)
    return velocity_interp(θ)  # returns dθ/dt

def evolve_trajectories(ψ_func, θ_vals, t_eval, θ0_array, ℏ=1.0, m=1.0):
    """Evolve Bohmian trajectories given initial positions θ0_array."""
    dθ = θ_vals[1] - θ_vals[0]
    trajectories = np.zeros((len(θ0_array), len(t_eval)))
    for i, θ0 in enumerate(θ0_array):
        sol = solve_ivp(trajectory_rhs_periodic, [t_eval[0], t_eval[-1]], [θ0],
                        args=(θ_vals, ψ_func, dθ, ℏ, m), t_eval=t_eval, method='RK45')
        trajectories[i, :] = np.mod(sol.y[0], 2*np.pi)
    return trajectories

def sample_initial_positions(ψ0, θ_vals, N_particles):
    """Sample from |ψ(θ)|² using inverse transform sampling."""
    pdf = np.abs(ψ0)**2
    pdf /= np.trapz(pdf, θ_vals)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    inverse_cdf = interp1d(cdf, θ_vals, kind='linear', fill_value="extrapolate")
    return inverse_cdf(np.random.rand(N_particles))

#####################################################################################
# Utility
def rtc(array):
    """ Shorthand to convert a row vector to a column vector (or vice-versa). 
    
        Example
        -------
        column_vector = rtc(sys.states[1][0][:,0]) """
    
    if len(array.shape) == 1:   vector = array.reshape(array.shape[0], 1)
    elif len(array.shape) == 2: vector = array.reshape(1, array.shape[0])
    return vector

def plot_density(θ, ψ):
    """ Constructs an initial wavefunction localized in θ near θ = π by 
        superposing eigenstates around a target energy E_0 using a Gaussian weight.
    
        Parameters
        ----------
        θ : angular grid """
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    
    axs[0].plot(θ, np.abs(ψ)**2, label=r'$|\ψ_func(\theta)|^2$')
    axs[0].axvline(np.pi, color='gray', linestyle='--', label=r'$\theta = \pi$')
    axs[0].set_xlabel(r'$\theta$')
    axs[0].set_ylabel('Probability density')
    axs[0].grid(True)

    axs[1].plot(θ, np.angle(ψ), label=r'$\arg[\ψ_func(\theta)]$')
    axs[1].set_xlabel(r'$\theta$')
    axs[1].set_ylabel('Phase')
    axs[1].grid(True)

    psi_k = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(ψ)))
    p = np.fft.fftshift(np.fft.fftfreq(len(θ), d=(θ[1]-θ[0])/(2*np.pi)))

    axs[2].plot(p, np.abs(psi_k)**2)
    axs[2].set_xlabel('p')
    axs[2].set_ylabel('Probability density')
    axs[2].grid(True)

    plt.show()

def plot_eigenstates():
    canvas = plt.figure(figsize=(14, 7))
    grid = GridSpec(nrows=2, ncols=3)
    graph_1 = canvas.add_subplot(grid[0,0], projection='polar')
    graph_2 = canvas.add_subplot(grid[0,1], projection='polar')
    graph_3 = canvas.add_subplot(grid[0,2], projection='polar')
    graph_4 = canvas.add_subplot(grid[1,:])
    
    ψ1, E1 = states[1][0][:,0],  states[0][:,0]
    ψ2, E2 = states[1][0][:,29], states[0][:,29]
    ψ3, E3 = states[1][0][:,64], states[0][:,64]
    ψ4, E4 = states[1][0][:,66], states[0][:,66]
    
    graph_1.plot(θ, np.abs(ψ1)**2, color='r')
    graph_2.plot(θ, np.abs(ψ2)**2, color='g')
    graph_3.plot(θ, np.abs(ψ4)**2, color='y')
    graph_3.plot(θ, np.abs(ψ3)**2, color='b')
    
    U = m*g*L * (1 - np.cos(θ))
    graph_4.plot(θ, U, color='k', linestyle='dotted')
    graph_4.plot(θ, E1*np.ones(θ.shape[0]), color='r')
    graph_4.plot(θ, E2*np.ones(θ.shape[0]), color='g')
    graph_4.plot(θ, E4*np.ones(θ.shape[0]), color='y')
    graph_4.plot(θ, E3*np.ones(θ.shape[0]), color='b')
    
    graph_1.set_theta_zero_location("S")
    graph_2.set_theta_zero_location("S")
    graph_3.set_theta_zero_location("S")
    plt.show()

def plot_expectation(operator, states, mod=False):
    
    # Process states matrix
    if type(states) == list:
        expectation_array = []
        
        # Compute expectations
        for i in range(states[1].shape[0]):
            temp_list_1 = []
            for j in range(states[1][i].shape[1]):
                temp_list_2 = plot_expectation(operator, rtc(states[1][i][:,j]))
                temp_list_1.append(temp_list_2)
            expectation_array.append(np.array(temp_list_1).T)
        
        # Convert to array and round to nearest integer if desired
        else: expectation_array = np.array(expectation_array)
        
        # Return results
        if mod:
            return np.array(expectation_array)
        else:
            plt.figure(figsize=(12,8))
            plt.plot(np.array(expectation_array[0]), 'o-')
            plt.grid(True)
            plt.show()
    
    # Process column vector
    else:
        if operator.all() != np.diag(θ).all():
            expectation_value = np.conj(states).T @ operator @ states
            return np.real(expectation_value.item())
        else:
            return mean_angle(states, θ)

def plot_uncertainty(operator, states):
    """ Calculates and returns (or plots) expectation values for one or more states. 
        
        Parameters
        ----------
        operator     : 2D array
        states       : states matrix or column vector
        sys          : commandline use; plots directly
        
        Returns
        -------
        output       : 2D array; one row per λ 
        
        Example
        -------
        uncertainty(sys.J_z, sys.states, sys=sys) """

    # Initialize data containers
    expectations, output = [[], []], []

    # Calculate expectation values
    expectations[0] = plot_expectation(operator,            states, mod=True)
    expectations[1] = plot_expectation(operator @ operator, states, mod=True)

    # Use expectation values to calculate uncertainty
    for i in range(states[1].shape[0]):
        cache = []
        for j in range(states[1].shape[2]):
            cache.append(np.sqrt(abs(expectations[1][i][j]-expectations[0][i][j]**2)))
        output.append(cache)
    
    # Return results or plot with system variable
    plt.figure(figsize=(12,8))
    plt.plot(np.array(output[0]), 'o-', color='k', label='$√(⟨x⟩^2-⟨x^2⟩)$')
    plt.plot(np.array([x**2 for x in expectations[0][0]]), 'o-', color='b', label=r'$⟨x⟩^2$')
    plt.plot(np.array(expectations[1][0]), 'o-', color='g', label=r'$⟨x^2⟩$')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trajectories(θ_vals, ψ_func, trajectories, t_eval, snapshots=[0, 5, 10]):
    """Plot Bohmian trajectories and snapshots of distributions."""
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    # Left: individual trajectories
    ax1 = fig.add_subplot(gs[0])
    for traj in trajectories[:10]:
        ax1.plot(t_eval, traj)
    ax1.set_xlabel("Time $t$")
    ax1.set_ylabel("Position $\\theta(t)$")
    ax1.set_title("Bohmian Trajectories")
    ax1.grid(True)

    # Right: snapshots of |ψ(θ,t)|² and histogram of particle positions
    gs_right = GridSpec(3, 1, height_ratios=[1,1,1], hspace=0.3,
                        left=0.7, right=0.95, top=0.95, bottom=0.05)

    for i, t_snap in enumerate(snapshots):
        ax = fig.add_subplot(gs_right[i])
        idx = np.argmin(np.abs(t_eval - t_snap))
        θ_t = trajectories[:, idx]
        ax.hist(θ_t, bins=50, density=True, alpha=0.6, label="Bohmian particles")
        ψ_t = ψ_func(θ_vals, t_snap)
        ax.plot(θ_vals, np.abs(ψ_t)**2, 'k--', label=r"$|\ψ_func(\theta,t)|^2$")
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylabel("Density")
        ax.set_title(f"$t = {t_snap:.1f}$")
        ax.grid(True)
        if i == len(snapshots) - 1: ax.set_xlabel(r"$\theta$")
        if i == 0: ax.legend()

    plt.show()

def ψ_func(θ_s, t):
    θ_s = θ
    ψ1, E1 = 0.1 * states[1][0][:,63], states[0][:,63]
    ψ2, E2 = 0.8 * states[1][0][:,64], states[0][:,64]
    ψ3, E3 = 0.1 * states[1][0][:,65], states[0][:,65]
    ψ = ψ1 * np.exp(-1j*E1*t) + ψ2 * np.exp(1j*E2*t) + ψ3 * np.exp(1j*E3*t)
    ψ = rtc(ψ / np.linalg.norm(ψ))
    return ψ

def mean_angle(ψ, θ):
    # Uses circular statistics: ⟨θ⟩ ≡ arg(⟨exp(iθ)⟩)
    ψ = np.roll(ψ, -N//2)
    P = np.abs(ψ.T.conj())**2
    P /= np.sum(P)
    θ_mean = np.angle(np.sum(P * np.exp(1j*θ))) % (2*np.pi)
    if θ_mean >= 2*np.pi-0.1: θ_mean = 0
    return θ_mean

#####################################################################################
# Generate data
## Find eigenstates
K_list = [build_kinetic() for _ in g_array]
U_list = [build_potential(g) for g in g_array]
H_list = [build_Hamiltonian(K_list[i], U_list[i]) for i in range(len(g_array))]
states = find_eigenstates(H_list)

## Plot energy values
#plot_expectation(H_list[0], states)
#plot_expectation(K_list[0], states)
#plot_expectation(U_list[0], states)

## Plot probability densities
#plot_eigenstates()

## Plot expected positions
#ψ = rtc(states[1][0][:,229])
#plot_uncertainty(np.diag(θ), states)



## Select initial state
#E_0 = 2 * m * g_array[0] * L
#σ_E = E_0 * 0.001
#ψ = E_Gaussian(states, E_0, σ_E)
#ψ = wrapped_Gaussian(θ, 0.05)

## Plot data
#plot_density(θ, ψ_func(θ, 0))

#θ0_array = np.array([np.pi-0.1, np.pi, np.pi+0.1])
#t_array  = np.linspace(0, 10, 101)
#trajectories = evolve_trajectories(ψ_func, θ, t_array, θ0_array)
#plot_trajectories(θ, ψ_func, trajectories, t_array)

#####################################################################################