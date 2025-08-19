###########################################################################################################################
# Classical estimation protocols
#
# Sends a noisy, Gaussian-broadened signal to an object of exact velocity, then returns a noisy, frequency-shifted signal.
# This shift is then estimated using three different methods.
#
# This was used to get some practice with classical metrology. I used it once for my activity log.
###########################################################################################################################

###########################################################################################################################
# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import matplotlib.gridspec as GridSpec
from scipy.signal import convolve

###########################################################################################################################
# Shorthand
π = np.pi

###########################################################################################################################
# Functions
def construct_output_signal(freqs, times, f_0, σ_0):
    """ Takes a center frequency and standard deviation, constructs a Gaussian in
        frequency space, applies an Hann envelope, then transforms a normalized
        version back to the time domain. """
    
    # Generate Gaussian spread of frequencies around f_0
    output_signal_freq = np.exp(-(1/2) * ((freqs-f_0)/σ_0)**2)
    noise = 0.01 * np.random.randn(len(times))
    output_signal_freq = output_signal_freq + noise
    
    # Normalize and apply window
    output_signal_time = np.fft.ifft(output_signal_freq)
    envelope           = (1/2) * (1 - np.cos(2*π*times/T))
    output_signal_time = envelope * output_signal_time
    
    return output_signal_freq, output_signal_time

def construct_return_signal(freqs, times, f_0, output_signal_freq, v, σ_v):
    """ Takes a center frequency and standard deviation, constructs a Gaussian in
        frequency space, applies an Hann envelope, then transforms it back to the
        time domain. """
    
    # Generate Gaussian spread of velocities around v
    f_D      = (v/c)*f_0
    #σ_D      = (σ_v/c)*f_0 * 1000
    #f_spread = np.exp(-(1/2) * ((freqs-(f_0+f_D))/σ_D)**2)
    #return_signal_freq = convolve(output_signal_freq, f_spread, mode='same')
    
    # Generate noisy return signal
    return_signal_freq = np.exp(-(1/2) * ((freqs-(f_0+f_D))/σ_0)**2)
    noise = 0.01 * np.random.randn(len(times))
    return_signal_freq = return_signal_freq + noise
    
    # Generate the return signal in the time domain
    return_signal_time = np.fft.ifft(return_signal_freq)
    envelope           = (1/2) * (1 - np.cos(2*π*times/T))
    return_signal_time = envelope * np.real(return_signal_time)
    
    return return_signal_freq, return_signal_time

def maximum_likelihood(times, signal, f_0, sample_rate):
    """ Maximum Likelihood Estimation (MLE) using FFT periodogram. """
    
    freqs    = np.fft.fftfreq(len(times), 1/sample_rate)
    spectrum = np.abs(np.fft.fft(signal))
    f_peak   = abs(freqs[np.argmax(spectrum)])
    return abs(f_0 - f_peak)

def method_of_moments(times, signal, f_0):
    """ Method of Moments (MoM) using phase unwrapping. """
    
    analytic_signal = hilbert(np.real(signal))
    phase           = np.unwrap(np.angle(analytic_signal))
    f_peak          = (phase[-1]-phase[0])/(2*π*(times[-1]-times[0]))
    return abs(f_0 - f_peak)

def autocorrelation_phase(signal, sample_rate):
    """ Constructs complex analytic signal from a real-valued signal, then computes
        the autocorrelation between that signal and a copy shifted by one sample. """
    
    analytic_signal = hilbert(signal)
    autocorrelation = np.sum(analytic_signal[1:] * np.conj(analytic_signal[:-1]))
    phase  = np.angle(autocorrelation)
    f_peak = phase*sample_rate/(2*π)
    return abs(f_0 - f_peak)

###########################################################################################################################
# Parameters
## Manual settings
### Static parameters
f_0 = 10e3  # center frequency of output signal in Hz
σ_0 = 75e1  # Gaussian width of output signal in Hz
v   = 40e6  # target velocity in m/s
σ_v = 10e6  # Gaussian width of target velocity in m/s
c   = 3e8   # speed of light in m/s
T   = 0.001 # observation time in s
sample_rate = 100/T # sampling rate in Hz

### Dynamic parameters
np.random.seed(122)

## Automatic calculations
n_samples = int(T * sample_rate)
times     = np.arange(n_samples) / sample_rate
freqs     = np.fft.fftfreq(n_samples, 1/sample_rate)
f_D       = (v/c)*f_0

###########################################################################################################################
# Generate data
## Construct output signal in time domain
output_signal_freq, output_signal_time = construct_output_signal(freqs, times, f_0, σ_0)

## Construct input signal in time domain
return_signal_freq, return_signal_time = construct_return_signal(freqs, times, f_0, output_signal_freq, v, σ_v)

## Estimate frequency shift
f_MLE = maximum_likelihood(times, return_signal_time, f_0, sample_rate)
f_MOM = method_of_moments(times, return_signal_time, f_0)
f_ACP = autocorrelation_phase(return_signal_time, sample_rate)
print('MLE', round(f_MLE), round((f_MLE*c)/f_0))
print('MOM', round(f_MOM), round((f_MOM*c)/f_0))
print('ACP', round(f_ACP), round((f_ACP*c)/f_0))
print('f_D', round(f_D),   round(v))

###########################################################################################################################
# Plot data
# GridSpec layout
fig = plt.figure(figsize=(15, 5))
gs = GridSpec.GridSpec(1, 4)

# Output signal in time domain
ax0 = fig.add_subplot(gs[0])
ax0.plot(times * 1e3, np.real(output_signal_time) * 1e3, label='Original Signal (Real Part)', color='black')
ax0.set_title("Output signal")
ax0.set_ylabel("Amplitude")
ax0.set_xlabel("Time [ms]")
ax0.grid(True)

# Output signal in frequency domain
ax1 = fig.add_subplot(gs[1])
ax1.scatter(freqs, np.real(output_signal_freq), label='Original Signal (Real Part)', color='black')
ax1.set_title("Output signal")
ax1.set_xlabel("Frequency [Hz]")
ax1.set_xlim(0, 20000)
ax1.grid(True)

# Return signal in time domain
ax2 = fig.add_subplot(gs[2])
ax2.plot(times * 1e3, np.real(return_signal_time) * 1e3, label='Original Signal (Real Part)', color='black')
ax2.set_title("Return signal")
ax2.set_xlabel("Time [ms]")
ax2.grid(True)

# Return signal in frequency domain
ax3 = fig.add_subplot(gs[3])
ax3.scatter(freqs, return_signal_freq, label='Original Signal (Real Part)', color='black')
ax3.axvline(abs(f_D),   color='k')
ax3.axvline(abs(f_MLE), color='r', linestyle='--', label='MLE')
ax3.axvline(abs(f_MOM), color='g', linestyle='--')
ax3.axvline(abs(f_ACP), color='b', linestyle='--')
ax3.set_title("Return signal")
ax3.set_xlabel("Frequency [Hz]")
ax3.set_xlim(0, 20000)
ax3.grid(True)

plt.tight_layout()
plt.show()

###########################################################################################################################