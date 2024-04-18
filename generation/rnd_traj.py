import numpy as np
import matplotlib.pyplot as plt

def modulated_sinusoidal(num_points, frequency_modulation, amplitude_modulation):
    # Generate time values
    t = np.linspace(0, 16*np.pi, num_points)
    
    # Modulate frequency and amplitude
    frequency = 1 + frequency_modulation * np.sin(t)
    amplitude = 1 + amplitude_modulation * np.sin(t)
    
    # Generate modulated sinusoidal waveform
    waveform = amplitude * np.sin(frequency * t)
    
    # Create array of points
    points = np.column_stack((t, waveform))
    
    return points

# Number of points in the waveform
num_points = 2000

# Modulation parameters
frequency_modulation = 0.5  # Modulation strength for frequency
amplitude_modulation = 0.2  # Modulation strength for amplitude

# Generate modulated sinusoidal waveform
points = modulated_sinusoidal(num_points, frequency_modulation, amplitude_modulation)

# Plot the waveform
plt.plot(points[:,1])
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Modulated Sinusoidal Waveform')
plt.grid(True)
plt.show()
