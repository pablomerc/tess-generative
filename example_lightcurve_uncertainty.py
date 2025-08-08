import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

# Choose which light curve to use
lightcurve = 'sinusoidal'  # Options: 'hogg' or 'sinusoidal'

# Choose whether to show the statistics box
show_stats_box = False  # Options: True or False

if lightcurve == 'hogg':
    # Load the light curve data
    file_path = '../tess-foundation/datasets/simulated_hogg/sims-T100-N10000-Q1000-K13-M17/data/simh_0_0_8282.txt'
    data = pd.read_csv(file_path)

    # Extract time and flux
    time = data['time'].values
    flux = data['flux'].values

elif lightcurve == 'sinusoidal':
    # Generate a controlled sinusoidal light curve
    np.random.seed(42)  # For reproducible results

    # Time array
    time = np.linspace(1, 100, 100)

    # Sinusoidal signal parameters
    amplitude = 1.0
    period = 20.0
    phase = 0.0
    baseline = 0.0

    # Generate clean sinusoidal signal
    clean_flux = baseline + amplitude * np.sin(2 * np.pi * time / period + phase)

    # Add Gaussian noise
    noise_level = 0.15
    noise = np.random.normal(0, noise_level, len(time))
    flux = clean_flux + noise

# Generate uncertainties based on the light curve type
if lightcurve == 'hogg':
    # For Hogg data, use proportional uncertainties
    flux_std = np.std(flux)
    uncertainties = np.random.normal(0.05 * flux_std, 0.02 * flux_std, len(flux))
    uncertainties = np.abs(uncertainties)

elif lightcurve == 'sinusoidal':
    # For sinusoidal data, use constant uncertainty (more realistic for controlled experiment)
    uncertainties = np.full(len(flux), 0.3)  # Constant uncertainty of 0.3

# Create the plot
plt.figure(figsize=(12, 6))

# Set larger font sizes for all text elements
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
})

# Plot the main light curve line
plt.plot(time, flux, 'b-', linewidth=2, label='Light Curve', alpha=0.8)

# Create the uncertainty band
upper_bound = flux + uncertainties
lower_bound = flux - uncertainties

# Fill the area between the bounds to show uncertainty
plt.fill_between(time, lower_bound, upper_bound, alpha=0.3, color='blue',
                 label='Uncertainty Band')

# Plot the data points on top
plt.scatter(time, flux, c='blue', s=20, alpha=0.6, zorder=5)

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Flux')
plt.title(f'{lightcurve.capitalize()} Light Curve with Uncertainties')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some statistics to the plot (optional)
if show_stats_box:
    if lightcurve == 'sinusoidal':
        stats_text = f'Type: Sinusoidal\nAmplitude: {amplitude}\nPeriod: {period}\nNoise level: {noise_level}\nMean uncertainty: {np.mean(uncertainties):.3f}'
    else:
        stats_text = f'Type: Hogg Data\nMean flux: {np.mean(flux):.3f}\nStd flux: {np.std(flux):.3f}\nMean uncertainty: {np.mean(uncertainties):.3f}'

    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12)

plt.tight_layout()
plt.show()

# Print some basic statistics
print(f"Light curve type: {lightcurve}")
if lightcurve == 'hogg':
    print(f"Data loaded from: {file_path}")
else:
    print(f"Generated sinusoidal light curve")
print(f"Number of data points: {len(time)}")
print(f"Time range: {time.min():.1f} to {time.max():.1f}")
print(f"Flux range: {flux.min():.3f} to {flux.max():.3f}")
print(f"Mean flux: {np.mean(flux):.3f}")
print(f"Flux standard deviation: {np.std(flux):.3f}")
print(f"Mean uncertainty: {np.mean(uncertainties):.3f}")
