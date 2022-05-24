import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import matplotlib.pyplot as plt

np.random.seed(128)

# Parameters
wavelength = 1.0 # Normalized wavelength. Recall that d0 = wavelength / 2.
d0 = wavelength / 2
power_source = 1.0
snrs = np.linspace(-20, 10, 4)
n_snapshots = 100
n_antenna_array = 8
sp_root_music = []
sources_places = np.array([-1.11701, 0, 0.401426, 1.01229])


esprit_locations = []
for snr in snrs:
    power_noise = power_source / (10 ** (snr / 10))
    ula = model.UniformLinearArray(n_antenna_array, d0)
    sources = model.FarField1DSourcePlacement(sources_places)
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
    estimator = estimation.Esprit1D(wavelength)
    resolved, estimates = estimator.estimate(R, sources.size, unit='deg')
    esprit_locations.append(estimates.locations)

print(esprit_locations)


# Root-MUSIC for varying array size [5, 10, 20, 50, 100] (SNR = 10dB, snapshots = 100)
antenna_array_numbers = [5, 50, 100]
esprit_locations = []
power_noise = power_source / (10 ** (10 / 10))
for antenna_numbers in antenna_array_numbers:
    ula = model.UniformLinearArray(antenna_numbers, d0)
    sources = model.FarField1DSourcePlacement(sources_places)
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
    estimator = estimation.Esprit1D(wavelength)
    resolved, estimates = estimator.estimate(R, sources.size, unit='deg')
    esprit_locations.append(estimates.locations)

print(esprit_locations)
print(sources_places * 180 / np.pi)