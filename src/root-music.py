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


root_music_locations = []
for snr in snrs:
    power_noise = power_source / (10 ** (snr / 10))
    ula = model.UniformLinearArray(n_antenna_array, d0)
    sources = model.FarField1DSourcePlacement(sources_places)
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
    estimator = estimation.RootMUSIC1D(wavelength)
    resolved, estimates = estimator.estimate(R, sources.size, unit='deg')
    root_music_locations.append(estimates.locations)

print(root_music_locations)

arr_of_ones = np.ones(4)
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot()
print(root_music_locations[0])
ax1.scatter(root_music_locations[0], arr_of_ones, marker='o', color='red', s=60, label='-20 dB')
ax1.scatter(root_music_locations[1], arr_of_ones, marker='o', color='blue', s=60, label='-10 dB')
ax1.scatter(root_music_locations[2], arr_of_ones, marker='o', color='green', s=60, label='0 dB')
ax1.scatter(root_music_locations[3], arr_of_ones, marker='o', color='orange', s=60, label='10 dB')
ax1.scatter(sources_places * 180 / np.pi, arr_of_ones, marker='x', color='black', s=20, label='Real Source Places')
ax1.legend(loc='best')
ax1.set_xlabel('DOA/deg')
fig.gca().set_ylabel(r'P($\theta$)')


# Root-MUSIC for varying array size [5, 10, 20, 50, 100] (SNR = 10dB, snapshots = 100)
antenna_array_numbers = [5, 50, 100]
root_music_locations = []
power_noise = power_source / (10 ** (10 / 10))
for antenna_numbers in antenna_array_numbers:
    ula = model.UniformLinearArray(antenna_numbers, d0)
    sources = model.FarField1DSourcePlacement(sources_places)
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
    estimator = estimation.RootMUSIC1D(wavelength)
    resolved, estimates = estimator.estimate(R, sources.size, unit='deg')
    root_music_locations.append(estimates.locations)

arr_of_ones = np.ones(4)
fig = plt.figure(figsize=(8, 8))
ax2 = fig.add_subplot()
ax2.scatter(root_music_locations[0], arr_of_ones, marker='o', color='red', s=60, label='N = 5 Elements')
ax2.scatter(root_music_locations[1], arr_of_ones, marker='o', color='blue', s=60, label='N = 50 Elements')
ax2.scatter(root_music_locations[2], arr_of_ones, marker='o', color='green', s=60, label='N = 100 Elements')
ax2.scatter(sources_places * 180 / np.pi, arr_of_ones, marker='x', color='black', s=20, label='Real Source Places')
ax2.legend(loc='best')
ax2.set_xlabel('DOA/deg')
fig.gca().set_ylabel(r'P($\theta$)')


# Root-MUSIC 2 sources. 3 degrees (SNR = 10dB, N = 8)
root_music_locations = []
power_noise = power_source / (10 ** (10 / 10))
ula = model.UniformLinearArray(n_antenna_array, d0)
sources = model.FarField1DSourcePlacement(np.linspace(-np.pi / 64, 0, 2))
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
_, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
estimator = estimation.RootMUSIC1D(wavelength)
resolved, estimates = estimator.estimate(R, sources.size, unit='deg')

arr_of_ones = np.ones(2)
fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot()
ax3.scatter(estimates.locations, arr_of_ones, marker='o', color='red', s=60, label='N = 8 Elements, SNR = 10dB')
ax3.scatter(np.linspace(-np.pi / 64, 0, 2) * 180 / np.pi, arr_of_ones, marker='x', color='black', s=20,
            label='Real Source Places')
ax3.legend(loc='best')
ax3.set_xlabel('DOA/deg')
fig.gca().set_ylabel(r'P($\theta$)')


# Root-MUSIC for varying array size [5, 10, 20, 50, 100] (SNR = 0dB)
root_music_locations = []
antenna_array_numbers = [5, 20, 100]

for antenna_numbers in antenna_array_numbers:
    power_noise = power_source / (10 ** (0 / 10))
    ula = model.UniformLinearArray(antenna_numbers, d0)
    sources = model.FarField1DSourcePlacement(np.linspace(0, np.pi / 16, 2))
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
    estimator = estimation.RootMUSIC1D(wavelength)
    resolved, estimates = estimator.estimate(R, sources.size, unit='deg')
    root_music_locations.append(estimates.locations)


arr_of_ones = np.ones(2)
fig = plt.figure(figsize=(8, 8))
ax3 = fig.add_subplot()
ax3.scatter(root_music_locations[0], arr_of_ones, marker='o', color='red', s=60, label='N = 5 Elements')
ax3.scatter(root_music_locations[1], arr_of_ones, marker='o', color='blue', s=60, label='N = 20 Elements')
ax3.scatter(root_music_locations[2], arr_of_ones, marker='o', color='green', s=60, label='N = 100 Elements')
ax3.scatter(np.linspace(0, np.pi / 16, 2) * 180 / np.pi, arr_of_ones, marker='x', color='black', s=20,
            label='Real Source Places')
ax3.legend(loc='best')
ax3.set_xlabel('DOA/deg')
fig.gca().set_ylabel(r'P($\theta$)')

plt.show()