import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.plotting as doaplot

np.random.seed(128)

# Parameters
wavelength = 1.0 # Normalized wavelength. Recall that d0 = wavelength / 2.
d0 = wavelength / 2
power_source = 1.0
snrs = np.linspace(-20, 10, 4)
n_snapshots = 100
n_antenna_array = 8
sp_music = []
sources_places = np.array([-1.11701, 0, 0.401426, 1.01229])


# MUSIC for varying SNR [-20dB, -15dB, -10dB, -5dB, 0dB, 5dB, 10dB]
for snr in snrs:
    power_noise = power_source / (10 ** (snr / 10))
    ula = model.UniformLinearArray(n_antenna_array, d0)
    sources = model.FarField1DSourcePlacement(sources_places)
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
    grid = estimation.FarField1DSearchGrid(start=-1.5, stop=1.5)
    estimator = estimation.MUSIC(ula, wavelength, grid)
    resolved, estimates, sp = estimator.estimate(R, sources.size, return_spectrum=True)
    sp_music.append(sp)
doaplot.plot_spectrum({'SNR = -20dB ': sp_music[0], 'SNR = -10dB ': sp_music[1], 'SNR = 0dB ': sp_music[2],
                       'SNR = 10dB ': sp_music[3]}, grid, ground_truth=sources, use_log_scale=True)


# MUSIC for varying snapshots [20, 200, 2000, 20000, 200000] (SNR = 10dB, N = 8)
arr_snapshots = []
n = 1
for i in range(5):
    n *= 10
    arr_snapshots.append(2 * n)
power_noise = power_source / (10 ** (10 / 10))
sp_music = []

for snapshot in arr_snapshots:
    ula = model.UniformLinearArray(n_antenna_array, d0)
    sources = model.FarField1DSourcePlacement(sources_places)
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          snapshot, return_covariance=True)
    grid = estimation.FarField1DSearchGrid(start=-1.5, stop=1.5)
    estimator = estimation.MUSIC(ula, wavelength, grid)
    resolved, estimates, sp = estimator.estimate(R, sources.size, return_spectrum=True)
    sp_music.append(sp)

doaplot.plot_spectrum({'Snapshots = 20 ': sp_music[0], 'Snapshots = 200 ': sp_music[1], 'Snapshots = 2000 ':
    sp_music[2], 'Snapshots = 20000 ': sp_music[3], 'Snapshots = 200000 ': sp_music[4]}, grid, ground_truth=sources,
    use_log_scale=True)


# MUSIC for varying array size [5, 10, 20, 50, 100] (SNR = 10dB, snapshots = 100)
antenna_array_numbers = [5, 10, 20, 50, 100]
sp_music = []

for antenna_numbers in antenna_array_numbers:
    ula = model.UniformLinearArray(antenna_numbers, d0)
    sources = model.FarField1DSourcePlacement(sources_places)
    source_signal = model.ComplexStochasticSignal(sources.size, power_source)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    _, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
    grid = estimation.FarField1DSearchGrid(start=-1.5, stop=1.5)
    estimator = estimation.MUSIC(ula, wavelength, grid)
    resolved, estimates, sp = estimator.estimate(R, sources.size, return_spectrum=True)
    sp_music.append(sp)

doaplot.plot_spectrum({'N = 5 ': sp_music[0], 'N = 10 ': sp_music[1], 'N = 20 ': sp_music[2], 'N = 50 ': sp_music[3],
                       'N = 100 ': sp_music[4]}, grid, ground_truth=sources, use_log_scale=True)


# MUSIC for a 5 element array (SNR = 10dB, snapshots = 100)
power_noise = power_source / (10 ** (10 / 10))
ula = model.UniformLinearArray(5, d0)
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
_, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal, n_snapshots,
                                      return_covariance=True)
grid = estimation.FarField1DSearchGrid(start=-1.5, stop=1.5)
estimator = estimation.MUSIC(ula, wavelength, grid)
resolved, estimates, sp = estimator.estimate(R, sources.size, return_spectrum=True)

doaplot.plot_spectrum({'N = 5 Elements': sp}, grid, ground_truth=sources, use_log_scale=True)

# MUSIC for a 200 element array (SNR = 10dB, snapshots = 100)
power_noise = power_source / (10 ** (10 / 10))
ula = model.UniformLinearArray(200, d0)
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
_, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal,
                                          n_snapshots, return_covariance=True)
grid = estimation.FarField1DSearchGrid(start=-1.5, stop=1.5)
estimator = estimation.MUSIC(ula, wavelength, grid)
resolved, estimates, sp = estimator.estimate(R, sources.size, return_spectrum=True)

doaplot.plot_spectrum({'N = 200 Elements': sp}, grid, ground_truth=sources, use_log_scale=True)


# MUSIC 2 sources. Good for 6 degrees, bad for 3 degrees (SNR = 10dB, snapshots = 100, N = 8)
power_noise = power_source / (10 ** (10 / 10))
ula = model.UniformLinearArray(8, d0)
sources = model.FarField1DSourcePlacement(np.linspace(-np.pi / 32, 0, 2))
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
_, R = model.get_narrowband_snapshots(ula, sources, wavelength, source_signal, noise_signal, n_snapshots,
                                      return_covariance=True)
grid = estimation.FarField1DSearchGrid(start=-0.2, stop=0.1)
estimator = estimation.MUSIC(ula, wavelength, grid)
resolved, estimates, sp = estimator.estimate(R, sources.size, return_spectrum=True)

doaplot.plot_spectrum({'N = 8, SNR = 10': sp}, grid, ground_truth=sources, use_log_scale=True)


