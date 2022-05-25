import numpy as np
import doatools.model as model
import doatools.estimation as estimation
import doatools.performance as perf
import matplotlib.pyplot as plt

wavelength = 1.0
d0 = wavelength / 2
sources_places = np.array([-0.628319, -0.0523599, 0, 1.51844])
ula = model.UniformLinearArray(12, d0)
sources = model.FarField1DSourcePlacement(sources_places)

power_source = 1
source_signal = model.ComplexStochasticSignal(sources.size, power_source)
n_snapshots = 200
estimator = estimation.Esprit1D(wavelength)

snrs = np.linspace(5, 100, 20)
n_repeats = 300

mses = np.zeros((len(snrs), 4))

for i, snr in enumerate(snrs):
    power_noise = power_source / (10**(10 / 10))
    ula = model.UniformLinearArray(snr, d0)
    noise_signal = model.ComplexStochasticSignal(ula.size, power_noise)
    cur_mse = np.zeros(4)
    cur_crb_det = 0.0
    for r in range(n_repeats):
        A = ula.steering_matrix(sources, wavelength)
        S = source_signal.emit(n_snapshots)
        N = noise_signal.emit(n_snapshots)
        Y = A @ S + N
        Rs = (S @ S.conj().T) / n_snapshots
        Ry = (Y @ Y.conj().T) / n_snapshots
        resolved, estimates = estimator.estimate(Ry, sources.size, d0)
        cur_mse[0] += np.mean((estimates.locations[0] - sources.locations[0])**2)
        cur_mse[1] += np.mean((estimates.locations[1] - sources.locations[1])**2)
        cur_mse[2] += np.mean((estimates.locations[2] - sources.locations[2])**2)
        cur_mse[3] += np.mean((estimates.locations[3] - sources.locations[3])**2)
        B_det = perf.ecov_music_1d(ula, sources, wavelength, Rs, power_noise,
                                   n_snapshots)
        cur_crb_det += np.mean(np.diag(B_det))
    B_sto = perf.crb_sto_farfield_1d(ula, sources, wavelength, power_source,
                                     power_noise, n_snapshots)
    B_stouc = perf.crb_stouc_farfield_1d(ula, sources, wavelength, power_source,
                                         power_noise, n_snapshots)
    mses[i][0] = cur_mse[0] / n_repeats
    mses[i][1] = cur_mse[1] / n_repeats
    mses[i][2] = cur_mse[2] / n_repeats
    mses[i][3] = cur_mse[3] / n_repeats

plt.figure(figsize=(8, 6))
plt.semilogy(
    snrs, mses[:, 0], '-x',
    snrs, mses[:, 1], '-x',
    snrs, mses[:, 2], '-x',
    snrs, mses[:, 3], '-x',

)

plt.xlabel('Array number (N)')
plt.ylabel(r'MSE / $\mathrm{rad}^2$')
plt.grid(True)
plt.legend(['phi = - 36', 'phi = -3', 'phi = 0', 'phi = 87'])
plt.title('MSE')
plt.margins(x=0)
plt.show()
