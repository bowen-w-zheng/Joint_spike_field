#!/usr/bin/env python3
# Multi-band LFP + multi-unit spikes: 5 latent bands in LFP; each unit couples to 3/5 bands.

from __future__ import annotations
import os, sys, math, pathlib
import numpy as np
import matplotlib.pyplot as plt
import mne

# --- Repo paths ---
cwd_parent  = str(pathlib.Path(os.getcwd()).parent)
root_parent = str(pathlib.Path(os.getcwd()).parent.parent)
for p in (cwd_parent, root_parent):
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Project imports (pure-NumPy parts; avoid early JAX) ---
from src.utils_multitaper import derotate_tfr
from src.priors import gamma_prior_simple
from src.generate_data import generate_spikes_from_latent
from src.plots import plot_lfp_and_spikes, plot_band_realparts

# --- RNG ---
seed = np.random.randint(0, 1_000_000)
rng  = np.random.default_rng(seed)
print(f"Random seed = {seed}")

# ────────── 1) Simulation set-up: 5 latent bands ──────────
freqs_hz     = np.asarray([10.0, 15.0, 20.0, 30.0, 40.0])  # 5 real bands
n_process    = len(freqs_hz)

fs           = 250.0
duration_sec = 300.0
dt           = 1/fs
n_samples    = int(round(duration_sec * fs))
time         = np.arange(0, duration_sec, dt)

window_sec   = 2
NW_product   = 2
n_tapers     = 2 * NW_product - 1

half_bw_hz     = np.asarray([0.01] * n_process)
lambda_true    = math.pi * half_bw_hz
sigma_v_true   = np.asarray([5.0, 6.0, 7.0, 6.0, 5.5])
sigma_eps_true = np.asarray([20.0] * n_process)

# Complex OU state per band
Z_true  = np.zeros((n_process, n_samples), np.complex128)
sqrt_dt_over_2 = math.sqrt(dt/2)
for j in range(n_process):
    for n in range(1, n_samples):
        Z_true[j, n] = ((1 - lambda_true[j]*dt) * Z_true[j, n-1] +
                        sigma_v_true[j] * sqrt_dt_over_2 *
                        (rng.standard_normal() + 1j * rng.standard_normal()))

noise_Z = (rng.standard_normal(Z_true.shape) + 1j * rng.standard_normal(Z_true.shape)) * sigma_eps_true[:, None]
Z_noisy = Z_true + noise_Z

# Embed 5 true bands into a dense frequency grid; others = complex white noise
all_freqs = np.arange(1, 51)
freqs     = all_freqs
Z_all = np.zeros((len(freqs), n_samples), dtype=np.complex128)
for j in range(len(freqs)):
    if freqs[j] in freqs_hz:
        row = np.where(freqs_hz == freqs[j])[0][0]
        Z_all[j, :] = Z_noisy[row, :]
    else:
        Z_all[j, :] = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) * sigma_eps_true[0]

# For reference: true latent per candidate freq
Z_true_all = np.zeros((len(freqs), n_samples), dtype=np.complex128)
for j in range(len(freqs)):
    if freqs[j] in freqs_hz:
        row = np.where(freqs_hz == freqs[j])[0][0]
        Z_true_all[j, :] = Z_true[row, :]

# LFP as sum of real parts of carrier-modulated coeffs
lfp = np.sum((np.exp(2j * math.pi * freqs[:, None] * time) * Z_all).real, axis=0)

# Per-band complex filtered signals (for plotting/coupling sanity checks)
filtered_signal = np.zeros((len(freqs_hz), n_samples), dtype=np.complex128)
for i in range(len(freqs_hz)):
    filtered_signal[i, :] = np.exp(2j * math.pi * freqs_hz[i] * time) * Z_true[i, :]

# ────────── 2) Multitaper spectrogram (complex) ──────────
tfr_raw = mne.time_frequency.tfr_array_multitaper(
    lfp[None, None, :],
    sfreq=fs,
    freqs=freqs,
    n_cycles=freqs * window_sec,
    time_bandwidth=2 * NW_product,
    output="complex",
    zero_mean=False,
)[0, 0]  # (K, F, T)

tfr_time = np.arange(tfr_raw.shape[2]) / fs
tfr      = tfr_raw.copy()
tfr_raw  = np.swapaxes(tfr_raw, 0, 1)  # -> (F, K, T)
tfr      = np.swapaxes(tfr, 0, 1)      # -> (F, K, T)

# Derotate and scale each taper
M = int(round(window_sec * fs))
decim = 1
tfr = derotate_tfr(tfr, freqs, fs, decim, M)

tapers, _ = mne.time_frequency.multitaper.dpss_windows(M, NW_product, Kmax=1)
scaling_factor = 2.0 / tapers.sum(axis=1)
tfr_raw = tfr_raw * scaling_factor
tfr     = tfr * scaling_factor

# Non-overlapping windows
tfr_ds      = tfr[:, :, ::M]
tfr_time_ds = tfr_time[::M]
selected_freqs = freqs_hz

# ────────── 3) Multi-unit spikes: 5 units, each couples to 3 of 5 bands ──────────
n_units   = 4
delta_spk = 0.005
n_lags    = 20

mu_g, _    = gamma_prior_simple(n_lags=n_lags, strong_neg=-5, mild_neg=-0.5, k_mild=4, tau_gamma=1.5)
gamma_true = mu_g.copy()

def random_beta_with_mask(rng, n_bands: int, k_active: int = 3):
    mask = np.zeros(n_bands, dtype=bool)
    active_idx = rng.choice(n_bands, size=k_active, replace=False)
    mask[active_idx] = True
    b0 = rng.normal(loc=-2.0, scale=0.4)
    beta = [b0]
    for b in range(n_bands):
        if mask[b]:
            mag   = rng.uniform(0.3, 0.8)*0.03
            theta = rng.uniform(-np.pi, np.pi)
            betaR = mag * np.cos(theta)
            betaI = mag * np.sin(theta)
        else:
            betaR = 0.0
            betaI = 0.0
        beta += [betaR, betaI]
    return np.array(beta, dtype=float), mask

spikes_units      = []
H_hist_units      = []
Z_til_real_units  = []
Z_bar_units       = []
beta_units        = []
band_masks        = []

for u in range(n_units):
    beta_u, mask_u = random_beta_with_mask(rng, n_bands=len(freqs_hz), k_active=3)
    spikes_u, H_hist_u, t_fine, Z_til_real_u, Z_bar_u = generate_spikes_from_latent(
        Z_true=Z_true,
        time=time,
        freqs_hz=freqs_hz,
        beta_true=beta_u,
        delta_spk=delta_spk,
        n_lags=n_lags,
        gamma_true=gamma_true,
        rng=rng
    )
    spikes_units.append(spikes_u)
    H_hist_units.append(H_hist_u)
    Z_til_real_units.append(Z_til_real_u)
    Z_bar_units.append(Z_bar_u)
    beta_units.append(beta_u)
    band_masks.append(mask_u)

spikes_units      = np.stack(spikes_units)        # (n_units, T_fine)
Z_til_real_units  = np.stack(Z_til_real_units)    # (n_units, n_bands, T_fine)
Z_bar_units       = np.stack(Z_bar_units)         # (n_units, T_blocks)
band_masks        = np.stack(band_masks)          # (n_units, n_bands)


# store all relevant information

all_data = {
    "duration_sec": duration_sec,
    "fs": fs,
    "seed": seed,
    "rng": rng,
    "n_units": n_units,
    "delta_spk": delta_spk,
    "n_lags": n_lags,
    "t_fine": t_fine,
    "spikes_units": spikes_units,
    "Z_til_real_units": Z_til_real_units,
    "Z_bar_units": Z_bar_units,
    "band_masks": band_masks,
    "beta_units": beta_units,
    "gamma_true": gamma_true,
    "lambda_true": lambda_true,
    "sigma_v_true": sigma_v_true,
    "sigma_eps_true": sigma_eps_true,
    "freqs_hz": freqs_hz,
    "freqs": freqs,
    "Z_true": Z_true,
    "Z_noisy": Z_noisy,
    "Z_true_all": Z_true_all,
    "lfp": lfp,
    "filtered_signal": filtered_signal,
    "tfr_raw": tfr_raw,
    "tfr": tfr,
    "tfr_ds": tfr_ds,
    "tfr_time_ds": tfr_time_ds,
}

import pickle
with open("all_data.pkl", "wb") as f:
    pickle.dump(all_data, f)

# test loading 
with open("all_data.pkl", "rb") as f:
    all_data = pickle.load(f)

print(all_data.keys())
print(all_data["spikes_units"].shape)
print(all_data["Z_til_real_units"].shape)
print(all_data["Z_bar_units"].shape)
print(all_data["band_masks"].shape)
print("Successfully loaded all data!")