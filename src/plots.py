import matplotlib.pyplot as plt
def plot_lfp_and_spikes(lfp, time, spikes, t_fine, start_t=10.0, sec_to_plot=2.0):
    dt = float(time[1] - time[0])
    idx_raw_start  = int(round(start_t / dt))
    idx_raw_end    = int(round((start_t + sec_to_plot) / dt))
    idx_fine_start = int(round(start_t / (t_fine[1] - t_fine[0])))
    idx_fine_end   = int(round((start_t + sec_to_plot) / (t_fine[1] - t_fine[0])))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time[idx_raw_start:idx_raw_end],
            lfp[idx_raw_start:idx_raw_end],
            lw=0.8, label="LFP (raw)")

    seg_mask  = spikes[idx_fine_start:idx_fine_end].astype(bool)
    spk_times = t_fine[idx_fine_start:idx_fine_end][seg_mask]

    ax_spk = ax.twinx()
    ax_spk.vlines(spk_times, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  color="crimson", lw=1.2, label="spikes")
    ax_spk.set_yticks([])
    ax.set_xlim(start_t, start_t + sec_to_plot)
    ax.set_xlabel("time [s]")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    plt.show()

def plot_band_realparts(freqs_hz, t_fine, Z_til_real, spikes, start_t=10.0, sec_to_plot=2.0):
    import matplotlib.pyplot as plt
    idx_fine_start = int(round(start_t / (t_fine[1] - t_fine[0])))
    idx_fine_end   = int(round((start_t + sec_to_plot) / (t_fine[1] - t_fine[0])))
    spk_times = t_fine[idx_fine_start:idx_fine_end][spikes[idx_fine_start:idx_fine_end].astype(bool)]

    figs, axs = plt.subplots(len(freqs_hz), 1, figsize=(12, 2.2*len(freqs_hz)), sharex=True)
    if len(freqs_hz) == 1: axs = [axs]
    for i, f in enumerate(freqs_hz):
        axs[i].plot(t_fine[idx_fine_start:idx_fine_end],
                    Z_til_real[i, idx_fine_start:idx_fine_end], lw=0.8)
        ax2 = axs[i].twinx()
        ax2.vlines(spk_times, ymin=axs[i].get_ylim()[0], ymax=axs[i].get_ylim()[1],
                   color="crimson", lw=1.0)
        ax2.set_yticks([])
        axs[i].set_ylabel(f"{f:.0f} Hz")
    axs[-1].set_xlabel("time [s]")
    figs.tight_layout()
    plt.show()

def plot_band_specific_trajectory(selected_freqs, coef_ests, all_freqs, tfr_times):
    fig, axs = plt.subplots(len(selected_freqs), 1, figsize=(10, 10), squeeze=False)
    axs = axs.flatten()
    for i, freq in enumerate(selected_freqs):
        # find the closest frequency
        closest_freq = np.argmin(np.abs(all_freqs - freq))
        print(f"Closest frequency to {freq} Hz is {all_freqs[closest_freq]} Hz")
        # plot the trajectory of the selected frequency
        # axs[i].plot(tfr_times, coef_ests[closest_freq, :, :].real.mean(axis=0), label=f"Estimated {freq} Hz (real) Averaged over tapers")
        axs[i].plot(tfr_times, coef_ests[closest_freq, :, :].real.mean(axis=0), label=f"Estimated {freq} Hz (real) Averaged over tapers")
        axs[i].legend()
        axs[i].set_title(f"{freq} Hz Trajectory")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Amplitude (real)")
    plt.tight_layout()
    return fig, axs  # Return both the figure and axes for further manipulation


def plot_lfp_and_spikes(lfp, time, spikes, t_fine, start_t=10.0, sec_to_plot=2.0):
    import matplotlib.pyplot as plt
    dt = float(time[1] - time[0])
    idx_raw_start  = int(round(start_t / dt))
    idx_raw_end    = int(round((start_t + sec_to_plot) / dt))
    idx_fine_start = int(round(start_t / (t_fine[1] - t_fine[0])))
    idx_fine_end   = int(round((start_t + sec_to_plot) / (t_fine[1] - t_fine[0])))

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(time[idx_raw_start:idx_raw_end],
            lfp[idx_raw_start:idx_raw_end],
            lw=0.8, label="LFP (raw)")

    seg_mask  = spikes[idx_fine_start:idx_fine_end].astype(bool)
    spk_times = t_fine[idx_fine_start:idx_fine_end][seg_mask]

    ax_spk = ax.twinx()
    ax_spk.vlines(spk_times, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                  color="crimson", lw=1.2, label="spikes")
    ax_spk.set_yticks([])
    ax.set_xlim(start_t, start_t + sec_to_plot)
    ax.set_xlabel("time [s]")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    plt.show()