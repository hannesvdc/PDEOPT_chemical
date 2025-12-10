import numpy as np
import matplotlib.pyplot as plt

def in_between(y):
    is_increasing = np.greater_equal(y[1:-1], y[0:-2]) & np.greater_equal(y[2:], y[1:-1])
    is_decreasing = np.less_equal(y[1:-1], y[0:-2]) & np.less_equal(y[2:], y[1:-1])

    return is_increasing | is_decreasing

def detectAndReplacePeeks(y):

    # set threshold, e.g. 5 times median deviation
    spike_mask = np.logical_not(in_between(y))
    spike_indices = np.where(spike_mask)[0] + 1  # shift by 1 for interior indexing

    # replace spikes by neighbor average
    print('Number of spikes: ', len(spike_indices))
    y_clean = y.copy()
    y_clean[spike_indices] = 0.5 * (y_clean[spike_indices-3] + y_clean[spike_indices+3])
    return y_clean

def plotSCurve():
    lower = np.load('./data/lower.npy')
    T_in_vals, CR_values, T_max_values = np.unstack(lower)
    upper = np.load('./data/upper.npy')
    T_in_vals_upper, CR_values_upper, T_max_values_upper = np.unstack(upper)
    idxs_lower = (T_in_vals >= 600.0)
    idxs_upper = (T_in_vals_upper >= 600.0)

    # Smooth the upper branch slightly
    CR_values = detectAndReplacePeeks(CR_values[idxs_lower])
    T_max_values = detectAndReplacePeeks(T_max_values[idxs_lower])
    CR_values_upper = detectAndReplacePeeks(CR_values_upper[idxs_upper])
    T_max_values_upper = detectAndReplacePeeks(T_max_values_upper[idxs_upper])
    
    # Plot the  conversion rates and max temperatures
    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    color2 = "tab:red"

    # Left y-axis: conversion
    ax1.set_xlabel(r"$T_{\mathrm{in}}$ [K]")
    ax1.set_ylabel("CO conversion", color=color1)
    l1 = ax1.plot(T_in_vals[idxs_lower], CR_values, color=color1, label="Conversion")
    ax1.plot(T_in_vals_upper[idxs_upper], CR_values_upper, color=color1, label="Conversion")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.0, 1.05)  # nice [0,1] scale for conversion

    # Right y-axis: max temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$T_{\max}$ [K]", color=color2)
    l2 = ax2.plot(T_in_vals[idxs_lower], T_max_values, color=color2, label=r"$T_{\max}$")
    ax2.plot(T_in_vals_upper[idxs_upper], T_max_values_upper, color=color2, label=r"$T_{\max}$")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    ax1.set_title(r"CO Conversion and hot-spot vs $T_{in}$")
    fig.tight_layout()
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig('./images/scurve.png', transparent=True)
    plt.show()

if __name__ == '__main__':
    plotSCurve()