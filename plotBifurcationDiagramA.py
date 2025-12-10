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

def plotBifurcationDiagramA():
    data = np.load('./data/bifurcation_diagram_A.npy')
    a_values, T_max_values, CR_values, grad_penalties, excess_penalties, J_values, dJ_values = np.unstack(data)
    T_max_values = detectAndReplacePeeks(T_max_values)
    CR_values = detectAndReplacePeeks(CR_values)

    # Load the solutions for gamma
    gamma_list = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    a_list = []
    CR_list = []
    Tmax_list = []
    for gamma in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        sol = np.load(f'./data/gamma={gamma}.npy')
        a_list.append(sol[0])
        CR_list.append(sol[1])
        Tmax_list.append(sol[2])

    # Plot the  conversion rates and max temperatures
    fig, ax1 = plt.subplots()

    color1 = "tab:blue"
    color2 = "tab:red"

    # Left y-axis: conversion
    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel("CO conversion", color=color1)
    l1 = ax1.plot(a_values, CR_values / 100.0, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0.0, 1.05)  # nice [0,1] scale for conversion

    # Right y-axis: max temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$T_{\max}$ [K]", color=color2)
    l2 = ax2.plot(a_values, T_max_values, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    lines = []
    for idx in range(0,5):
        print(a_list[idx], CR_list[idx], Tmax_list[idx])
        li = ax1.scatter(a_list[idx], CR_list[idx]/100.0, label=fr"$\gamma = {gamma_list[idx]}$")
        lines.append(li)

    # Combined legend
    #lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    #ax1.set_title(r"CO Conversion and hot-spot vs $a$")
    fig.tight_layout()
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.savefig('./images/acurve.png', transparent=True)

    # Also plot the objective values, gradient penalties, excess penalties, J and dJ
    plt.figure()
    plt.semilogy(a_values, CR_values, label=r"Conversion Rate $(\%)$")
    plt.semilogy(a_values, grad_penalties, label='Gradient Penalty')
    plt.semilogy(a_values, excess_penalties, label='Excess Penalty')
    plt.xlabel(r"$a$")
    plt.legend()

    # Determine the optimal objective function
    plt.figure()
    CR_loss = 100.0 - CR_values
    gamma_values = [1.0, 10.0, 100.0]
    for gamma in gamma_values:
        plt.semilogy(a_values, CR_loss + gamma * detectAndReplacePeeks(excess_penalties), label=rf"$\gamma = {gamma}$")
    plt.xlabel(r"$a$")
    plt.ylabel(r'$J(a)$')
    plt.legend()
    
    plt.show()

if __name__ == '__main__':
    plotBifurcationDiagramA()