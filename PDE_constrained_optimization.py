from dolfin import *
from dolfin_adjoint import *

import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from PDE_setup import inlet_concentrations, C_CO_inlet, C_O2_inlet, T_inlet, W, bcs, ds, OUTLET_ID

set_log_level(LogLevel.ERROR)
a_const = Constant(3e4)

N_alpha = 50
alpha_range = np.arange(N_alpha) / N_alpha

def solve_reactor_with_profile(y_CO_in, T_in, u_g, a_val, alpha=1.0, U_init=None):
    """
    Solve the steady-state reactor PDE for given inlet CO mole fraction,
    inlet temperature, superficial velocity u_g, and a constant
    a_s(z) profile defined by a_val.
    
    Returns (z_coords, C_CO(z), C_O2(z), T(z), U).
    """
    # Update inlet boundary values
    Cco_in, Co2_in = inlet_concentrations(y_CO_in, T_in)
    C_CO_inlet.assign(Cco_in)
    C_O2_inlet.assign(Co2_in)
    T_inlet.assign(T_in)

    # Build a_s(z) field from a_vals
    a_const.assign(a_val)

    # Fresh unknown & test functions
    U = Function(W)
    Cco, Co2, T = split(U)
    v1, v2, v3 = TestFunctions(W)

    # Reaction rate (same regularization as before)
    Co2_eff = conditional(ge(Co2, 1e-12), Co2, 1e-12)
    T_eff   = conditional(ge(T, 200.0),   T,   200.0)
    r_local = k0 * exp(-E_a / (R_gas * T_eff)) \
        * (K_CO * Cco * K_O2 * sqrt(Co2_eff)) \
        / (1.0 + K_CO * Cco + K_O2 * sqrt(Co2_eff))**2

    # Residual with a_s_field instead of scalar a_s
    F_co = ( D_ax * dot(grad(Cco), grad(v1))
             + u_g * Cco.dx(0) * v1
             + alpha * a_const * r_local * v1 ) * dx

    F_o2 = ( D_ax * dot(grad(Co2), grad(v2))
             + u_g * Co2.dx(0) * v2
             + 0.5 * alpha * a_const * r_local * v2 ) * dx

    F_T = ( lambda_eff * dot(grad(T), grad(v3))
            + rho_g * cp_g * u_g * T.dx(0) * v3
            - q * alpha * a_const * r_local * v3
            + h_w * P_over_A * (T - T_wall) * v3 ) * dx

    F = F_co + F_o2 + F_T

    # Initial guess
    if U_init is None:
        U.interpolate(Constant((Cco_in, Co2_in, T_in)))
    else:
        U.vector()[:] = U_init.vector()

    # Jacobian for Newton
    dU = TrialFunction(W)
    J = derivative(F, U, dU)

    problem = NonlinearVariationalProblem(F, U, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["maximum_iterations"] = 25
    prm["newton_solver"]["absolute_tolerance"] = 1e-10
    prm["newton_solver"]["error_on_nonconvergence"] = False

    solver.solve()

    return U

def forward_objective_and_gradient(a_vec, y_CO_in, T_in, u_g, gamma, U_init, verbose=False):
    get_working_tape().clear_tape()

    # Aolve at alpha=1.0 WITH annotation
    U = solve_reactor_with_profile(
        y_CO_in, T_in, u_g,
        a_val=a_vec, alpha=1.0, U_init=U_init)
    Cco_fun, Co2_fun, T_fun = U.split()

    # Main Objective: Reduction CO Concentration.
    # Outlet CO "average" → in 1D this acts like evaluation at z=L
    C_in_val, _ = inlet_concentrations(y_CO_in, T_in)
    J_conv_form = (1.0 / C_in_val) * Cco_fun * ds(OUTLET_ID)

    # Penalty: excess temperature over inlet
    gamma_c = Constant(gamma)
    excess = T_fun - T_in
    smooth_relu = 0.5 * (excess + sqrt(excess**2 + 1.e-4))  # ~ max(dT, 0)
    excess_penalty_form = gamma_c * smooth_relu**2 * dx

    # Assemble the total objective
    J_form = 100.0 * J_conv_form + excess_penalty_form
    J = assemble(J_form)

    # Reduced functional + gradient wrt a_const
    m = Control(a_const)
    J_reduced = ReducedFunctional(J, m)
    grad_list = J_reduced.derivative()
    grad_J = np.array(grad_list, dtype=float)

    # Some additional information
    with stop_annotating():
        CR = 100.0 * (1.0 - assemble(J_conv_form))
        excess_penalty = assemble(excess_penalty_form)
        T_max = np.max(T_fun.vector().get_local())
        if verbose:
            print('max T, conversion, penalty =', T_max, CR, excess_penalty)
    return float(J), grad_J, T_max, CR, excess_penalty, U

def gradient_descent(a0, y_CO_in, T_in, u_g, gamma, max_iters=1000, step_size=1.0, tol=1e-6, verbose=True):
    a = a0

    # Find a decent initial condition through alpha-continuation
    print(f'\n\nComputing an initial point near a = {a} ...')
    U_init = None
    with stop_annotating():
        for alpha in alpha_range:
            U_init = solve_reactor_with_profile(
                y_CO_in, T_in, u_g,
                a_val=a,
                alpha=alpha,
                U_init=U_init,
            )
    print('... Done')

    print('\n\nStarting Optimization ..')
    J, grad, T_max, CR, penalty, U = forward_objective_and_gradient(a, y_CO_in, T_in, u_g, gamma, U_init)
    a_history = [a]
    J_history = [J]
    grad_history = [np.linalg.norm(grad)]
    conversion_rates = [CR]
    T_max_values = [T_max]
    penalties = [penalty]
    for k in range(max_iters):
        # take gradient step
        a -= step_size * grad * a
        a = np.maximum(a, 0.0)

        U_init = U.copy(deepcopy=True)
        J, grad, T_max, CR, penalty, U = forward_objective_and_gradient(a, y_CO_in, T_in, u_g, gamma, U_init)
        if verbose and k % 10 == 0:
            print(f"Iteration {k}: a = {a:2e}. Conversion: {CR:2e}, Tmax:  {T_max:2e}.  J = {J:.8e},  |grad| = {np.linalg.norm(a * grad):.2e}")

        if np.abs(a * grad) < tol:
            print(f'Optimization Finished at a = {a} with |grad| = {np.linalg.norm(a * grad):.2e}')
            break

        a_history.append(a)
        J_history.append(J)
        grad_history.append(np.linalg.norm(grad))
        conversion_rates.append(CR)
        T_max_values.append(T_max)
        penalties.append(penalty)

    return a, a_history, J_history, grad_history, conversion_rates, T_max_values, penalties

if __name__ == '__main__':
    y_CO_in = 0.034
    T_in = 740.0
    u_g = 0.25

    lr = 1.0
    gamma = 10000.0
    a_val = 3e4
    a_opt, a_history, J_history, grad_history, conversion_rates, T_max_values, penalties = gradient_descent(a_val, y_CO_in, T_in, u_g, gamma, step_size=lr, verbose=True)
    np.save(f'./data/gamma={gamma}.npy', np.array([a_opt, conversion_rates[-1], T_max_values[-1]]))

    # Plot the J-values and grad norms
    iters = np.arange(len(J_history))
    plt.semilogy(iters, a_history, label='a')
    plt.semilogy(iters, J_history, label='J(a)')
    plt.semilogy(iters, grad_history, label='dJ/da')
    plt.xlabel('Iteration')
    plt.legend()

    plt.figure()
    plt.semilogy(iters, conversion_rates, label=r'Conversion Rate $(\%)$')
    plt.semilogy(iters, penalties, label='Excess Temperature Penalty')
    plt.xlabel('Iteration')
    plt.legend()

    fig, ax1 = plt.subplots()
    color1 = "tab:blue"
    color2 = "tab:red"

    # Left y-axis: conversion
    ax1.set_xlabel(r"$a$")
    ax1.set_ylabel("CO conversion", color=color1)
    l1 = ax1.plot(a_history, conversion_rates, color=color1, label=r"Conversion Rate $(\%)$")
    ax1.tick_params(axis="y", labelcolor=color1)

    # Right y-axis: max temperature
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$T_{\max}$ [K]", color=color2)
    l2 = ax2.plot(a_history, T_max_values, color=color2, label=r"$T_{\max}$")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)
    ax1.set_title(r"CO Conversion and hot-spot vs $a$")
    fig.tight_layout()
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)

    plt.show()
