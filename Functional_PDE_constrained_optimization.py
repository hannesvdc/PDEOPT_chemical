from dolfin import *
from dolfin_adjoint import *
from ufl import And, conditional

import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from PDE_setup import inlet_concentrations, C_CO_inlet, C_O2_inlet, T_inlet, W, bcs, ds, OUTLET_ID, mesh

set_log_level(LogLevel.ERROR)

N_ZONES = 10
a_const_list = [Constant(3e4) for _ in range(N_ZONES)]
z = SpatialCoordinate(mesh)[0]
z_edges = np.linspace(0.0, L, N_ZONES + 1)
zone_indicators = []
for k in range(N_ZONES):
    zL, zR = z_edges[k], z_edges[k+1]
 
    inside = And(z >= zL, z < zR)
    chi_k = conditional( inside, 1.0, 0.0 )

    zone_indicators.append(chi_k)

def a_s_expression(a_consts):
    """
    Build UFL expression a_s(z) = sum_k a_const_list[k] * chi_k(z)
    where chi_k are the precomputed zone indicators.
    """
    a_s = 0
    for a_c, chi_k in zip(a_consts, zone_indicators):
        a_s += a_c * chi_k
    return a_s

N_alpha = 50
alpha_range = np.arange(N_alpha) / N_alpha

def solve_reactor_with_profile(y_CO_in, T_in, u_g, a_vals, alpha=1.0, U_init=None):
    """
    Solve the steady-state reactor PDE for given inlet CO mole fraction,
    inlet temperature, superficial velocity u_g, and a piecewise-constant
    a_s(z) profile defined by a_vals (len = N_ZONES).
    Returns (z_coords, C_CO(z), C_O2(z), T(z), U).
    """
    # Update inlet boundary values
    Cco_in, Co2_in = inlet_concentrations(y_CO_in, T_in)
    C_CO_inlet.assign(Cco_in)
    C_O2_inlet.assign(Co2_in)
    T_inlet.assign(T_in)

    # Build a_s(z) field from a_vals
    # a_const_list is assumed to be a list of dolfin.Constant, one per zone
    assert len(a_const_list) == len(a_vals), \
        "len(a_const_list) must match len(a_vals)"

    for a_c, val in zip(a_const_list, a_vals):
        a_c.assign(float(val))   # ensure plain float, in case val is np scalar
    a_s = a_s_expression(a_const_list)

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
             + alpha * a_s * r_local * v1 ) * dx

    F_o2 = ( D_ax * dot(grad(Co2), grad(v2))
             + u_g * Co2.dx(0) * v2
             + 0.5 * alpha * a_s * r_local * v2 ) * dx

    F_T = ( lambda_eff * dot(grad(T), grad(v3))
            + rho_g * cp_g * u_g * T.dx(0) * v3
            - q * alpha * a_s * r_local * v3
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

def forward_objective_and_gradient(a_vals, y_CO_in, T_in, u_g, U_init, verbose=False):
    get_working_tape().clear_tape()

    # Aolve at alpha=1.0 WITH annotation
    U = solve_reactor_with_profile(
        y_CO_in, T_in, u_g,
        a_vals=a_vals, alpha=1.0, U_init=U_init)
    Cco_fun, Co2_fun, T_fun = U.split()

    # Main Objective: Reduction CO Concentration.
    # Outlet CO "average" → in 1D this acts like evaluation at z=L
    C_in_val, _ = inlet_concentrations(y_CO_in, T_in)
    J_conv_form = (1.0 / C_in_val) * Cco_fun * ds(OUTLET_ID)

    # Penalty: excess temperature over inlet
    gamma = Constant(1e1)
    excess = T_fun - T_in
    smooth_relu = 0.5 * (excess + sqrt(excess**2 + 1.e-4))  # ~ max(dT, 0)
    excess_penalty_form = gamma * smooth_relu**2 * dx

    # Assemble the total objective
    J_form = 100.0 * J_conv_form + excess_penalty_form
    J = assemble(J_form)

    # Reduced functional + gradient wrt a_const
    m = [Control(a_c) for a_c in a_const_list]
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

def gradient_descent(a0_vals, y_CO_in, T_in, u_g, max_iters=10000, step_size=1.0, tol=1e-6, verbose=True):
    a_vals = a0_vals

    # Find a decent initial condition through alpha-continuation
    print(f'\n\nComputing an initial point near a = {a_vals} ...')
    U_init = None
    with stop_annotating():
        for alpha in alpha_range:
            U_init = solve_reactor_with_profile( y_CO_in, T_in, u_g, a_vals=a_vals,
                alpha=alpha,
                U_init=U_init,
            )
    print('... Done')

    print('\n\nStarting Optimization ..')
    J, grad, T_max, CR, penalty, U = forward_objective_and_gradient(a_vals, y_CO_in, T_in, u_g, U_init)
    a_history = [a_vals]
    J_history = [J]
    grad_history = [np.linalg.norm(grad)]
    conversion_rates = [CR]
    T_max_values = [T_max]
    penalties = [penalty]
    for k in range(max_iters):
        # take gradient step
        a_vals -= step_size * grad * a_vals 
        a_vals = np.maximum(a_vals, 0.0)

        U_init = U.copy(deepcopy=True)
        J, grad, T_max, CR, penalty, U = forward_objective_and_gradient(a_vals, y_CO_in, T_in, u_g, U_init)
        if verbose and k % 10 == 0:
            formatted_avals = " ".join(f"{v: .2e}" for v in a_vals)
            print(f"Iteration {k}: a = {formatted_avals}. Conversion: {CR:2e}, Tmax:  {T_max:2e}.  J = {J:.8e},  |grad| = {np.linalg.norm(a_vals * grad):.2e}")

        if np.linalg.norm(a_vals * grad) < tol:
            formatted_avals = " ".join(f"{v: .2e}" for v in a_vals)
            print(f'Optimization Finished at a = {formatted_avals} with |grad| = {np.linalg.norm(a_vals * grad):.2e}')
            break

        a_history.append(a_vals)
        J_history.append(J)
        grad_history.append(np.linalg.norm(grad))
        conversion_rates.append(CR)
        T_max_values.append(T_max)
        penalties.append(penalty)

    return a_vals, a_history, J_history, grad_history, conversion_rates, T_max_values, penalties

if __name__ == '__main__':
    y_CO_in = 0.034
    T_in = 740.0
    u_g = 0.25

    lr = 1.0
    a_vals = 3e4 * np.ones(N_ZONES)
    a_opt, a_history, J_history, grad_history, conversion_rates, T_max_values, penalties = gradient_descent(a_vals, y_CO_in, T_in, u_g, step_size=lr, verbose=True)

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