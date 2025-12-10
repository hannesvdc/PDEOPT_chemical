from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from PDE_setup import inlet_concentrations, C_CO_inlet, C_O2_inlet, T_inlet, W, bcs

# ----------------------------
# High-level solve function
# ----------------------------
def solve_reactor(y_CO_in, T_in, u_g, alpha=1.0, U_init=None):
    """
    Solve the steady-state reactor PDE for given inlet CO mole fraction,
    inlet temperature, and superficial velocity u_g.
    Returns (z_coords, C_CO, C_O2, T).
    """
    # Update inlet boundary values
    Cco_in, Co2_in = inlet_concentrations(y_CO_in, T_in)
    C_CO_inlet.assign(Cco_in)
    C_O2_inlet.assign(Co2_in)
    T_inlet.assign(T_in)

    # Fresh unknown & test functions
    U = Function(W)
    Cco, Co2, T = split(U)
    v1, v2, v3 = TestFunctions(W)

    # Reaction rate
    Co2_eff = conditional(ge(Co2, 1e-12), Co2, 1e-12)
    T_eff = conditional(ge(T, 200.0), T, 200.0)
    r_local = k0 * exp(-E_a / (R_gas * T_eff)) \
        * (K_CO * Cco * K_O2 * sqrt(Co2_eff)) \
          / (1.0 + K_CO * Cco + K_O2 * sqrt(Co2_eff))**2

    # Residual
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

    # Initial guess (important for convergence)
    if U_init is None:
        U.interpolate(Constant((Cco_in, Co2_in, T_in)))
    else:
        U.vector()[:] = U_init.vector()

    # Jacobian for Newton
    dU = TrialFunction(W)
    J = derivative(F, U, dU)

    # Nonlinear solve
    problem = NonlinearVariationalProblem(F, U, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters
    prm["newton_solver"]["maximum_iterations"] = 25
    prm["newton_solver"]["absolute_tolerance"] = 1e-8
    prm["newton_solver"]["relative_tolerance"] = 1e-8
    prm["newton_solver"]["error_on_nonconvergence"] = True
    prm["newton_solver"]["relaxation_parameter"] = 0.8

    set_log_level(LogLevel.ERROR)
    try:
        solver.solve()
        converged = True
    except RuntimeError:
        converged = False

    # Extract solutions on array
    Cco_sol, Co2_sol, T_sol = U.split(deepcopy=True)
    
    # Coordinates in DOF order for this scalar function space
    V = Cco_sol.function_space()
    x = V.tabulate_dof_coordinates().reshape(-1)

    # Values
    Cco_vals = Cco_sol.vector().get_local()
    Co2_vals = Co2_sol.vector().get_local()
    T_vals = T_sol.vector().get_local()

    # Sort by x just to be safe
    idx = np.argsort(x)
    x_sorted = x[idx]
    Cco_sorted = Cco_vals[idx]
    Co2_sorted = Co2_vals[idx]
    T_sorted = T_vals[idx]

    return x_sorted, Cco_sorted, Co2_sorted, T_sorted, U, converged

def computeSteadyState(y_CO_in, T_in, u_g, verbose=True):
    U_prev = None
    N_alpha = 100
    alpha_range = (1 + np.arange(N_alpha)) / N_alpha
    for alpha in alpha_range:
        if verbose:
            print(f'alpha = {alpha}')
        z, Cco_z, Co2_z, T_z, U_prev, _ = solve_reactor(y_CO_in, T_in, u_g, alpha=alpha, U_init=U_prev)

    return z, Cco_z, Co2_z, T_z, U_prev

if __name__ == "__main__":
    # Example: 3.4% CO, 750K - 800K inlet, moderate velocity
    y_CO_in = 0.034
    T_in_1 = 750.0 
    T_in_2 = 800.0 
    u_g = 1.0

    z_1, Cco_z_1, Co2_z_1, T_z_1, _ = computeSteadyState(y_CO_in, T_in_1, u_g)
    z_2, Cco_z_2, Co2_z_2, T_z_2, _ = computeSteadyState(y_CO_in, T_in_2, u_g)

    color_conv = "tab:blue"   # use for CO
    color_temp = "tab:red"    # use for T

    fig, ax_co = plt.subplots()

    # ---- Left axis: CO concentration ----
    l_co1, = ax_co.plot(z_1, Cco_z_1, color=color_conv,
                        label=rf"CO, $T_{{in}}={T_in_1}\,$K")
    l_co2, = ax_co.plot(z_2, Cco_z_2, color=color_conv, linestyle="--",
                        label=rf"CO, $T_{{in}}={T_in_2}\,$K")
    ax_co.set_xlabel(r"$z$")
    ax_co.set_ylabel(r"CO concentration", color=color_conv)
    ax_co.tick_params(axis="y", labelcolor=color_conv)
    ax_co.grid(True, alpha=0.3)

    # ---- Right axis: Temperature ----
    ax_T = ax_co.twinx()
    l_T1, = ax_T.plot(z_1, T_z_1, color=color_temp,
                    label=rf"$T(z), T_{{in}}={T_in_1}\,$K")
    l_T2, = ax_T.plot(z_2, T_z_2, color=color_temp, linestyle="--",
                    label=rf"$T(z), T_{{in}}={T_in_2}\,$K")
    ax_T.set_ylabel(r"$T$", color=color_temp)
    ax_T.tick_params(axis="y", labelcolor=color_temp)

    # ---- Legend (combine both axes) ----
    lines = [l_co1, l_co2, l_T1, l_T2]
    labels = [ln.get_label() for ln in lines]
    ax_co.legend(lines, labels, loc="best")

    plt.title(r"Temperature and CO profiles for two $T_{in}$ values")
    plt.tight_layout()
    plt.savefig('./images/pde_solutions.png', transparent=True)
    plt.show()