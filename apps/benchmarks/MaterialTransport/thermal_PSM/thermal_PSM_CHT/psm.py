import sympy as sp
from typing import cast

from dataclasses import replace

from pystencils import Assignment, Field

from lbmpy import LBMConfig, create_lb_method, create_lb_collision_rule
from lbmpy.relaxationrates import get_shear_relaxation_rate
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.creationfunctions import LbmCollisionRule


def psm_bounce_back_collision(
        lbm_config: LBMConfig,
        solid_fraction: sp.Expr,
        solid_velocity: tuple[sp.Expr, ...],
        psm_output: dict[str, Field] | None = None,
):
    """Derive an LBM collision rule from the given method description, while
    adding partially saturated bounce-back terms for a given solid fraction.

    Args:
        lbm_config: LB Method Configuration
        solid_fraction: Expression representing the solid fraction on each cell
        solid_velocity: Tuple of expressions representing the velocity at the boundary
        psm_output: Dictionary mapping names of PSM parameters (currently epsilon and B)
            to output fields
    """
    ε, B, τ, ω_psm = sp.symbols(
        "epsilon, B, tau, omega_psm"
    )  # solid fraction and weighting parameter B
    half = sp.Rational(1, 2)

    #   Determine PSM parameters
    raw_method = create_lb_method(lbm_config=lbm_config)
    assert raw_method is not None
    ω_s = get_shear_relaxation_rate(raw_method)

    parameters = [
        Assignment(ε, solid_fraction),
        Assignment(τ, 1 / ω_s),
        Assignment(B, (ε * (τ - half)) / ((1 - ε) + (τ - half))),
    ]

    #   Update relaxation rates
    raw_rrates: tuple[sp.Expr, ...] = raw_method.relaxation_rates

    psm_rrates = sp.symbols(f"psm_omega_:{len(raw_rrates)}")
    psm_lb_config = replace(
        lbm_config, relaxation_rate=None, relaxation_rates=psm_rrates
    )

    for ω_psm, ω_raw in zip(psm_rrates, raw_rrates):
        parameters.append(Assignment(ω_psm, (1 - B) * ω_raw))

    lb_method = create_lb_method(lbm_config=psm_lb_config)
    assert lb_method is not None

    #   Output params
    output_asms = []
    if psm_output:
        if "epsilon" in psm_output:
            output_asms.append(Assignment(psm_output["epsilon"].center, ε))
        if "B" in psm_output:
            output_asms.append(Assignment(psm_output["B"].center, B))

    #   Derive fluid collision
    raw_col: LbmCollisionRule = create_lb_collision_rule(
        lb_method=lb_method, lbm_config=psm_lb_config
    )

    stencil = lb_method.stencil
    pre_collision_pdf_symbols = lb_method.pre_collision_pdf_symbols
    post_collision_pdf_symbols = lb_method.post_collision_pdf_symbols

    #   Workaround: In ZC methods, rho might have been dropped
    cqc: DensityVelocityComputation = cast(
        DensityVelocityComputation, lb_method.conserved_quantity_computation
    )
    if cqc.density_symbol not in raw_col.subexpressions_dict:
        if cqc.density_deviation_symbol in raw_col.subexpressions_dict:
            raw_col.subexpressions.append(
                Assignment(
                    cqc.density_symbol,
                    cqc.density_deviation_symbol + cqc.background_density,
                    )
            )
        else:
            cqc_eqs = cqc.equilibrium_input_equations_from_pdfs(pre_collision_pdf_symbols)
            density_eq = cqc_eqs.main_assignments_dict[cqc.density_symbol]
            raw_col.subexpressions.append(
                Assignment(
                    cqc.density_symbol,
                    density_eq
                )
            )

    #   Move fluid collision terms to subexprs
    main_asms_dict = raw_col.main_assignments_dict

    fluid_post_symbols = sp.symbols(f"f_post_fluid_:{stencil.Q}")
    fluid_collisions = [
        Assignment(f_post_f, main_asms_dict[f_post])
        for f_post_f, f_post in zip(fluid_post_symbols, post_collision_pdf_symbols)
    ]

    remaining_main_asms = [
        Assignment(lhs, rhs)
        for lhs, rhs in main_asms_dict.items()
        if lhs not in post_collision_pdf_symbols
    ]

    #   Derive solid collision

    #    - Solid Equilibrium Terms
    fluid_eq_symbols = sp.symbols(f"f_eq_fluid_:{stencil.Q}")
    equilibrium_fluid = [
        Assignment(f_eq_symbol, f_eq_term)
        for f_eq_symbol, f_eq_term in zip(
            fluid_eq_symbols, lb_method.get_equilibrium_terms()
        )
    ]

    #    - Set up solid equilibrium
    solid_eq_symbols = sp.symbols(f"f_eq_solid_:{stencil.Q}")
    equilibrium_solid = []
    for eq_s_symbol, eq_fluid in zip(solid_eq_symbols, equilibrium_fluid):
        eq_sol = eq_fluid.rhs
        vel_subs = {sp.Symbol(f"u_{i}"): solid_velocity[i] for i in range(stencil.D)}
        eq_sol = eq_sol.subs(vel_subs)
        equilibrium_solid.append(Assignment(eq_s_symbol, eq_sol))

    #    - Derive solid collision operator
    solid_post_symbols = sp.symbols(f"f_post_solid_:{stencil.Q}")
    solid_collisions = []

    for i, (f_post_solid, f_eq_solid, f, offset) in enumerate(
            zip(solid_post_symbols, solid_eq_symbols, pre_collision_pdf_symbols, stencil)
    ):
        i_inv = stencil.inverse_index(offset)
        f_inv = pre_collision_pdf_symbols[i_inv]
        f_eq_inv = fluid_eq_symbols[i_inv]

        solid_collisions.append(
            Assignment(f_post_solid, (f_inv - f_eq_inv) - (f - f_eq_solid))
        )

    #   Combine into update rule
    pdfs_update = [
        Assignment(f_post, f_post_fluid + B * f_post_solid)
        for f_post, f_post_fluid, f_post_solid in zip(
            post_collision_pdf_symbols, fluid_post_symbols, solid_post_symbols
        )
    ]

    #   Finalize
    subexps = (
            parameters
            + raw_col.subexpressions
            + fluid_collisions
            + equilibrium_fluid
            + equilibrium_solid
            + solid_collisions
    )
    mains = pdfs_update + output_asms + remaining_main_asms

    return LbmCollisionRule(lb_method, main_assignments=mains, subexpressions=subexps)
