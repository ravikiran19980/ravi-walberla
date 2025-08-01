import sympy as sp
from typing import cast

from dataclasses import replace
from collections import OrderedDict
from pystencils import Assignment, Field

from lbmpy import LBMConfig, create_lb_method, create_lb_collision_rule
from lbmpy.relaxationrates import get_shear_relaxation_rate
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.methods import create_from_equilibrium
from lbmpy.creationfunctions import LbmCollisionRule
from lbmpy.moments import get_default_moment_set_for_stencil
from equilibirumCHT import DiscreteThermalMaxwellianCHT
from src.lbm_mesapd_coupling.partially_saturated_cells_method.codegen.PSMCodegen import lbm_config, MaxParticlesPerCell


def psm_bounce_back_collision(
        lbm_config: LBMConfig,
        solid_fraction: sp.Expr,
        solid_velocity: tuple[sp.Expr, ...],
        psm_output: dict[str, Field] | None = None,

        particle_temperature_field: Field = None,
        solid_omega: sp.Symbol  = sp.Symbol("omegaT_s"),
        energy_field: Field = None,
        temperature_field: Field = None,
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
    # reading parameters from lbm config
    assert lbm_config.psm_config is not None
    fraction_field = lbm_config.psm_config.fraction_field
    object_velocity_field = lbm_config.psm_config.object_velocity_field
    solid_collision_operator = lbm_config.psm_config.SC
    MaxParticlesPerCell = lbm_config.psm_config.MaxParticlesPerCell
    individual_fraction_field = lbm_config.psm_config.individual_fraction_field


    stencil = lbm_config.stencil
    rho_Cp_T = sp.Symbol("rho_Cp_T")
    c_s_sq=sp.Rational(1, 3)
    compressible = lbm_config.compressible
    zero_centered = lbm_config.zero_centered

    rho_f,tau_f, omegaT_f, Cp_f = sp.symbols(
        "rho_f, tau_f,omegaT_f, Cp_f"
    )
    rho_s,tau_s, omegaT_s,Cp_s = sp.symbols(
        "rho_s, tau_s,omegaT_s, Cp_s"
    )
    eps, B = sp.symbols(
        "eps,B"
    )
    kwargs = {
        'compressible': lbm_config.compressible,
        'zero_centered': lbm_config.zero_centered,
        'delta_equilibrium': lbm_config.delta_equilibrium,
        'equilibrium_order': lbm_config.equilibrium_order,
        'force_model': lbm_config.force_model,
        'continuous_equilibrium': lbm_config.continuous_equilibrium,
        'c_s_sq': lbm_config.c_s_sq,
        'collision_space_info': lbm_config.collision_space_info,
        'fraction_field': fraction_field,
    }

    T = sp.Symbol("T")  # symbol for the temperature in each cell
    Cp_ref = 2*Cp_s*Cp_f/(Cp_s + Cp_f)

    moments = get_default_moment_set_for_stencil(stencil)
    moment_to_relaxation_rate_dict = OrderedDict([(m, omegaT_f) for m in moments])

    equilibrium_cht = DiscreteThermalMaxwellianCHT(stencil, rho_Cp_T=rho_Cp_T, u=sp.symbols("u_:3"),
                                                   order=2,
                                                   c_s_sq=c_s_sq, substitutions=None, temperature = T, Cp_ref=Cp_ref)

    cqc_cht = DensityVelocityComputation(stencil, compressible, zero_centered, force_model=None,
                                         c_s_sq=c_s_sq, density_symbol=rho_Cp_T,
                                         )

    thermal_lb_method =  create_from_equilibrium(stencil, equilibrium_cht, cqc_cht, moment_to_relaxation_rate_dict,
                                   zero_centered=zero_centered, force_model=None, **kwargs)

    assert thermal_lb_method is not None

    parameters = [
        Assignment(eps, fraction_field),
        Assignment(tau_f, 1/omegaT_f),
        Assignment(tau_s, 1/omegaT_s),
        Assignment(B, eps),   # weighting_T = 1 for temperature PSM
    ]

    #   Update relaxation rates
    psm_lb_config = replace(
        lbm_config, relaxation_rate=omegaT_f
    )

    parameters.append(Assignment(omegaT_f, (1 - B) * omegaT_f))

    zeroth_moment_symbol = thermal_lb_method.conserved_quantity_computation.zeroth_moment_symbol
    rho_cp_eff = ((1.0 - B.center)* rho_f *Cp_f*omegaT_f + B.center*rho_s*Cp_s*omegaT_s)/((1-B.center)*omegaT_f + B.center*omegaT_s)
    T = zeroth_moment_symbol/rho_cp_eff

    #   Output params
    output_asms = []
    if psm_output:
        if "Temperature" in psm_output:
            output_asms.append(Assignment(psm_output["Temperature"].center, T))


    #   Derive fluid collision
    raw_col: LbmCollisionRule = create_lb_collision_rule(
        lb_method=thermal_lb_method, lbm_config=psm_lb_config
    )

    stencil = thermal_lb_method.stencil
    pre_collision_pdf_symbols = thermal_lb_method.pre_collision_pdf_symbols
    post_collision_pdf_symbols = thermal_lb_method.post_collision_pdf_symbols

    if cqc_cht.density_symbol not in raw_col.subexpressions_dict:

        cqc_cht_eqs = cqc_cht.equilibrium_input_equations_from_pdfs(pre_collision_pdf_symbols)
        density_eq = cqc_cht_eqs.main_assignments_dict[cqc_cht.density_symbol]
        raw_col.subexpressions.append(
            Assignment(
                cqc_cht.density_symbol,
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

    #    Fluid temperature Equilibrium Terms
    solid_collisions = [0]*stencil.Q
    for p in range(MaxParticlesPerCell):
        fluid_eq_symbols = sp.symbols(f"f_eq_fluid_:{stencil.Q}")
        equilibrium_fluid = [
            Assignment(f_eq_symbol, f_eq_term)
            for f_eq_symbol, f_eq_term in zip(
                fluid_eq_symbols, thermal_lb_method.get_equilibrium_terms()
            )
        ]
        temp_fluid_subs = {sp.Symbol("T"): zeroth_moment_symbol/(rho_f*Cp_f)}
        Cp_fluid_subs   = {sp.Symbol("Cp"): Cp_f}
        all_subs = {**temp_fluid_subs, **Cp_fluid_subs}
        equilibrium_fluid = [
        Assignment(asm.lhs, asm.rhs.subs(all_subs)) for asm in equilibrium_fluid]



    #    - Set up solid equilibrium
        solid_eq_symbols = sp.symbols(f"f_eq_solid_:{stencil.Q}")
        equilibrium_solid = []
        for eq_s_symbol, eq_fluid in zip(solid_eq_symbols, equilibrium_fluid):
            eq_sol = eq_fluid.rhs
            vel_subs = {sp.Symbol(f"u_{i}"): object_velocity_field.center(p * stencil.D + i) for i in range(stencil.D)}
            temp_solid_subs = {sp.Symbol("T"): zeroth_moment_symbol/(rho_s*Cp_s)}
            Cp_solid_subs   = {sp.Symbol("Cp"): Cp_s}
            all_subs = {**vel_subs, **temp_solid_subs, **Cp_solid_subs}
            eq_sol = eq_sol.subs(all_subs)
            equilibrium_solid.append(Assignment(eq_s_symbol, eq_sol))


        for i, (f_eq_solid, f, offset) in enumerate(
                zip(solid_eq_symbols, pre_collision_pdf_symbols, stencil)
        ):

            sc_term = lbm_config.psm_config.individual_fraction_field.center(p) * (
                (
                         omegaT_s * (f_eq_solid - f)
                )

            )
            solid_collisions[i] += sc_term

    #   Derive solid collision operator
    solid_post_symbols = sp.symbols(f"f_post_solid_:{stencil.Q}")

    for i,f_post_solid in enumerate(solid_post_symbols):
        Assignment(f_post_solid, solid_collisions[i])

    #   Combine into update rule
    pdfs_update = [
        Assignment(f_post, f_post_fluid + B * f_post_solid)
        for f_post, f_post_fluid, f_post_solid in zip(
            post_collision_pdf_symbols, fluid_post_symbols, solid_collisions
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

    return LbmCollisionRule(thermal_lb_method, main_assignments=mains, subexpressions=subexps)
