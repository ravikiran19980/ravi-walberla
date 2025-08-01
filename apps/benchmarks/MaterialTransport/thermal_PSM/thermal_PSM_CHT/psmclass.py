import sympy as sp
from typing import cast

from dataclasses import replace
from collections import OrderedDict
from pystencils import Assignment, Field

from lbmpy import LBMConfig, create_lb_method, create_lb_collision_rule
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.methods import create_from_equilibrium
from lbmpy.creationfunctions import LbmCollisionRule
from lbmpy.moments import get_default_moment_set_for_stencil
from equilibirumCHT import DiscreteThermalMaxwellianCHT



def create_thermal_lb_method(lbm_config):
    stencil = lbm_config.stencil
    c_s_sq=sp.Rational(1, 3)
    compressible = lbm_config.compressible
    zero_centered = lbm_config.zero_centered

    # Symbols
    rho_Cp_T = sp.Symbol("rho_Cp_T")
    rho_f, omegaT_f, Cp_f = sp.symbols("rho_f, omegaT_f, Cp_f")
    rho_s, omegaT_s, Cp_s = sp.symbols("rho_s, omegaT_s, Cp_s")
    B = sp.Symbol("B")
    T = sp.Symbol("T")
    Cp_ref = 2 * Cp_s * Cp_f / (Cp_s + Cp_f)

    moments = get_default_moment_set_for_stencil(stencil)
    moment_to_relaxation_rate_dict = OrderedDict((m, omegaT_f) for m in moments)

    equilibrium_cht = DiscreteThermalMaxwellianCHT(
        stencil=stencil,
        rho_Cp_T=rho_Cp_T,
        u=sp.symbols("u_:3"),
        order=2,
        c_s_sq=c_s_sq,
        substitutions=None,
        temperature=T,
        Cp_ref=Cp_ref
    )

    cqc_cht = DensityVelocityComputation(
        stencil=stencil,
        compressible=lbm_config.compressible,
        zero_centered=lbm_config.zero_centered,
        force_model=None,
        c_s_sq=c_s_sq,
        density_symbol=rho_Cp_T
    )

    thermal_lb_method = create_from_equilibrium(stencil, equilibrium_cht, cqc_cht, moment_to_relaxation_rate_dict,
                                                zero_centered=zero_centered, force_model=None, **kwargs)

    return thermal_lb_method, cqc_cht


def create_psm_thermal_collision_rule(lbm_config, psm_output=None):
    thermal_lb_method, cqc_cht = create_thermal_lb_method(lbm_config)
    MaxParticlesPerCell = lbm_config.psm_config.MaxParticlesPerCell
    # Symbols
    rho_f, omegaT_f, Cp_f = sp.symbols("rho_f, omegaT_f, Cp_f")
    rho_s, omegaT_s, Cp_s = sp.symbols("rho_s, omegaT_s, Cp_s")
    tau_f, tau_s = sp.symbols("tau_f, tau_s")
    B = sp.Symbol("B")
    eps = sp.Symbol("eps")

    # Parameters
    parameters = [
        Assignment(eps, lbm_config.psm_config.fraction_field.center),
        Assignment(tau_f, 1 / omegaT_f),
        Assignment(tau_s, 1 / omegaT_s),
        Assignment(B, eps)
    ]

    psm_lb_config = replace(lbm_config, relaxation_rate=omegaT_f)
    parameters.append(Assignment(omegaT_f, (1 - B) * omegaT_f))

    zeroth_moment_symbol = thermal_lb_method.conserved_quantity_computation.zeroth_moment_symbol
    rho_cp_eff = ((1.0 - B.center) * rho_f * Cp_f * omegaT_f + B.center * rho_s * Cp_s * omegaT_s) / \
                 ((1 - B.center) * omegaT_f + B.center * omegaT_s)
    T = zeroth_moment_symbol / rho_cp_eff

    output_asms = []
    if psm_output:
        if "Temperature" in psm_output:
            output_asms.append(Assignment(psm_output["Temperature"].center, T))

    raw_col = thermal_lb_method.get_collision_rule(lbm_config=psm_lb_config)
    stencil = thermal_lb_method.stencil
    pre_collision_pdf_symbols = thermal_lb_method.pre_collision_pdf_symbols
    post_collision_pdf_symbols = thermal_lb_method.post_collision_pdf_symbols

    if cqc_cht.density_symbol not in raw_col.subexpressions_dict:
        cqc_cht_eqs = cqc_cht.equilibrium_input_equations_from_pdfs(pre_collision_pdf_symbols)
        density_eq = cqc_cht_eqs.main_assignments_dict[cqc_cht.density_symbol]
        raw_col.subexpressions.append(Assignment(cqc_cht.density_symbol, density_eq))

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

    solid_collisions = [0] * stencil.Q
    for p in range(MaxParticlesPerCell):
        fluid_eq_symbols = sp.symbols(f"f_eq_fluid_:{stencil.Q}")
        equilibrium_fluid = [
            Assignment(eq_sym, eq_val)
            for eq_sym, eq_val in zip(
                fluid_eq_symbols, thermal_lb_method.get_equilibrium_terms()
            )
        ]

        temp_fluid_subs = {sp.Symbol("T"): zeroth_moment_symbol / (rho_f * Cp_f)}
        Cp_fluid_subs = {sp.Symbol("Cp"): Cp_f}
        all_subs = {**temp_fluid_subs, **Cp_fluid_subs}
        equilibrium_fluid = [
            Assignment(asm.lhs, asm.rhs.subs(all_subs))
            for asm in equilibrium_fluid
        ]

        solid_eq_symbols = sp.symbols(f"f_eq_solid_:{stencil.Q}")
        equilibrium_solid = []
        for eq_s_symbol, eq_fluid in zip(solid_eq_symbols, equilibrium_fluid):
            eq_sol = eq_fluid.rhs
            vel_subs = {
                sp.Symbol(f"u_{i}"): lbm_config.psm_config.object_velocity_field.center(p * stencil.D + i)
                for i in range(stencil.D)
            }
            temp_solid_subs = {sp.Symbol("T"): zeroth_moment_symbol / (rho_s * Cp_s)}
            Cp_solid_subs = {sp.Symbol("Cp"): Cp_s}
            all_subs = {**vel_subs, **temp_solid_subs, **Cp_solid_subs}
            eq_sol = eq_sol.subs(all_subs)
            equilibrium_solid.append(Assignment(eq_s_symbol, eq_sol))

        for i, (f_eq_solid, f, offset) in enumerate(zip(solid_eq_symbols, pre_collision_pdf_symbols, stencil)):
            sc_term = lbm_config.psm_config.individual_fraction_field.center(p) * (
                    omegaT_s * (f_eq_solid - f)
            )
            solid_collisions[i] += sc_term

    solid_post_symbols = sp.symbols(f"f_post_solid_:{stencil.Q}")
    solid_post_asms = [Assignment(f_post_solid, solid_collisions[i])
                       for i, f_post_solid in enumerate(solid_post_symbols)]

    pdfs_update = [
        Assignment(f_post, f_post_fluid + B * f_post_solid)
        for f_post, f_post_fluid, f_post_solid in zip(
            post_collision_pdf_symbols, fluid_post_symbols, solid_collisions
        )
    ]

    subexps = parameters + raw_col.subexpressions + fluid_collisions + equilibrium_fluid + equilibrium_solid + solid_collisions
    mains = pdfs_update + output_asms + remaining_main_asms

    return LbmCollisionRule(thermal_lb_method, main_assignments=mains, subexpressions=subexps)
