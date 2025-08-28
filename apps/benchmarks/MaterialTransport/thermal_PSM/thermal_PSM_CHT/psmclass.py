import lbmpy
import sympy as sp
from typing import Union, cast, Type

from dataclasses import dataclass, field, replace
from collections import OrderedDict
from pystencils import Assignment, Field, AssignmentCollection

from lbmpy import LBMConfig, create_lb_method, create_lb_collision_rule
from lbmpy.enums import Stencil, Method, CollisionSpace
from lbmpy.stencils import LBStencil
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.methods import create_from_equilibrium
from lbmpy.methods.creationfunctions import CollisionSpaceInfo
from lbmpy.creationfunctions import LbmCollisionRule
from lbmpy.moments import get_default_moment_set_for_stencil
from equilibirumCHT import DiscreteThermalMaxwellianCHT
from lbmpy.maxwellian_equilibrium import get_weights
#from equilibirumCHT import Cp

@dataclass
class ThermalPSMConfig(LBMConfig):

    # General Parameters
    temperature_output: dict[str, Field] | None = None,
    temperature_symbol: sp.Symbol = sp.Symbol("T")
    continuous_equilibrium: bool = False
    zero_centered: bool = False
    collision_space_info: CollisionSpaceInfo = CollisionSpaceInfo(CollisionSpace.POPULATIONS)
    energy_density_symbol: sp.Symbol = sp.Symbol("rho_Cp_T")

    # Fluid quantities:
    fluid_density: sp.Symbol = sp.Symbol("rho_f")
    fluid_specific_heat: sp.Symbol = sp.Symbol("Cp_f")

    # Particle quantities:
    fraction_field: Field = None
    object_velocity_field: Field = None
    SC: int = 5
    MaxParticlesPerCell: int = 1
    individual_fraction_field: Field = None
    particle_force_field = None,
    particle_temperature_field: Field = None
    particle_density: sp.Symbol = sp.Symbol("rho_p")
    particle_specific_heat: sp.Symbol = sp.Symbol("Cp_s")
    solid_relaxation_rate: sp.Symbol = sp.Symbol("omegaT_s")
    energy_field: Field = None
    temperature_field_output: Field = None
    heat_source: Union[int, float, Type[sp.Symbol]] = None


def create_thermal_lb_method(lbm_config: ThermalPSMConfig):
    stencil = lbm_config.stencil
    compressible = lbm_config.compressible
    zero_centered = lbm_config.zero_centered

    if compressible or zero_centered:
        raise ValueError("Compressible and zero-centered methods are not supported for thermal PSM.")
    if lbm_config.collision_space_info.collision_space != CollisionSpace.POPULATIONS:
        raise ValueError("Collision space must be POPULATIONS for thermal PSM")
    if lbm_config.delta_equilibrium:
        raise ValueError("Delta equilibrium must be set to None for thermal PSM")
    if lbm_config.equilibrium_order != 2:
        raise ValueError("Equilibrium order must be set to 2 for thermal PSM")
    if lbm_config.force_model is not None:
        raise ValueError("Force model is not supported for thermal PSM (yet)")
    if lbm_config.continuous_equilibrium:
        raise ValueError("Continuous equilibrium is not supported for thermal PSM")

    # Symbols
    rho_Cp_T = lbm_config.energy_density_symbol
    rho_f, omegaT_f, Cp_f = lbm_config.fluid_density, lbm_config.relaxation_rate, lbm_config.fluid_specific_heat
    rho_s, omegaT_s, Cp_s = lbm_config.particle_density, lbm_config.solid_relaxation_rate, lbm_config.particle_specific_heat

    moments = get_default_moment_set_for_stencil(stencil)
    moment_to_relaxation_rate_dict = OrderedDict((m, omegaT_f) for m in moments)

    equilibrium_cht = DiscreteThermalMaxwellianCHT(
        stencil=stencil,
        rho_Cp_T=rho_Cp_T,
        u=sp.symbols("u_:3"),
        order=2,
        c_s_sq=lbm_config.c_s_sq,
        substitutions=None,
        #temperature=T,
        Cp_ref=sp.Symbol("rho_Cp_ref")
    )

    cqc_cht = DensityVelocityComputation(
        stencil=stencil,
        compressible=lbm_config.compressible,
        zero_centered=lbm_config.zero_centered,
        force_model=None,
        c_s_sq=lbm_config.c_s_sq,
        density_symbol=rho_Cp_T
    )
    kwargs = {
        #'compressible': lbm_config.compressible,
        #'zero_centered': lbm_config.zero_centered,
        #'delta_equilibrium': None,
        #'equilibrium_order': 2,
        #'force_model': None,
        #'continuous_equilibrium': lbm_config.continuous_equilibrium,
        #'c_s_sq': lbm_config.c_s_sq,
        'collision_space_info': lbm_config.collision_space_info,
        #'fraction_field': lbm_config.fraction_field,
    }
    thermal_lb_method = create_from_equilibrium(stencil, equilibrium_cht, cqc_cht, moment_to_relaxation_rate_dict,
                                                zero_centered=zero_centered, force_model=None, **kwargs)
    thermal_lb_method.override_weights(get_weights(stencil))
    return thermal_lb_method,cqc_cht


def create_psm_thermal_collision_rule(lbm_config):
    thermal_lb_method,cqc_cht = create_thermal_lb_method(lbm_config)
    MaxParticlesPerCell = lbm_config.MaxParticlesPerCell
    psm_output = lbm_config.temperature_field_output
    # Symbols
    rho_f, omegaT_f, Cp_f = lbm_config.fluid_density, lbm_config.relaxation_rate, lbm_config.fluid_specific_heat
    rho_s, omegaT_s, Cp_s = lbm_config.particle_density, lbm_config.solid_relaxation_rate, lbm_config.particle_specific_heat
    B = lbm_config.fraction_field


    #   Update relaxation rates
    psm_lb_config = replace(
        lbm_config, relaxation_rate=omegaT_f
    )

    zeroth_moment_symbol = thermal_lb_method.conserved_quantity_computation.zeroth_order_moment_symbol
    rho_cp_eff = ((1.0 - B.center)* rho_f *Cp_f*omegaT_f + B.center*rho_s*Cp_s*omegaT_s)/((1-B.center)*omegaT_f + B.center*omegaT_s)
    temperature_symbol = zeroth_moment_symbol/rho_cp_eff

    #   Output params
    output_asms = []
    if psm_output:
        output_asms.append(Assignment(psm_output.center, temperature_symbol))

    stencil = thermal_lb_method.stencil

    #   Derive fluid collision
    raw_col: LbmCollisionRule = create_lb_collision_rule(
        lb_method=thermal_lb_method, lbm_config=psm_lb_config
    )

    pre_collision_pdf_symbols = thermal_lb_method.pre_collision_pdf_symbols
    post_collision_pdf_symbols = thermal_lb_method.post_collision_pdf_symbols
    print("raw coll subexp is  ", raw_col.subexpressions_dict)
    if cqc_cht.density_symbol not in raw_col.subexpressions_dict:

        cqc_cht_eqs = cqc_cht.equilibrium_input_equations_from_pdfs(pre_collision_pdf_symbols)
        density_eq = cqc_cht_eqs.main_assignments_dict[cqc_cht.density_symbol]
        raw_col.subexpressions.append(
            Assignment(
                cqc_cht.density_symbol,
                density_eq
            )
        )
    cqc_cht_eqs = cqc_cht.equilibrium_input_equations_from_pdfs(pre_collision_pdf_symbols)
    u_in = lbm_config.velocity_input
    if u_in is not None and isinstance(u_in, Field):
        u_in = u_in.center_vector
    cqe_main_assignments = cqc_cht_eqs.main_assignments_dict
    for u_sym, u in zip(cqc_cht.velocity_symbols, u_in):
        #cqe_main_assignments[u_sym] = u
        raw_col.subexpressions.append(
            Assignment(u_sym, u)
        )

    #   Move fluid collision terms to subexprs
    main_asms_dict = raw_col.main_assignments_dict
    fluid_post_symbols = sp.symbols(f"f_post_fluid_:{stencil.Q}")
    fluid_collisions = [
        Assignment(f_post_f, main_asms_dict[f_post] - f_pre)
        for f_post_f, f_post, f_pre in zip(fluid_post_symbols, post_collision_pdf_symbols, pre_collision_pdf_symbols)
    ]
    #print("fluid collisions are ", fluid_collisions)

    remaining_main_asms = [
        Assignment(lhs, rhs)
        for lhs, rhs in main_asms_dict.items()
        if lhs not in post_collision_pdf_symbols
    ]

    #   Derive solid collision
    equilibrium_fluid = []
    equilibrium_solid = []
    #    Fluid temperature Equilibrium Terms
    solid_collisions = [0]*stencil.Q
    fluid_eq_symbols = sp.symbols(f"f_eq_fluid_:{stencil.Q}")
    equilibrium_fluid = [
        Assignment(f_eq_symbol, f_eq_term)
        for f_eq_symbol, f_eq_term in zip(
            fluid_eq_symbols, thermal_lb_method.get_equilibrium_terms()
        )
    ]
    for p in range(MaxParticlesPerCell):

        equilibrium_fluid_for_solid_subs = equilibrium_fluid
        #print("eq fluid is ", equilibrium_fluid_for_solid_subs)

        #    - Set up solid equilibrium
        solid_eq_symbols = sp.symbols(f"f_eq_solid_:{stencil.Q}")
        equilibrium_solid = []
        for eq_s_symbol, eq_fluid in zip(solid_eq_symbols, equilibrium_fluid):
            eq_sol = eq_fluid
            vel_subs = {sp.Symbol(f"u_{i}"): lbm_config.object_velocity_field.center(p * stencil.D + i) for i in range(stencil.D)}
            temp_solid_subs = {sp.Symbol("rho_f"): rho_s}
            Cp_solid_subs   = {sp.Symbol("Cp_f"): Cp_s}
            all_subs = {**vel_subs,**Cp_solid_subs, **temp_solid_subs}
            eq_sol = eq_sol.rhs.subs(all_subs)
            equilibrium_solid.append(Assignment(eq_s_symbol, eq_sol))

        #print("eq solid is ", equilibrium_solid)
        for i, (f_eq_solid, f, offset) in enumerate(
                zip(solid_eq_symbols, pre_collision_pdf_symbols, stencil)
        ):

            sc_term = lbm_config.individual_fraction_field.center(p) * (
                (
                        omegaT_s * (f_eq_solid - f)
                )

            )
            solid_collisions[i] += sc_term

    #   Derive solid collision operator
    solid_post_symbols = sp.symbols(f"f_post_solid_:{stencil.Q}")

    #for i,f_post_solid in enumerate(solid_post_symbols):
    #    Assignment(f_post_solid, solid_collisions[i])

    solid_post_assignments = []  # <- collect assignments here
    stencil_weights = get_weights(stencil)
    #print("stencil weights are ", stencil_weights)
    heat_source = lbm_config.heat_source
    for i, f_post_solid in enumerate(solid_post_symbols):
        if heat_source is not None:
            solid_post_assignments.append(
                Assignment(f_post_solid, solid_collisions[i] + stencil_weights[i] * heat_source * B.center)
            )
        else:
            solid_post_assignments.append(
                Assignment(f_post_solid, solid_collisions[i])
            )


    #   Combine into update rule
    pdfs_update = [
        Assignment(f_post, f_pre + (1- B.center)*f_post_fluid +  f_post_solid)
        for f_post, f_pre, f_post_fluid, f_post_solid in zip(
            post_collision_pdf_symbols, pre_collision_pdf_symbols, fluid_post_symbols, solid_post_symbols
        )
    ]

    #   Finalize
    subexps = (
            raw_col.subexpressions
            + fluid_collisions
            + equilibrium_fluid
            + equilibrium_solid
            + solid_post_assignments
    )
    mains = pdfs_update + output_asms + remaining_main_asms
    #print("main asses are  ", output_asms)
    for item in mains:
        if not isinstance(item, Assignment):
            print("❌ Invalid main assignment:", item, type(item))

    for item in subexps:
        if not isinstance(item, Assignment):
            print("❌ Invalid subexpression:", item, type(item))


    return LbmCollisionRule(thermal_lb_method, main_assignments=mains, subexpressions=subexps)
