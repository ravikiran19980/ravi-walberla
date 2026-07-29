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
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.moments import is_even
from lbmpy.relaxationrates import relaxation_rate_from_magic_number

@dataclass
class ThermalPSMConfig(LBMConfig):

    # General Parameters
    temperature_output: dict[str, Field] | None = None
    temperature_symbol: sp.Symbol = sp.Symbol("T")
    continuous_equilibrium: bool = False
    zero_centered: bool = False
    collision_space_info: CollisionSpaceInfo = field(default_factory=lambda: CollisionSpaceInfo(CollisionSpace.POPULATIONS))
    energy_density_symbol: sp.Symbol = sp.Symbol("rho_Cp_T")

    # Fluid quantities:
    fluid_density: sp.Symbol = sp.Symbol("rho_f")
    fluid_specific_heat: sp.Symbol = sp.Symbol("Cp_f")
    fluid_conductivity: sp.Symbol = sp.Symbol("k_f")

    # Particle quantities:
    fraction_field: Field = None
    object_velocity_field: Field = None
    SC: int = 5
    MaxParticlesPerCell: int = 1
    individual_fraction_field: Field = None
    particle_force_field: Field = None
    particle_temperature_field: Field = None
    particle_density: sp.Symbol = sp.Symbol("rho_p")
    particle_specific_heat: sp.Symbol = sp.Symbol("Cp_s")
    solid_relaxation_rate: sp.Symbol = sp.Symbol("omegaT_s")
    energy_field: Field = None
    temperature_field_output: Field = None
    heat_source: Union[int, float, Type[sp.Symbol]] = None
    solid_conductivity: sp.Symbol = sp.Symbol("k_s")



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



    if lbm_config.method == Method.SRT:
        print("SRT collision selected")
        moments = get_default_moment_set_for_stencil(stencil)
        moment_to_relaxation_rate_dict = OrderedDict((m, lbm_config.relaxation_rate) for m in moments)

    else:
        print("TRT collision selected")
        moments = get_default_moment_set_for_stencil(stencil)
        relaxation_rate_odd_moments = lbm_config.relaxation_rate
        relaxation_rate_even_moments = relaxation_rate_from_magic_number(relaxation_rate_odd_moments, sp.Rational(1,4))
        moment_to_relaxation_rate_dict = OrderedDict([(m, relaxation_rate_even_moments if is_even(m) else relaxation_rate_odd_moments)
                            for m in moments])


    equilibrium_cht = DiscreteThermalMaxwellianCHT(
        stencil=stencil,
        rho_Cp_T=rho_Cp_T,
        u=sp.symbols("u_:3"),
        order=2,
        c_s_sq=lbm_config.c_s_sq,
        substitutions=None,
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
        'collision_space_info': lbm_config.collision_space_info,
    }
    thermal_lb_method = create_from_equilibrium(stencil, equilibrium_cht, cqc_cht, moment_to_relaxation_rate_dict,
                                                zero_centered=zero_centered, force_model=None, **kwargs)
    thermal_lb_method.override_weights(get_weights(stencil))
    return thermal_lb_method,cqc_cht


def create_psm_thermal_collision_rule(lbm_config):

    psm_output = lbm_config.temperature_field_output
    # Symbols
    rho_f, omegaT_f, Cp_f = lbm_config.fluid_density, lbm_config.relaxation_rate, lbm_config.fluid_specific_heat
    rho_s, omegaT_s, Cp_s = lbm_config.particle_density, lbm_config.solid_relaxation_rate, lbm_config.particle_specific_heat
    B = lbm_config.fraction_field
    k_eff = (1 - B.center)*lbm_config.fluid_conductivity + B.center*lbm_config.solid_conductivity
    rho_Cp_ref = (2*rho_f*Cp_f*rho_s*Cp_s)/(rho_f*Cp_f + rho_s*Cp_s)
    thermal_diffusivity_eff = k_eff / rho_Cp_ref
    omega_eff = relaxation_rate_from_lattice_viscosity(thermal_diffusivity_eff)
    #   Update relaxation rates

    thermal_lb_method, cqc_cht = create_thermal_lb_method(
        replace(lbm_config, relaxation_rate=omega_eff),
    )

    zeroth_moment_symbol = thermal_lb_method.conserved_quantity_computation.zeroth_order_moment_symbol
    rho_cp_eff = ((1.0 - B.center)* rho_f *Cp_f + B.center*rho_s*Cp_s)
    temperature_symbol = zeroth_moment_symbol/rho_cp_eff

    #   Output params
    output_asms = []
    if psm_output:
        output_asms.append(Assignment(psm_output.center, temperature_symbol))

    stencil = thermal_lb_method.stencil

    #   Derive fluid collision
    raw_col: LbmCollisionRule = create_lb_collision_rule(
        lb_method=thermal_lb_method, lbm_config=replace(lbm_config, relaxation_rate=omega_eff)
    )
    print("raw collision main assignments ", raw_col.main_assignments)
    print("raw collision subexpressions ", raw_col.subexpressions)

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
    #print("main ass dict ", main_asms_dict)
    fluid_post_symbols = sp.symbols(f"f_post_fluid_:{stencil.Q}")
    fluid_collisions = [
        Assignment(f_post_f, main_asms_dict[f_post])
        for f_post_f, f_post in zip(fluid_post_symbols, post_collision_pdf_symbols)
    ]

    stencil_weights = get_weights(stencil)
    heat_source = lbm_config.heat_source
    fluid_post_symbols_with_heatsource = sp.symbols(f"f_post_fluid_heat_:{stencil.Q}")
    fluid_collisions_with_heatsource = [
        Assignment(f_post_f, main_asms_dict[f_post_f] + stencil_weights[i] * heat_source * B.center)
        for f_post_f, i in zip(post_collision_pdf_symbols, range(stencil.Q))
    ]

    # Have this for debugging and understanding purpose:

    #for f_post_f, i in zip(post_collision_pdf_symbols, range(stencil.Q)):
    #    print(main_asms_dict[f_post_f])

    #print(fluid_collisions)
    #print(fluid_collisions_with_heatsource)

    remaining_main_asms = [
        Assignment(lhs, rhs)
        for lhs, rhs in main_asms_dict.items()
        if lhs not in post_collision_pdf_symbols
    ]
    print("remaining main asms are ", remaining_main_asms)

    #   Finalize
    subexps = (
            [Assignment(sp.Symbol("rho_cp") , rho_cp_eff)]
            + raw_col.subexpressions
            + [Assignment(sp.Symbol("T") , temperature_symbol)]
    )
    mains =     fluid_collisions_with_heatsource  + output_asms + remaining_main_asms
    for item in mains:
        if not isinstance(item, Assignment):
            print("❌ Invalid main assignment:", item, type(item))

    for item in subexps:
        if not isinstance(item, Assignment):
            print("❌ Invalid subexpression:", item, type(item))


    return LbmCollisionRule(thermal_lb_method, main_assignments=mains, subexpressions=subexps)
