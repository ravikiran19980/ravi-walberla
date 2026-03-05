import lbmpy
import sympy as sp
from typing import Union, cast, Type

from dataclasses import dataclass, field, replace
from collections import OrderedDict

from lbmpy.moment_transforms import PdfsToCentralMomentsByShiftMatrix
from pystencils import Assignment, Field, AssignmentCollection

from lbmpy import LBMConfig, create_lb_method, create_lb_collision_rule
from lbmpy.enums import Stencil, Method, CollisionSpace
from lbmpy.stencils import LBStencil
from lbmpy.methods.conservedquantitycomputation import DensityVelocityComputation
from lbmpy.methods import create_from_equilibrium
from lbmpy.methods.creationfunctions import CollisionSpaceInfo
from lbmpy.creationfunctions import LbmCollisionRule
from lbmpy.moments import get_default_moment_set_for_stencil
from lbmpy.equilibrium import ContinuousHydrodynamicMaxwellian
from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.relaxationrates import relaxation_rate_from_lattice_viscosity
from lbmpy.moments import is_even, get_order
from lbmpy.relaxationrates import relaxation_rate_from_magic_number
from lbmpy.methods.momentbased import CentralMomentBasedLbMethod
from lbmpy.methods.cumulantbased import CumulantBasedLbMethod




@dataclass
class ThermalCentralMomentBasedLbMethod(CentralMomentBasedLbMethod):
    def __init__(self, stencil, equilibrium, relaxation_dict,
                 conserved_quantity_computation=None,
                 force_model=None, zero_centered=False,
                 central_moment_transform_class=PdfsToCentralMomentsByShiftMatrix):
        super(ThermalCentralMomentBasedLbMethod, self).__init__(stencil, equilibrium, relaxation_dict,
                                                                conserved_quantity_computation, force_model,
                                                                zero_centered,central_moment_transform_class)
    def override_weights(self, weights):
        self._weights = weights

def create_thermal_lb_method(lbm_config):
    stencil = lbm_config.stencil
    compressible = lbm_config.compressible
    zero_centered = lbm_config.zero_centered

    if   zero_centered:
        raise ValueError("Zero-centered methods are not supported for thermal LB.")

    if lbm_config.delta_equilibrium:
        raise ValueError("Delta equilibrium must be set to None for thermal LB")

    if lbm_config.force_model is not None:
        raise ValueError("Force model is not supported for thermal LB (yet)")

    if not lbm_config.continuous_equilibrium:
        raise ValueError("Set Continuous equilibrium to True for thermal LB")



    equilibrium_thermal = ContinuousHydrodynamicMaxwellian(dim=stencil.D, compressible=True,
                                                           deviation_only=False,
                                                           order=2, rho=sp.Symbol('T'),c_s_sq=sp.Rational(1, 3),u = tuple(lbm_config.velocity_input.center_vector)) #u = lbm_config.velocity_input.center_vector
    cqc_thermal = DensityVelocityComputation(
        stencil=stencil,
        compressible=lbm_config.compressible,
        zero_centered=lbm_config.zero_centered,
        force_model=None,
        c_s_sq=sp.Symbol("c_s")**2,
        density_symbol=sp.Symbol("T")
    )



    if lbm_config.method == Method.SRT:

        print("SRT thermal collision selected")
        moments = get_default_moment_set_for_stencil(stencil)
        moment_to_relaxation_rate_dict = OrderedDict((m, lbm_config.relaxation_rates[0]) for m in moments)

        return create_from_equilibrium(stencil, equilibrium_thermal, cqc_thermal, moment_to_relaxation_rate_dict,
                                zero_centered=zero_centered, force_model=None)

    elif lbm_config.method == Method.TRT :

        print("TRT thermal collision selected")
        moments = get_default_moment_set_for_stencil(stencil)
        relaxation_rate_odd_moments = lbm_config.relaxation_rates[0]

        relaxation_rate_even_moments = relaxation_rate_from_magic_number(relaxation_rate_odd_moments, sp.Rational(1,4))
        moment_to_relaxation_rate_dict = OrderedDict([(m, relaxation_rate_even_moments if is_even(m) else relaxation_rate_odd_moments)
                            for m in moments])

        return create_from_equilibrium(stencil, equilibrium_thermal, cqc_thermal, moment_to_relaxation_rate_dict,
                                zero_centered=zero_centered, force_model=None)

    elif lbm_config.method == Method.CENTRAL_MOMENT:

        print("Central Moment thermal collision selected")
        cspace = CollisionSpaceInfo(CollisionSpace.CENTRAL_MOMENTS)
        central_nested_moments = get_default_moment_set_for_stencil(stencil)
        central_nested_moments = [(c,) for c in central_nested_moments]

        relaxation_rates = lbm_config.relaxation_rates
        moment_to_relaxation_rate_dict = get_thermal_relaxation_info_dict(relaxation_rates, central_nested_moments)

        return ThermalCentralMomentBasedLbMethod(stencil, equilibrium_thermal, moment_to_relaxation_rate_dict, conserved_quantity_computation=cqc_thermal,
                                          force_model=None, zero_centered=False,
                                          central_moment_transform_class=PdfsToCentralMomentsByShiftMatrix)

    elif lbm_config.method == Method.MONOMIAL_CUMULANT:

        print("Cumulant thermal collision selected")
        thermal_cumulants = get_default_moment_set_for_stencil(stencil)
        nested_thermal_cumulants = [(c,) for c in thermal_cumulants]

        relaxation_rates = lbm_config.relaxation_rates
        mom_to_rr_dict = get_thermal_relaxation_info_dict(relaxation_rates, nested_thermal_cumulants)

        cspace = CollisionSpaceInfo(collision_space=CollisionSpace.CUMULANTS)


        return CumulantBasedLbMethod(stencil, equilibrium_thermal, mom_to_rr_dict, conserved_quantity_computation=cqc_thermal,
                                              force_model=None, zero_centered=False,
                                              central_moment_transform_class=PdfsToCentralMomentsByShiftMatrix,
                                              cumulant_transform_class=cspace.cumulant_transform_class)
    else:
        raise NotImplementedError("Only SRT, TRT, Central Moment MRT and Cumulant implemented for Thermal LB")


def get_thermal_relaxation_info_dict(relaxation_rates, nested_moments):

    result = OrderedDict()
    maximum_moment_order = max(
        get_order(moment)
        for group in nested_moments
        for moment in group
    )
    print("maximum order of the moments considered is  ", maximum_moment_order)
    relaxation_rates = relaxation_rates[:maximum_moment_order]

    for group in nested_moments:

        for moment in group:
            if get_order(moment) == 0:           # conserved moments are default with 0
                result[moment] = 0.0

            if get_order(moment) == 1:
                result[moment] = relaxation_rates[0]

            if get_order(moment) == 2:
                result[moment] = relaxation_rates[1]

            if get_order(moment) == 3:
                result[moment] = relaxation_rates[2]

            if get_order(moment) == 4:
                result[moment] = relaxation_rates[3]

            if get_order(moment) == 5:
                result[moment] = relaxation_rates[4]

            if get_order(moment) == 6:                   # maximum possible order
                result[moment] = relaxation_rates[5]

    return result




