import copy
import sympy as sp
import pystencils as ps
from lbmpy.methods import CollisionSpaceInfo
from sympy.core.add import Add
from sympy.codegen.ast import Assignment
import sys
import numpy as np

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel
from lbmpy.partially_saturated_cells import PSMConfig

from lbmpy.boundaries import NoSlip, UBB, FixedDensity, FreeSlip, DiffusionDirichlet, NeumannByCopy, SimpleExtrapolationOutflow,ExtrapolationOutflow
from lbmpy.creationfunctions import (
    create_lb_update_rule,
    create_lb_method,
    create_psm_update_rule,
    create_lb_collision_rule,
)

from lbmpy.macroscopic_value_kernels import (
    macroscopic_values_getter,
    macroscopic_values_setter,
)

from pystencils_walberla import (
    CodeGeneration,
    generate_info_header,
    generate_sweep,
    generate_pack_info_from_kernel,
)
from lbmpy_walberla import generate_boundary
from lbmpy_walberla.additional_data_handler import DiffusionDirichletAdditionalDataHandler
from pystencils.cache import clear_cache
from thermalMethods import create_thermal_lb_method#,create_psm_thermal_collision_rule
from lbmpy.maxwellian_equilibrium import get_weights
from lbmpy.enums import Stencil, Method, CollisionSpace
from lbmpy.flow_statistics import welford_assignments
#clear_cache()



info_header = """
namespace walberla{{
namespace codegen{{
static constexpr uint_t flow_axis = {flow_axis};
static constexpr uint_t wall_axis = {wall_axis};
static constexpr uint_t remaining_axis= {remaining_axis};
using ScalarField_T = GhostLayerField< real_t, 1 >;
using VectorField_T = GhostLayerField< real_t, Stencil_Fluid_T::D >;
using TensorField_T = GhostLayerField< real_t, Stencil_Fluid_T::D*Stencil_Fluid_T::D >;
constexpr uint_t scalarSize= 1;
constexpr uint_t vectorSize= 3;
constexpr uint_t tensorSize= 9;
}};
}};
"""

def check_axis(flow_axis, wall_axis):
    assert flow_axis != wall_axis, "Axes must be distinct."
    assert all(0 <= axis < 3 for axis in (flow_axis, wall_axis)), "Axes must be between 0 and 2."


with CodeGeneration() as ctx:
    data_type = "float64" if ctx.double_accuracy else "float32"
    stencil_fluid = LBStencil(Stencil.D3Q19)
    stencil_temperature = LBStencil(Stencil.D3Q19)
    omega = sp.Symbol("omega")  # for now same for both the sweeps
    init_density_fluid = sp.Symbol("init_density_fluid")
    rho_0 = sp.Symbol("rho_0")
    T0 = sp.Symbol("T0")
    alpha = sp.Symbol("alpha")
    gravity_LBM = sp.Symbol("gravityLB")
    omega_f = sp.Symbol("omega_f")
    omega_t = sp.Symbol("omega_t")
    forcex = sp.Symbol("forcex")


    layout = "fzyx"
    config_tokens = ctx.config.split("_")
    print(config_tokens[0]," ", config_tokens[1])

    MaxParticlesPerCell = int(2)
    methods = {
        "srt": Method.SRT,
        "trt": Method.TRT,
        "mrt": Method.MRT,
        "cumulant": Method.MONOMIAL_CUMULANT,
        "srt-smagorinsky": Method.SRT,
        "trt-smagorinsky": Method.TRT,
    }


    flow_axis  = 0
    wall_axis  = 1
    remaining_axis = 3 - (flow_axis + wall_axis)

    check_axis(flow_axis=flow_axis, wall_axis=wall_axis)
    force_on_fluid = [0] * 3
    force_on_fluid[flow_axis] = forcex

    # Fluid PDFs and fields
    pdfs_fluid, pdfs_fluid_tmp, velocity_field, density_field = ps.fields(
        f"pdfs_fluid({stencil_fluid.Q}), pdfs_fluid_tmp({stencil_fluid.Q}), velocity_field({stencil_fluid.D}), density_field({1}): {data_type}[3D]",
        layout=layout,
    )
    mean_velocity_field = ps.fields(f"mean_velocity_field({stencil_fluid.D}): {data_type}[{stencil_fluid.D}D]", layout=layout)
    sum_of_squares_velocity_field = ps.fields(f"sum_of_squares_velocity_field({stencil_fluid.D**2}): {data_type}[{stencil_fluid.D}D]", layout=layout)


    # Fluid LBM optimisation
    lbm_fluid_opt = LBMOptimisation(
        cse_global=True,
        symbolic_field=pdfs_fluid,
        symbolic_temporary_field=pdfs_fluid_tmp,
        field_layout=layout,
    )


    lbm_config = LBMConfig(stencil=stencil_fluid,
                           method=Method.SRT,
                           force_model=ForceModel.GUO,
                           force=tuple(force_on_fluid),
                           relaxation_rate=omega,
                           compressible=True,
                           output={'velocity': velocity_field})
    update_rule = create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_fluid_opt)
    lbm_method = update_rule.method
    pdfs_setter = macroscopic_values_setter(lbm_method,
                                            1,
                                            velocity_field.center_vector,
                                            pdfs_fluid.center_vector)
    #   Macroscopic Values Setter
    generate_sweep(ctx, "TurbulentChannel_Setter", pdfs_setter, target=ps.Target.CPU, ghost_layers_to_include=1)

    generate_sweep(ctx, "TurbulentChannel_Sweep", update_rule, field_swaps=[(pdfs_fluid, pdfs_fluid_tmp)],target= ps.Target.CPU)


    # =====================
    # Generate method
    # =====================

    init_velocity = sp.symbols("init_velocity_:3")

    pdfs_fluid_setter = macroscopic_values_setter(
        lbm_method, density=init_density_fluid, velocity=velocity_field.center_vector, pdfs=pdfs_fluid.center_vector
    )


    # specify the target

    if ctx.gpu:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU


    generate_pack_info_from_kernel(
        ctx,
        "PackInfoFluid",
        create_lb_update_rule(lbm_config=lbm_config, lbm_optimisation=lbm_fluid_opt),
        target=target,
    )

    generate_sweep(ctx, "InitializeFluidDomain", pdfs_fluid_setter, target=target)

    # Fluid Boundary conditions
    generate_boundary(
        ctx,
        "BC_Fluid_NoSlip",
        NoSlip(),
        lbm_method,
        field_name=pdfs_fluid.name,
        streaming_pattern="pull",
        target=target,
    )


    # welford for the combined fluid-particle field:


    mean_velocity_field = ps.fields(f"mean_velocity_field({stencil_fluid.D}): {data_type}[{stencil_fluid.D}D]", layout=layout)
    sos_velocity_field = ps.fields(f"sos_velocity_field({stencil_fluid.D**2}): {data_type}[{stencil_fluid.D}D]", layout=layout)


    welford_update_velocity_field = welford_assignments(field=velocity_field, mean_field=mean_velocity_field,
                                                  sum_of_squares_field=sos_velocity_field)

    generate_sweep(ctx, "WelfordVelocity", welford_update_velocity_field, target=target)



    stencil_typedefs = {"Stencil_Fluid_T": stencil_fluid, "CommunicationStencil_Fluid_T": stencil_fluid
        , "Stencil_Temperature_T":stencil_temperature, "CommunicationStencil_Temperature_T":stencil_temperature}
    field_typedefs = {
        "PdfField_fluid_T": pdfs_fluid,
        "DensityField_fluid_T": density_field,
        "VelocityField_fluid_T": velocity_field,
    }


    info_header_params={
        'flow_axis': flow_axis,
        'wall_axis': wall_axis,
        'remaining_axis': remaining_axis,
    }
    generate_info_header(
        ctx,
        "GeneralInfoHeader",
        stencil_typedefs=stencil_typedefs,
        field_typedefs=field_typedefs,
        additional_code=info_header.format(**info_header_params)
    )

    # Getter & setter to compute moments from pdfs
    pdfs_fluid_getter = macroscopic_values_getter(
        lbm_method, density=density_field, velocity=velocity_field.center_vector,pdfs=pdfs_fluid.center_vector
    )

    generate_sweep(ctx, "FluidMacroSetter", pdfs_fluid_setter)
    generate_sweep(ctx, "FluidMacroGetter", pdfs_fluid_getter)