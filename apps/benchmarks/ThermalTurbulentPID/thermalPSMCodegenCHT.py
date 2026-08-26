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
from pystencils.typing import TypedSymbol
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


sweep_block_size = (TypedSymbol("gpuBlockSize0", np.int32),
                    TypedSymbol("gpuBlockSize1", np.int32),
                    TypedSymbol("gpuBlockSize2", np.int32))

gpu_indexing_params = {'block_size': sweep_block_size}


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

    # Solid collision variant
    SC = int(config_tokens[1])

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

    # temperature PDFs and fields
    temperature_field = ps.fields(
        f"temperature_field({1}): {data_type}[3D]",
        layout=layout,
    )

    mean_temperature_field = ps.fields(f"mean_temperature_field({1}): {data_type}[{stencil_fluid.D}D]", layout=layout)
    sum_of_squares_temperature_field = ps.fields(f"sum_of_squares_temperature_field({1}): {data_type}[{stencil_fluid.D}D]", layout=layout)


    pdfs_temperature, pdfs_temperature_tmp = ps.fields(
        f"pdfs_temperature({stencil_temperature.Q}), pdfs_temperature_tmp({stencil_temperature.Q}): {data_type}[3D]",
        layout=layout,)

    # particle related fields (considering for all the particle, i.e: MaxParticlesPerCell
    particle_velocities, particle_forces, Bs = ps.fields(
        f"particle_v({MaxParticlesPerCell * stencil_fluid.D}), particle_f({MaxParticlesPerCell * stencil_fluid.D}), Bs({MaxParticlesPerCell}): {data_type}[3D]",
        layout=layout,
    )
    particle_temperatures = ps.fields(f"particle_t({MaxParticlesPerCell}) :{data_type}[3D]", layout=layout)


    # Solid fraction field
    B = ps.fields(f"b({1}): {data_type}[3D]", layout=layout)

    force_temperature_on_fluid = sp.Matrix([0,(rho_0)*alpha*(temperature_field.center - T0)*gravity_LBM, 0])

    # Fluid LBM optimisation
    lbm_fluid_opt = LBMOptimisation(
        cse_global=True,
        symbolic_field=pdfs_fluid,
        symbolic_temporary_field=pdfs_fluid_tmp,
        field_layout=layout,
    )


    # temperature LBM optimisation
    lbm_temperature_opt = LBMOptimisation(
        cse_global=True,
        symbolic_field=pdfs_temperature,
        symbolic_temporary_field=pdfs_temperature_tmp,
        field_layout=layout,
    )


    # Fluid PSM config
    psm_config_F = PSMConfig(
        fraction_field=B,
        object_velocity_field=particle_velocities,
        SC=SC,
        MaxParticlesPerCell=MaxParticlesPerCell,
        individual_fraction_field=Bs,
        particle_force_field=particle_forces,
    )

    psm_fluid_config = LBMConfig(
        stencil=stencil_fluid,
        method=Method.CUMULANT,
        relaxation_rate=omega_f,
        output={"velocity": velocity_field},
        force_model=ForceModel.GUO,
        force= tuple(force_on_fluid),
        compressible=True,
        psm_config=psm_config_F,
    )

    relaxation_rates = list(sp.symbols(f'omega_t_{n+1}') for n in range(stencil_temperature.Q))
    # temperature PSM config
    psm_temperature_config = LBMConfig(
        stencil=stencil_temperature,
        method=Method.CENTRAL_MOMENT,
        relaxation_rates=relaxation_rates,
        velocity_input=velocity_field,
        output={"density": temperature_field},
        compressible=True,
        continuous_equilibrium=True,
        zero_centered=False,
    )

    # =====================
    # Generate method
    # =====================

    method_fluid = create_lb_method(lbm_config=psm_fluid_config)
    method_temperature = create_thermal_lb_method(lbm_config=psm_temperature_config)
    method_temperature.override_weights(get_weights(stencil_temperature))
    init_velocity = sp.symbols("init_velocity_:3")

    pdfs_fluid_setter = macroscopic_values_setter(
        method_fluid, density=init_density_fluid, velocity=velocity_field.center_vector, pdfs=pdfs_fluid.center_vector
    )

    pdfs_temperature_setter = macroscopic_values_setter(
        method_temperature, density=temperature_field.center, velocity= velocity_field.center_vector,pdfs=pdfs_temperature.center_vector
    )

    # Use average velocity of all intersecting particles when setting PDFs (mandatory for SC=3)
    rhs = []
    for i, sub_exp in enumerate(pdfs_fluid_setter.subexpressions[-3:]):
        if(len(sub_exp.rhs.args) > 0):
            for summand in (sub_exp.rhs.args):
                rhs.append(summand * (1.0 - B.center))
        else:
            rhs.append(sub_exp.rhs * (1.0 - B.center))
        for p in range(2):
            rhs.append(particle_velocities(p * stencil_fluid.D + i) * Bs.center(p))
        pdfs_fluid_setter.subexpressions.remove(sub_exp)
        pdfs_fluid_setter.subexpressions.append(Assignment(sub_exp.lhs, Add(*rhs)))
        rhs = []



    ## for temperature

    sub_exp_temperature = pdfs_temperature_setter.subexpressions[0]
    rhs_temperature = []
    rhs_temperature.append((1 - B.center) * temperature_field.center + (B.center) * sp.Symbol("Tp"))
    #rhs_temperature.append(temperature_field.center)
    pdfs_temperature_setter.subexpressions.remove(sub_exp_temperature)
    pdfs_temperature_setter.subexpressions.append(Assignment(sub_exp_temperature.lhs, Add(*rhs_temperature)))
    #pdfs_temperature_setter.subexpressions.append(Assignment(sp.Symbol("c_s"), 1/sp.sqrt(3)))


    #print("temperature setter after manip ", pdfs_temperature_setter.subexpressions)

    # specify the target

    if ctx.gpu:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU
    max_threads = 256 if target == ps.Target.GPU else None
    node_collection_fluid = create_psm_update_rule(lbm_config=psm_fluid_config, lbm_optimisation=lbm_fluid_opt)
    collision_rule_temperature = create_lb_collision_rule(lb_method = method_temperature,lbm_config=psm_temperature_config,lbm_optimisation=lbm_temperature_opt)
    print("collision rule temp is  ", collision_rule_temperature)
    print("symbol   ", method_temperature.first_order_equilibrium_moment_symbols)

    # Build the accumulation as a pure SymPy expression (no kernel assignments involved)
    acc_expr = sum(
        Bs.center(p) * particle_temperatures.center(p)
        for p in range(MaxParticlesPerCell)  # ensure this is an int
    )

    T_init_fluid = sp.Symbol("T_init_fluid")  # kernel parameter

    @ps.kernel
    def initializetemperatureField():
        temperature_field.center @= (1 - B.center) * T_init_fluid + acc_expr

    initializetemperatureField_ac = ps.AssignmentCollection(initializetemperatureField)
    generate_sweep(ctx, "initializeTemperatureField", initializetemperatureField_ac)

    @ps.kernel
    def initializeVelocityFieldParticles():
        for p in range(MaxParticlesPerCell):
            for i in range(stencil_fluid.D):
                particle_velocities.center[p * stencil_fluid.D + i] @= B.center * velocity_field.center_vector[i]

    initializeVelocityFieldParticles_ac = ps.AssignmentCollection(initializeVelocityFieldParticles)
    generate_sweep(ctx, "initializeVelocityFieldParticles", initializeVelocityFieldParticles_ac, target=target)


    generate_sweep(
        ctx,
        "PSMFluidSweep",
        node_collection_fluid,
        field_swaps=[(pdfs_fluid, pdfs_fluid_tmp)],
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )

    generate_sweep(
        ctx,
        "PSMTemperatureSweep",
        create_lb_update_rule(collision_rule=collision_rule_temperature, lbm_config=psm_temperature_config, lbm_optimisation=lbm_temperature_opt),
        field_swaps=[(pdfs_temperature, pdfs_temperature_tmp)],
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )

    generate_pack_info_from_kernel(
        ctx,
        "PackInfoFluid",
        create_lb_update_rule(lbm_config=psm_fluid_config, lbm_optimisation=lbm_fluid_opt),
        target=target,
    )


    generate_pack_info_from_kernel(
        ctx,
        "PackInfoTemperature",
        create_lb_update_rule(lbm_config=psm_temperature_config, lbm_optimisation=lbm_temperature_opt),
        target=target,
    )

    generate_sweep(
        ctx,
        "InitializeFluidDomain",
        pdfs_fluid_setter,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
       # max_threads=max_threads,
    )
    generate_sweep(
        ctx,
        "InitializeTemperatureDomain",
        pdfs_temperature_setter,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )

    # Fluid Boundary conditions
    generate_boundary(
        ctx,
        "BC_Fluid_NoSlip",
        NoSlip(),
        method_fluid,
        field_name=pdfs_fluid.name,
        streaming_pattern="pull",
        target=target,
    )

    bc_velocity_fluid = sp.symbols("bc_velocity_fluid_:3")
    generate_boundary(
        ctx,
        "BC_Fluid_UBB",
        UBB(bc_velocity_fluid),
        method_fluid,
        field_name=pdfs_fluid.name,
        streaming_pattern="pull",
        target=target,
    )

    bc_density_fluid = sp.Symbol("bc_density_fluid")
    generate_boundary(
        ctx,
        "BC_Fluid_Density",
        FixedDensity(bc_density_fluid),
        method_fluid,
        field_name=pdfs_fluid.name,
        streaming_pattern="pull",
        target=target,
    )

    generate_boundary(
        ctx,
        "BC_Fluid_FreeSlip",
        FreeSlip(stencil_fluid),
        method_fluid,
        field_name=pdfs_fluid.name,
        streaming_pattern="pull",
        target=target,
    )

    generate_boundary(
        ctx,
        "BC_Fluid_Outflow",
        ExtrapolationOutflow((0,0,1),method_fluid),
        method_fluid,
        field_name=pdfs_fluid.name,
        streaming_pattern="pull",
        target=target,
    )

    # temperature boundary conditions
    dirichlet_bc_dynamic = DiffusionDirichlet(lambda *args: None, velocity_field, data_type=data_type)
    diffusion_data_handler = DiffusionDirichletAdditionalDataHandler(stencil_temperature, dirichlet_bc_dynamic)
    generate_boundary(ctx, 'BC_Temperature_DiffusionDirichlet_dynamic', dirichlet_bc_dynamic, method_temperature,
                      additional_data_handler=diffusion_data_handler,
                      target=target, streaming_pattern='pull', data_type=data_type)

    bc_density_temperature = sp.Symbol("bc_density_temperature")
    generate_boundary(
        ctx,
        "BC_Temperature_DiffusionDirichlet_static",
        DiffusionDirichlet(bc_density_temperature),
        method_temperature,
        field_name=pdfs_temperature.name,
        streaming_pattern="pull",
        target=target,
    )

    generate_boundary(
        ctx,
        "BC_Temperature_Neumann",
        NeumannByCopy(stencil_temperature),
        method_temperature,
        field_name=pdfs_temperature.name,
        streaming_pattern="pull",
        target=target,
    )



    # welford for the combined fluid-particle field:


    mean_velocity_field = ps.fields(f"mean_velocity_field({stencil_fluid.D}): {data_type}[{stencil_fluid.D}D]", layout=layout)
    sos_velocity_field = ps.fields(f"sos_velocity_field({stencil_fluid.D**2}): {data_type}[{stencil_fluid.D}D]", layout=layout)

    mean_temperature_field = ps.fields(f"mean_temperature_field({1}): {data_type}[{stencil_fluid.D}D]", layout=layout)
    sos_temperature_field = ps.fields(f"sos_temperature_field({1}): {data_type}[{stencil_fluid.D}D]", layout=layout)



    welford_update_velocity_field = welford_assignments(field=velocity_field, mean_field=mean_velocity_field,
                                                  sum_of_squares_field=sos_velocity_field)

    generate_sweep(
        ctx,
        "WelfordVelocity",
        welford_update_velocity_field,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,

    )

    welford_update_temperature_field = welford_assignments(field=temperature_field, mean_field=mean_temperature_field,
                                                     sum_of_squares_field=sos_temperature_field)
    generate_sweep(
        ctx,
        "WelfordTemperature",
        welford_update_temperature_field,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )



    stencil_typedefs = {"Stencil_Fluid_T": stencil_fluid, "CommunicationStencil_Fluid_T": stencil_fluid
        , "Stencil_Temperature_T":stencil_temperature, "CommunicationStencil_Temperature_T":stencil_temperature}
    field_typedefs = {
        "PdfField_fluid_T": pdfs_fluid,
        "DensityField_fluid_T": density_field,
        "VelocityField_fluid_T": velocity_field,
        "DensityField_temperature_T": temperature_field,
        "PdfField_temperature_T": pdfs_temperature,
        "DensityField_temperature_T": temperature_field,
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
        method_fluid, density=density_field, velocity=velocity_field.center_vector,pdfs=pdfs_fluid.center_vector
    )

    pdfs_temperature_getter = macroscopic_values_getter(
        method_temperature, density=temperature_field, velocity=None,pdfs=pdfs_temperature.center_vector
    )

    generate_sweep(
        ctx,
        "FluidMacroSetter",
        pdfs_fluid_setter,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )
    generate_sweep(
        ctx,
        "FluidMacroGetter",
        pdfs_fluid_getter,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )
    generate_sweep(
        ctx,
        "TemperatureMacroSetter",
        pdfs_temperature_setter,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )
    generate_sweep(
        ctx,
        "TemperatureMacroGetter",
        pdfs_temperature_getter,
        target=target,
        gpu_indexing_params=gpu_indexing_params,
        #max_threads=max_threads,
    )
