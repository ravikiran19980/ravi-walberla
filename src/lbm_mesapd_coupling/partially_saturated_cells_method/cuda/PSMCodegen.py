import copy
import sympy as sp
import pystencils as ps
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.codegen.ast import Assignment

from lbmpy import LBMConfig, LBMOptimisation, LBStencil, Method, Stencil, ForceModel

from lbmpy.boundaries import NoSlip, UBB, FixedDensity, FreeSlip
from lbmpy.creationfunctions import create_lb_update_rule, create_lb_method
from pystencils.astnodes import Conditional, SympyAssignment, Block
from pystencils.node_collection import NodeCollection
from lbmpy.creationfunctions import create_lb_collision_rule
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

# Based on the following paper: https://doi.org/10.1016/j.compfluid.2017.05.033

info_header = """
const char * infoStencil = "{stencil}";
const char * infoStreamingPattern = "{streaming_pattern}";
const char * infoCollisionSetup = "{collision_setup}";
const bool infoCseGlobal = {cse_global};
const bool infoCsePdfs = {cse_pdfs};
"""

with CodeGeneration() as ctx:
    data_type = "float64" if ctx.double_accuracy else "float32"
    stencil = LBStencil(Stencil.D3Q27)
    omega = sp.Symbol("omega")
    init_density = sp.Symbol("init_density")
    init_velocity = sp.symbols("init_velocity_:3")
    pdfs_inter = sp.symbols("pdfs_inter:" + str(stencil.Q))
    layout = "fzyx"
    MaxParticlesPerCell = 2
    config_tokens = ctx.config.split("_")
    methods = {"srt": Method.SRT, "trt": Method.TRT}
    # Solid collision variant
    SC = int(config_tokens[1][2])
    split = bool(
        int(config_tokens[2][1])
    )  # Splitting scheme was introduced in the following paper: https://doi.org/10.1016/j.powtec.2022.117556

    pdfs, pdfs_tmp, velocity_field, density_field = ps.fields(
        f"pdfs({stencil.Q}), pdfs_tmp({stencil.Q}), velocity_field({stencil.D}), density_field({1}): {data_type}[3D]",
        layout=layout,
    )

    particle_velocities, particle_forces, Bs = ps.fields(
        f"particle_v({MaxParticlesPerCell * stencil.D}), particle_f({MaxParticlesPerCell * stencil.D}), Bs({MaxParticlesPerCell}): {data_type}[3D]",
        layout=layout,
    )

    # Solid fraction field
    B = ps.fields(f"b({1}): {data_type}[3D]", layout=layout)

    psm_opt = LBMOptimisation(
        cse_global=True,
        symbolic_field=pdfs,
        symbolic_temporary_field=pdfs_tmp,
        field_layout=layout,
    )

    # TODO: set magic number?
    psm_config = LBMConfig(
        stencil=stencil,
        method=methods[config_tokens[0]],
        relaxation_rate=omega,
        force=sp.symbols("F_:3"),
        force_model=ForceModel.LUO,
        compressible=False,
    )

    # =====================
    # Code generation for the fluid part (regular LBM collision with overlap fraction as prefactor)
    # =====================

    method = create_lb_method(lbm_config=psm_config)
    collision_rule = create_lb_collision_rule(
        lbm_config=psm_config, lbm_optimisation=psm_opt
    )

    collision_rhs = []
    for assignment in collision_rule.main_assignments:
        rhsSum = 0
        for arg in assignment.rhs.args:
            if arg not in method.pre_collision_pdf_symbols:
                rhsSum += (1 - B.center) * arg
            else:
                rhsSum += arg
        collision_rhs.append(rhsSum)

    # =====================
    # Code generation for the solid parts
    # =====================

    forces_rhs = [0] * MaxParticlesPerCell * stencil.D
    solid_collisions = [0] * stencil.Q
    for p in range(MaxParticlesPerCell):
        equilibriumFluid = method.get_equilibrium_terms()
        equilibriumSolid = []
        for eq in equilibriumFluid:
            eq_sol = eq
            for i in range(stencil.D):
                eq_sol = eq_sol.subs(
                    sp.Symbol("u_" + str(i)),
                    particle_velocities.center(p * stencil.D + i),
                )
            if split:
                eq_sol = eq_sol.subs(
                    sp.Symbol("delta_rho"),
                    sp.Symbol("delta_rho_inter"),
                )
            equilibriumSolid.append(eq_sol)
        if split:
            equilibriumFluidTmp = equilibriumFluid
            equilibriumFluid = []
            for eq in equilibriumFluidTmp:
                for i in range(stencil.D):
                    eq = eq.subs(
                        sp.Symbol("u_" + str(i)),
                        sp.Symbol("u_" + str(i) + "_inter"),
                    )
                eq = eq.subs(
                    sp.Symbol("delta_rho"),
                    sp.Symbol("delta_rho_inter"),
                )
                equilibriumFluid.append(eq)

        # Assemble right-hand side of collision assignments
        # Add solid collision part to collision right-hand side and forces right-hand side
        for i, (eqFluid, eqSolid, f, offset) in enumerate(
            zip(
                equilibriumFluid,
                equilibriumSolid,
                method.pre_collision_pdf_symbols,
                stencil,
            )
        ):
            inverse_direction_index = stencil.stencil_entries.index(
                stencil.inverse_stencil_entries[i]
            )
            if SC == 1:
                solid_collision = Bs.center(p) * (
                    (
                        method.pre_collision_pdf_symbols[inverse_direction_index]
                        - equilibriumFluid[inverse_direction_index]
                    )
                    - (f - eqSolid)
                )
            elif SC == 2:
                solid_collision = Bs.center(p) * (
                    (eqSolid - f) + (1 - omega) * (f - eqFluid)
                )
            elif SC == 3:
                solid_collision = Bs.center(p) * (
                    (
                        method.pre_collision_pdf_symbols[inverse_direction_index]
                        - equilibriumSolid[inverse_direction_index]
                    )
                    - (f - eqSolid)
                )
            else:
                raise ValueError("Only SC=1, SC=2 and SC=3 are supported.")
            solid_collisions[i] += solid_collision
            for j in range(stencil.D):
                forces_rhs[p * stencil.D + j] -= solid_collision * int(offset[j])

    # =====================
    # Assemble update rule
    # =====================

    # Assemble collision assignments
    collision_assignments = []
    if not split:
        for d, c, sc in zip(
            method.post_collision_pdf_symbols, collision_rhs, solid_collisions
        ):
            collision_assignments.append(ps.Assignment(d, c + sc))
    else:
        # First, add only fluid collision
        for d, c in zip(pdfs_inter, collision_rhs):
            collision_assignments.append(ps.Assignment(d, c))

        # Second, define quantities to compute the equilibrium as functions of the pdfs_inter
        cqc_inter = (
            method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
                pdfs_inter, False
            )
        )
        for cq in cqc_inter.all_assignments:
            for i in range(stencil.D):
                cq = cq.subs(
                    sp.Symbol("u_" + str(i)), sp.Symbol("u_" + str(i) + "_inter")
                )
                cq = cq.subs(
                    sp.Symbol("vel" + str(i) + "Term"),
                    sp.Symbol("vel" + str(i) + "Term_inter"),
                )
            cq = cq.subs(sp.Symbol("delta_rho"), sp.Symbol("delta_rho_inter"))
            cq = cq.subs(sp.Symbol("rho"), sp.Symbol("rho_inter"))
            collision_assignments.append(cq)

        # Third, add solid collision using quantities from pdfs_inter
        for d, d_inter, sc in zip(
            method.post_collision_pdf_symbols, pdfs_inter, solid_collisions
        ):
            collision_assignments.append(ps.Assignment(d, d_inter + sc))

    # Add hydrodynamic force calculations to collision assignments
    for p in range(MaxParticlesPerCell):
        for i in range(stencil.D):
            collision_assignments.append(
                ps.Assignment(
                    particle_forces.center(p * stencil.D + i),
                    forces_rhs[p * stencil.D + i],
                )
            )

    # Define quantities to compute the equilibrium as functions of the pdfs
    # TODO: maybe incorporate some of these assignments into the subexpressions
    # cqc = method.conserved_quantity_computation.equilibrium_input_equations_from_pdfs(
    #    method.pre_collision_pdf_symbols, False
    # )

    up = ps.AssignmentCollection(
        collision_assignments, subexpressions=collision_rule.subexpressions
    )
    output_eqs = method.conserved_quantity_computation.output_equations_from_pdfs(
        method.pre_collision_pdf_symbols, psm_config.output
    )
    up = up.new_merged(output_eqs)
    up.method = method

    # Create assignment collection for the complete update rule
    lbm_update_rule = create_lb_update_rule(
        collision_rule=up, lbm_config=psm_config, lbm_optimisation=psm_opt
    )

    # =====================
    # Add conditionals for the solid parts
    # =====================

    # Transform the assignment collection into a node collection to be able to add conditionals
    node_collection = NodeCollection.from_assignment_collection(lbm_update_rule)

    conditionals = []
    for p in range(MaxParticlesPerCell):
        # One conditional for every potentially overlapping particle
        conditional_assignments = []

        # Move force computations to conditional
        for i in range(stencil.D):
            for node in node_collection.all_assignments:
                if type(node) == SympyAssignment and node.lhs == particle_forces.center(
                    p * stencil.D + i
                ):
                    conditional_assignments.append(node)
                    node_collection.all_assignments.remove(node)

        # Move solid collisions to conditional
        for node in node_collection.all_assignments:
            if type(node) == SympyAssignment and type(node.rhs) == Add:
                rhs = node.rhs.args
                # Maximum one solid collision for each potentially overlapping particle per assignment
                solid_collision = next(
                    (
                        summand
                        for summand in rhs
                        if type(summand) == Mul and Bs.center(p) in summand.args
                    ),
                    None,
                )
                if solid_collision is not None:
                    conditional_assignments.append(
                        SympyAssignment(
                            copy.deepcopy(node.lhs),
                            Add(solid_collision, copy.deepcopy(node.lhs)),
                        )
                    )
                    node.rhs = Add(
                        *[
                            summand
                            for summand in rhs
                            if not (
                                type(summand) == Mul and Bs.center(p) in summand.args
                            )
                        ]
                    )

        conditional = Conditional(Bs.center(p) > 0.0, Block(conditional_assignments))

        conditionals.append(conditional)
        # Append conditional at the end of previous conditional
        # because n+1 particles can only overlap if at least n particles overlap
        if p > 0:
            conditionals[-2].true_block.append(conditional)

    # Add first conditional to node collection, the other conditionals are nested inside the first one
    node_collection.all_assignments.append(conditionals[0])

    # Print nodes into file
    def build_markdown(n):
        if type(n) == Conditional:
            tex = "$if " + str(n.condition_expr) + "$: <br />\n"
            for a in n.true_block.args:
                tex += build_markdown(a)
            tex += "$fi$ <br />\n"
            assert n.false_block is None
            return tex
        return n._repr_html_() + " <br />\n"

    markdown_string = ""
    for assignment in node_collection.all_assignments:
        markdown_string += build_markdown(assignment)

    markdown_string = markdown_string.replace("_tmp", "_{tmp}")
    with open("PSMSweep.md", "w") as f:
        f.write(markdown_string)

    pdfs_setter = macroscopic_values_setter(
        method, init_density, init_velocity, pdfs.center_vector
    )

    # Use average velocity of all intersecting particles when setting PDFs (mandatory for SC=3)
    for i, sub_exp in enumerate(pdfs_setter.subexpressions[-3:]):
        rhs = []
        for summand in sub_exp.rhs.args:
            rhs.append(summand * (1.0 - B.center))
        for p in range(MaxParticlesPerCell):
            rhs.append(particle_velocities(p * stencil.D + i) * Bs.center(p))
        pdfs_setter.subexpressions.remove(sub_exp)
        pdfs_setter.subexpressions.append(Assignment(sub_exp.lhs, Add(*rhs)))

    if ctx.cuda:
        target = ps.Target.GPU
    else:
        target = ps.Target.CPU

    # Generate files
    generate_sweep(
        ctx,
        "PSMSweep",
        node_collection,
        field_swaps=[(pdfs, pdfs_tmp)],
        target=target,
    )

    generate_sweep(
        ctx,
        "PSMSweepSplit",
        node_collection,
        field_swaps=[(pdfs, pdfs_tmp)],
        target=target,
        inner_outer_split=True,
    )

    generate_pack_info_from_kernel(ctx, "PSMPackInfo", lbm_update_rule, target=target)

    generate_sweep(ctx, "InitializeDomainForPSM", pdfs_setter, target=target)

    # Boundary conditions
    generate_boundary(
        ctx,
        "PSM_NoSlip",
        NoSlip(),
        method,
        field_name=pdfs.name,
        streaming_pattern="pull",
        target=target,
    )

    bc_velocity = sp.symbols("bc_velocity_:3")
    generate_boundary(
        ctx,
        "PSM_UBB",
        UBB(bc_velocity),
        method,
        field_name=pdfs.name,
        streaming_pattern="pull",
        target=target,
    )

    bc_density = sp.Symbol("bc_density")
    generate_boundary(
        ctx,
        "PSM_Density",
        FixedDensity(bc_density),
        method,
        field_name=pdfs.name,
        streaming_pattern="pull",
        target=target,
    )

    generate_boundary(
        ctx,
        "PSM_FreeSlip",
        FreeSlip(stencil),
        method,
        field_name=pdfs.name,
        streaming_pattern="pull",
        target=target,
    )

    # Info header containing correct template definitions for stencil and fields
    infoHeaderParams = {
        "stencil": stencil.name,
        "streaming_pattern": psm_config.streaming_pattern,
        "collision_setup": config_tokens[0],
        "cse_global": int(psm_opt.cse_global),
        "cse_pdfs": int(psm_opt.cse_pdfs),
    }

    stencil_typedefs = {"Stencil_T": stencil, "CommunicationStencil_T": stencil}
    field_typedefs = {
        "PdfField_T": pdfs,
        "DensityField_T": density_field,
        "VelocityField_T": velocity_field,
    }

    generate_info_header(
        ctx,
        "PSM_InfoHeader",
        stencil_typedefs=stencil_typedefs,
        field_typedefs=field_typedefs,
        additional_code=info_header.format(**infoHeaderParams),
    )

    # Getter & setter to compute moments from pdfs
    setter_assignments = macroscopic_values_setter(
        method,
        velocity=velocity_field.center_vector,
        pdfs=pdfs.center_vector,
        density=1.0,
    )
    getter_assignments = macroscopic_values_getter(
        method,
        density=density_field,
        velocity=velocity_field.center_vector,
        pdfs=pdfs.center_vector,
    )
    generate_sweep(ctx, "PSM_MacroSetter", setter_assignments)
    generate_sweep(ctx, "PSM_MacroGetter", getter_assignments)
