waLBerla_link_files_to_builddir("*.prm")

if (WALBERLA_BUILD_WITH_CODEGEN)
    if (NOT WALBERLA_BUILD_WITH_GPU_SUPPORT OR (WALBERLA_BUILD_WITH_GPU_SUPPORT AND (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 OR WALBERLA_BUILD_WITH_HIP)))
        foreach(solid_collision 1 2 3)

            set(config sc_${solid_collision})
            message(STATUS "Config value in thermal PSM is: ${config}" )
            waLBerla_generate_target_from_python(NAME smooth_profile_lbm_${config}
                    FILE thermalPSMCodegenCHT.py
                    CODEGEN_CFG ${config}
                    OUT_FILES LBMFluidSweep.h LBMFluidSweep.${CODEGEN_FILE_SUFFIX}
                    PSMFluidSweep.h PSMFluidSweep.${CODEGEN_FILE_SUFFIX}
                    LBMFluidSplitSweep.h LBMFluidSplitSweep.${CODEGEN_FILE_SUFFIX}
                    PSMFluidSweepSplit.h PSMFluidSweepSplit.${CODEGEN_FILE_SUFFIX}
                    PSMEnergySweep.h PSMEnergySweep.${CODEGEN_FILE_SUFFIX}
                    PackInfoFluid.h PackInfoFluid.${CODEGEN_FILE_SUFFIX}
                    PackInfoEnergy.h PackInfoEnergy.${CODEGEN_FILE_SUFFIX}
                    InitializeFluidDomain.h InitializeFluidDomain.${CODEGEN_FILE_SUFFIX}
                    InitializeEnergyDomain.h InitializeEnergyDomain.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_NoSlip.h BC_Fluid_NoSlip.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_UBB.h BC_Fluid_UBB.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_Density.h BC_Fluid_Density.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_FreeSlip.h BC_Fluid_FreeSlip.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_Outflow.h BC_Fluid_Outflow.${CODEGEN_FILE_SUFFIX}
                    BC_energy_DiffusionDirichlet_dynamic.h BC_energy_DiffusionDirichlet_dynamic.${CODEGEN_FILE_SUFFIX}
                    BC_Energy_Neumann.h BC_Energy_Neumann.${CODEGEN_FILE_SUFFIX}
                    BC_energy_DiffusionDirichlet_static.h BC_energy_DiffusionDirichlet_static.${CODEGEN_FILE_SUFFIX}
                    FluidMacroSetter.h FluidMacroSetter.cpp
                    FluidMacroGetter.h FluidMacroGetter.cpp
                    EnergyMacroSetter.h EnergyMacroSetter.cpp
                    EnergyMacroGetter.h EnergyMacroGetter.cpp
                    compute_temperature_field.h compute_temperature_field.cpp
                    compute_temperature_field_particle.h compute_temperature_field_particle.cpp
                    initializeConcentrationField.h initializeConcentrationField.cpp
                    GeneralInfoHeader.h
                    compute_velocity_field.h compute_velocity_field.cpp
            )
            message(STATUS "generated in thermal PSM is: ${config}" )
            add_executable(smooth_profile_${config}  thermalPSMCHT.cpp ../../utilities/InitializerFunctions.cpp)
            target_link_libraries(smooth_profile_${config} walberla::blockforest walberla::core walberla::field walberla::geometry walberla::gpu walberla::lbm walberla::lbm_mesapd_coupling walberla::mesa_pd walberla::sqlite walberla::vtk smooth_profile_lbm_${config})
            target_compile_definitions(smooth_profile_${config} PRIVATE Weighting=1)
        endforeach ()


    endif ()
endif ()


