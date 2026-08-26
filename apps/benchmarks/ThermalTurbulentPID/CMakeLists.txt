#waLBerla_link_files_to_builddir("*.prm")

if (WALBERLA_BUILD_WITH_CODEGEN)
    if (NOT WALBERLA_BUILD_WITH_GPU_SUPPORT OR (WALBERLA_BUILD_WITH_GPU_SUPPORT AND (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 OR WALBERLA_BUILD_WITH_HIP)))
        foreach(solid_collision 1 2)
            if (run_with_temperature)
                add_definitions(-Drun_with_temperature)
            endif()
            set(config sc_${solid_collision})
            message(STATUS "Config value in thermal PSM is: ${config}" )
            waLBerla_generate_target_from_python(NAME ThermalTurbulentChannel_PID_codegen_${config}
                    FILE thermalPSMCodegenCHT.py
                    CODEGEN_CFG ${config}
                    OUT_FILES PSMFluidSweep.h PSMFluidSweep.${CODEGEN_FILE_SUFFIX}
                    PSMTemperatureSweep.h PSMTemperatureSweep.${CODEGEN_FILE_SUFFIX}
                    PackInfoFluid.h PackInfoFluid.${CODEGEN_FILE_SUFFIX}
                    PackInfoTemperature.h PackInfoTemperature.${CODEGEN_FILE_SUFFIX}
                    InitializeFluidDomain.h InitializeFluidDomain.${CODEGEN_FILE_SUFFIX}
                    InitializeTemperatureDomain.h InitializeTemperatureDomain.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_NoSlip.h BC_Fluid_NoSlip.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_UBB.h BC_Fluid_UBB.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_Density.h BC_Fluid_Density.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_FreeSlip.h BC_Fluid_FreeSlip.${CODEGEN_FILE_SUFFIX}
                    BC_Fluid_Outflow.h BC_Fluid_Outflow.${CODEGEN_FILE_SUFFIX}
                    BC_Temperature_DiffusionDirichlet_dynamic.h BC_Temperature_DiffusionDirichlet_dynamic.${CODEGEN_FILE_SUFFIX}
                    BC_Temperature_Neumann.h BC_Temperature_Neumann.${CODEGEN_FILE_SUFFIX}
                    BC_Temperature_DiffusionDirichlet_static.h BC_Temperature_DiffusionDirichlet_static.${CODEGEN_FILE_SUFFIX}
                    WelfordVelocity.h WelfordVelocity.${CODEGEN_FILE_SUFFIX}
                    WelfordTemperature.h WelfordTemperature.${CODEGEN_FILE_SUFFIX}
                    FluidMacroSetter.h FluidMacroSetter.${CODEGEN_FILE_SUFFIX}
                    FluidMacroGetter.h FluidMacroGetter.${CODEGEN_FILE_SUFFIX}
                    TemperatureMacroSetter.h TemperatureMacroSetter.${CODEGEN_FILE_SUFFIX}
                    TemperatureMacroGetter.h TemperatureMacroGetter.${CODEGEN_FILE_SUFFIX}
                    initializeVelocityFieldParticles.h initializeVelocityFieldParticles.${CODEGEN_FILE_SUFFIX}
                    initializeTemperatureField.h initializeTemperatureField.cpp
                    GeneralInfoHeader.h
            )
            message(STATUS "generated in thermal PSM is: ${config}" )
            if (WALBERLA_BUILD_WITH_HIP OR WALBERLA_BUILD_WITH_GPU_SUPPORT)
                set(MAINSRC ThermalTurbulentChannelGPU.cpp)

            else()
                set(MAINSRC ThermalTurbulentChannel.cpp)
            endif()

            set(BASE_SOURCES
                    ${MAINSRC}
                    ./utilities/InitializerFunctions.h
                    ./utilities/InitializerFunctions.cpp
                    PIDController.h
                    PIDController.cpp
                    CheckpointStatistics.h
            )
            if (WALBERLA_BUILD_WITH_HIP OR WALBERLA_BUILD_WITH_GPU_SUPPORT)
                list(APPEND BASE_SOURCES
                        ./utilities/bulkVelocityReductionKernel.${CODEGEN_FILE_SUFFIX}
                        ./utilities/temperatureWrapperKernels.h
                        ./utilities/temperatureWrapperKernels.${CODEGEN_FILE_SUFFIX}
                )
            endif()
            add_executable(ThermalTurbulentChannel_PID_${config}
                    ${SRC}
                    ${BASE_SOURCES}
            )


            target_link_libraries(ThermalTurbulentChannel_PID_${config} walberla::blockforest walberla::core walberla::field walberla::geometry walberla::gpu walberla::lbm walberla::lbm_mesapd_coupling walberla::mesa_pd walberla::sqlite walberla::vtk ThermalTurbulentChannel_PID_codegen_${config})
            target_compile_definitions(ThermalTurbulentChannel_PID_${config} PRIVATE Weighting=1)
        endforeach ()


    endif ()
endif ()