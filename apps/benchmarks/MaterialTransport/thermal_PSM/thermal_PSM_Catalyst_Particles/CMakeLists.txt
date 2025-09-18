waLBerla_link_files_to_builddir("*.prm")

if (WALBERLA_BUILD_WITH_CODEGEN)
    if (NOT WALBERLA_BUILD_WITH_GPU_SUPPORT OR (WALBERLA_BUILD_WITH_GPU_SUPPORT AND (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 OR WALBERLA_BUILD_WITH_HIP)))
        foreach(solid_collision 1 2 3)

            set(config sc_${solid_collision})
            message(STATUS "Config value in thermal PSM is: ${config}" )

            message(STATUS "generated in thermal PSM is: ${config}" )
            waLBerla_add_executable(NAME thermalPSMCatalyst_${config} FILES thermalPSMCHT.cpp ../../utilities/InitializerFunctions.cpp
                    DEPENDS walberla::blockforest walberla::core walberla::field walberla::geometry walberla::gpu walberla::lbm walberla::lbm_mesapd_coupling walberla::mesa_pd walberla::sqlite walberla::vtk CodegenThermalPSMCatalyst_${config})
            walberla_generate_sources(
                    thermalPSMCatalyst_${config}
                    SCRIPTS thermalPSMCodegenCHT.py
                    SCRIPT_ARGS ${config}
            )
            target_compile_definitions(thermalPSMCatalyst_${config} PRIVATE Weighting=2)
        endforeach ()


    endif ()
endif ()



