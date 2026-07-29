waLBerla_link_files_to_builddir("*.prm")

if (WALBERLA_BUILD_WITH_CODEGEN)
    if (NOT WALBERLA_BUILD_WITH_GPU_SUPPORT OR (WALBERLA_BUILD_WITH_GPU_SUPPORT AND (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 OR WALBERLA_BUILD_WITH_HIP)))
        foreach(solid_collision 1 2 3)

            set(config sc_${solid_collision})
            message(STATUS "Config value in thermal PSM is: ${config}" )

            message(STATUS "generated in thermal PSM is: ${config}" )
            add_executable(surrogates_cht_virtual${config} surrogates.cpp ../../utilities/InitializerFunctions.cpp)
            target_link_libraries(surrogates_cht_virtual${config} walberla::blockforest walberla::core walberla::field walberla::geometry walberla::gpu walberla::lbm walberla::lbm_mesapd_coupling walberla::mesa_pd walberla::sqlite walberla::vtk codegenThermal_${config})
            target_compile_definitions(surrogates_cht_virtual${config} PRIVATE Weighting=1)
        endforeach ()


    endif ()
endif ()
if (WALBERLA_BUILD_WITH_GPU_SUPPORT AND (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 OR WALBERLA_BUILD_WITH_HIP))
    target_sources( walberla_lbm_mesapd_coupling
            PRIVATE
            ../../utilities/temperatureWrapperKernels.${CODEGEN_FILE_SUFFIX}
    )
endif ()

