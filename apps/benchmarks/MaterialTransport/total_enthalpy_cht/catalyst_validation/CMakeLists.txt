waLBerla_link_files_to_builddir("*.prm")

if (WALBERLA_BUILD_WITH_CODEGEN)
    if (WALBERLA_BUILD_WITH_GPU_SUPPORT)
        message(FATAL_ERROR
                "The catalyst validation test is only supported for CPU builds. "
                "Please configure with WALBERLA_BUILD_WITH_GPU_SUPPORT=OFF."
        )
    endif ()
    if (NOT WALBERLA_BUILD_WITH_GPU_SUPPORT)
        foreach (solid_collision 1 2 3)

            set(config sc_${solid_collision})
            add_executable(catalyst_virtualfluid_cht_${config} catalyst.cpp ../../utilities/InitializerFunctions.cpp)
            target_link_libraries(catalyst_virtualfluid_cht_${config} walberla::blockforest walberla::core walberla::field walberla::geometry walberla::gpu walberla::lbm walberla::lbm_mesapd_coupling walberla::mesa_pd walberla::sqlite walberla::vtk codegenThermal_${config})
            target_compile_definitions(catalyst_virtualfluid_cht_${config} PRIVATE Weighting=1)
        endforeach ()


    endif ()
endif ()


