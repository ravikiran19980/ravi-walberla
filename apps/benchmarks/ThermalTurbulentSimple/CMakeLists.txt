waLBerla_link_files_to_builddir("*.prm")

if (WALBERLA_BUILD_WITH_CODEGEN)
    if (NOT WALBERLA_BUILD_WITH_GPU_SUPPORT OR (WALBERLA_BUILD_WITH_GPU_SUPPORT AND (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60 OR WALBERLA_BUILD_WITH_HIP)))
        waLBerla_generate_target_from_python(NAME simplifiedChannel_CodeGeneration_${config}
                FILE thermalPSMCodegenCHT.py
                CODEGEN_CFG ${config}
                OUT_FILES TurbulentChannel_Setter.h TurbulentChannel_Setter.${CODEGEN_FILE_SUFFIX}
                TurbulentChannel_Sweep.h TurbulentChannel_Sweep.${CODEGEN_FILE_SUFFIX}
                PackInfoFluid.h PackInfoFluid.${CODEGEN_FILE_SUFFIX}
                InitializeFluidDomain.h InitializeFluidDomain.${CODEGEN_FILE_SUFFIX}
                BC_Fluid_NoSlip.h BC_Fluid_NoSlip.${CODEGEN_FILE_SUFFIX}
                BC_Fluid_Qbb.h BC_Fluid_Qbb.${CODEGEN_FILE_SUFFIX}
                TurbulentChannel_WFB_bottom.h TurbulentChannel_WFB_bottom.${CODEGEN_FILE_SUFFIX}
                TurbulentChannel_WFB_top.h TurbulentChannel_WFB_top.${CODEGEN_FILE_SUFFIX}
                WelfordVelocity.h WelfordVelocity.${CODEGEN_FILE_SUFFIX}
                FluidMacroSetter.h FluidMacroSetter.cpp
                FluidMacroGetter.h FluidMacroGetter.cpp
                GeneralInfoHeader.h
        )

        set(MAINSRC simplifiedBenchmark.cpp)
        set(BASE_SOURCES
                ${MAINSRC}
        )

        add_executable(simplifiedChannel_${config}
                ${SRC}
                ${BASE_SOURCES}
        )
        target_link_libraries(simplifiedChannel_${config} walberla::blockforest walberla::core walberla::field walberla::geometry walberla::gpu walberla::lbm walberla::lbm_mesapd_coupling walberla::mesa_pd walberla::sqlite walberla::vtk simplifiedChannel_CodeGeneration_${config})
        target_compile_definitions(simplifiedChannel_${config} PRIVATE Weighting=1)
    endif ()
endif ()



