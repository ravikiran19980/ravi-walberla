if (WALBERLA_BUILD_WITH_CUDA)
    target_sources(lbm_mesapd_coupling
            PRIVATE
            ParticleAndVolumeFractionMappingGPU.h
            PSMSweepCollectionGPU.h
            )
    if (CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 60)
        target_sources(lbm_mesapd_coupling
                PRIVATE
                PSMWrapperKernels.cu
                PSMWrapperKernels.h
                PSMUtilityGPU.cuh
                PSMWrapperSweepsGPU.h
                )
        if (WALBERLA_BUILD_WITH_CODEGEN)
            foreach (collision_setup srt trt)
                foreach (solid_collision 1 2 3)
                    foreach (split 0 1)
                        set(config ${collision_setup}_sc${solid_collision}_s${split})
                        walberla_generate_target_from_python(NAME PSMCodegenPython_${config}
                                FILE PSMCodegen.py
                                CODEGEN_CFG ${config}
                                OUT_FILES PSMSweep.cu PSMSweep.h PSMSweepSplit.cu PSMSweepSplit.h
                                PSMPackInfo.cu PSMPackInfo.h InitializeDomainForPSM.cu InitializeDomainForPSM.h
                                PSM_NoSlip.cu PSM_NoSlip.h PSM_UBB.cu PSM_UBB.h PSM_Density.cu PSM_Density.h
                                PSM_FreeSlip.cu PSM_FreeSlip.h PSM_InfoHeader.h PSM_MacroGetter.cpp PSM_MacroGetter.h
                                PSM_MacroSetter.cpp PSM_MacroSetter.h)
                    endforeach ()
                endforeach ()
            endforeach ()
        endif ()
    endif ()
endif ()
