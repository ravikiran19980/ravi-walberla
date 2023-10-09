target_sources( lbm_mesapd_coupling
    PRIVATE
    PSMSweep.h
    ParticleAndVolumeFractionMapping.h
    PSMUtility.h
    )

if( WALBERLA_BUILD_WITH_CUDA )
    add_subdirectory(cuda)
endif()
