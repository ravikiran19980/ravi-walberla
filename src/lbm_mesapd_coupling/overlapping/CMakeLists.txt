target_sources(lbm_mesapd_coupling
        PRIVATE
        OverlapFraction.h
        )

if( WALBERLA_BUILD_WITH_CUDA )
    add_subdirectory(cuda)
endif()
