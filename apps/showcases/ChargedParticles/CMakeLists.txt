waLBerla_link_files_to_builddir( "*.prm" )

waLBerla_add_executable ( NAME ChargedParticles FILES ChargedParticles.cpp
        DEPENDS blockforest boundary core pde domain_decomposition field lbm lbm_mesapd_coupling mesa_pd timeloop vtk )

waLBerla_add_executable ( NAME TestIterativeSolvers FILES TestIterativeSolvers.cpp
        DEPENDS blockforest boundary core pde domain_decomposition field timeloop vtk )

waLBerla_add_executable ( NAME PotentialValidation FILES PotentialValidation.cpp
        DEPENDS blockforest boundary geometry core pde domain_decomposition field timeloop vtk )
