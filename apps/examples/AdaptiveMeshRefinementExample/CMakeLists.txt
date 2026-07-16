walberla_link_files_to_builddir (*.prm)

waLBerla_generate_target_from_python (
    NAME AdaptiveMeshRefinementExampleCodeGen
    FILE AdaptiveMeshRefinementExample.py
    OUT_FILES NoSlipBC.h
              NoSlipBC.cpp
              UBBBC.h
              UBBBC.cpp
              OutflowBC.h
              OutflowBC.cpp
              FreeSlipBC.h
              FreeSlipBC.cpp
              AdaptiveMeshRefinementExampleStorageSpecification.h
              AdaptiveMeshRefinementExampleStorageSpecification.cpp
              AdaptiveMeshRefinementExampleSweepCollection.h
              AdaptiveMeshRefinementExampleSweepCollection.cpp
              AdaptiveMeshRefinementExampleBoundaryCollection.h
              InfoHeader.h
)

add_executable (AdaptiveMeshRefinementExample AdaptiveMeshRefinementExample.cpp)
target_link_libraries (
    AdaptiveMeshRefinementExample
    walberla::blockforest
    walberla::boundary
    walberla::core
    walberla::domain_decomposition
    walberla::field
    walberla::geometry
    walberla::lbm_generated
    walberla::timeloop
    walberla::vtk
    AdaptiveMeshRefinementExampleCodeGen
)
