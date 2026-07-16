//!
//! SPDX-License-Identifier: CC-0
//! SPDX-FileCreatedBy: Philipp Suffa  philipp.suffa@fau.de
//! SPDX-FileCreatedUsing: [Claude Opus 4.8]
//!
//! \file AdaptiveRefinement.cpp
//!
//! \brief Tests adaptive (dynamic) mesh refinement for the lbm_generated module.
//!
//! A fully periodic box is initialised with a spatially uniform flow field, which is an exact fixed
//! point of the lattice Boltzmann method. The block forest is then adaptively refined and coarsened
//! during the simulation (blockforest->refresh()). Because a uniform equilibrium flow is preserved
//! exactly by the coarse-to-fine (explosion) and fine-to-coarse (coalescence) operations as well as
//! by the non-uniform communication, the velocity has to remain uniform throughout. This exercises
//! the whole adaptive-refinement pipeline: rebuilding of the recursive time step's block lists,
//! migration of the PDF field data across level changes and recomputation of the non-uniform
//! communication data.
//
//======================================================================================================================

#include "blockforest/SetupBlockForest.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/communication/NonUniformBufferedScheme.h"
#include "blockforest/loadbalancing/DynamicCurve.h"
#include "blockforest/loadbalancing/NoPhantomData.h"
#include "blockforest/loadbalancing/StaticCurve.h"
#include "blockforest/Initialization.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/debug/TestSubsystem.h"
#include "core/math/Vector3.h"
#include "core/mpi/Reduce.h"

#include "field/all.h"


#include "geometry/InitBoundaryHandling.h"

#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"

// include the generated header file. It includes all generated classes
#include "AdaptiveRefinementInfoHeader.h"

using namespace walberla;

using StorageSpecification_T = lbm::AdaptiveRefinementStorageSpecification;
using Stencil_T              = StorageSpecification_T::Stencil;
using CommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T             = lbm_generated::PdfField< StorageSpecification_T >;

using SweepCollection_T = lbm::AdaptiveRefinementSweepCollection;

using VelocityField_T = GhostLayerField< real_t, StorageSpecification_T::Stencil::D >;

using flag_t               = walberla::uint8_t;
using FlagField_T          = FlagField< flag_t >;
using BoundaryCollection_T = lbm::AdaptiveRefinementBoundaryCollection< FlagField_T >;

const FlagUID fluidFlagUID("Fluid");

//**********************************************************************************************************************
/*! Refinement criterion: every block whose AABB intersects the given region is refined up to
 *  targetLevel, all other blocks are coarsened down to the coarsest level.
 */
//**********************************************************************************************************************
class RefineInRegion
{
 public:
   RefineInRegion(const AABB& region, const uint_t targetLevel) : region_(region), targetLevel_(targetLevel) {}

   void operator()(std::vector< std::pair< const Block*, uint_t > >& minTargetLevels,
                   std::vector< const Block* >&, const BlockForest&) const
   {
      for (auto& [block, targetLevel] : minTargetLevels)
      {
         targetLevel = (region_.intersects(block->getAABB())) ? targetLevel_ : uint_t(0);
      }
   }

 private:
   AABB   region_;
   uint_t targetLevel_;
};

int main(int argc, char** argv)
{
   walberla::Environment walberlaEnv(argc, argv);
   mpi::MPIManager::instance()->useWorldComm();

   debug::enterTestMode();

   // ------------------------------------------------------------------------------------------------------------------
   //   Parameters
   // ------------------------------------------------------------------------------------------------------------------
   const Vector3< uint_t > rootBlocks(uint_c(4), uint_c(2), uint_c(2));
   const Vector3< uint_t > cellsPerBlock(uint_c(16), uint_c(16), uint_c(16));
   const AABB              domain(real_c(0), real_c(0), real_c(0), real_c(4), real_c(2), real_c(2));

   const real_t omega = real_c(1.6);
   const real_t u_x   = real_c(0.05);
   const uint_t refinementDepth = uint_c(1);

   // Region that is refined in the second phase of the test (one column of root blocks in x).
   const AABB refinementRegion(real_c(1.1), real_c(0.1), real_c(0.1), real_c(2.1), real_c(1.9), real_c(1.9));
   // Region that is never intersected -> used to coarsen the whole domain again.
   const AABB emptyRegion(real_c(-2), real_c(-2), real_c(-2), real_c(-1), real_c(-1), real_c(-1));

   const real_t velEps = real_c(1e-5);

   // ------------------------------------------------------------------------------------------------------------------
   //   Block forest (initially uniform, fully periodic)
   // ------------------------------------------------------------------------------------------------------------------
   SetupBlockForest setupBfs;
   setupBfs.addWorkloadMemorySUIDAssignmentFunction(blockforest::uniformWorkloadAndMemoryAssignment);
   setupBfs.init(domain, rootBlocks[0], rootBlocks[1], rootBlocks[2], true, true, true);
   setupBfs.balanceLoad(blockforest::StaticLevelwiseCurveBalance(true), uint_c(MPIManager::instance()->numProcesses()));

   auto forest = std::make_shared< BlockForest >(uint_c(MPIManager::instance()->worldRank()), setupBfs);
   auto blocks = std::make_shared< StructuredBlockForest >(forest, cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2]);
   blocks->createCellBoundingBoxes();

   // Enable dynamic (adaptive) refinement.
   forest->recalculateBlockLevelsInRefresh(true);
   forest->alwaysRebalanceInRefresh(true);
   forest->allowRefreshChangingDepth(true);
   forest->allowMultipleRefreshCycles(false);
   forest->checkForEarlyOutInRefresh(false);
   forest->checkForLateOutInRefresh(false);
   forest->setRefreshPhantomBlockMigrationPreparationFunction(
      blockforest::DynamicCurveBalance< blockforest::NoPhantomData >(true, true, false));

   // ------------------------------------------------------------------------------------------------------------------
   //   Fields
   // ------------------------------------------------------------------------------------------------------------------
   const StorageSpecification_T storageSpec = StorageSpecification_T();
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", storageSpec, uint_c(2));
   const BlockDataID velFieldId  = field::addToStorage< VelocityField_T >(blocks, "velocity", real_c(0), field::fzyx, uint_c(2));
   const BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flags", uint_c(2));

   SweepCollection_T sweepCollection(blocks, pdfFieldId, velFieldId, omega);

   // ------------------------------------------------------------------------------------------------------------------
   //   Initialisation: spatially uniform flow field
   // ------------------------------------------------------------------------------------------------------------------
   auto initialiseUniformFlow = [&]() {
      for (auto& block : *blocks)
      {
         auto* velField = block.getData< VelocityField_T >(velFieldId);
         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(velField,
            velField->get(x, y, z, 0) = u_x;
            velField->get(x, y, z, 1) = real_c(0);
            velField->get(x, y, z, 2) = real_c(0);
         )
         sweepCollection.initialise(&block, cell_idx_c(1));
      }
   };
   initialiseUniformFlow();

   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, fluidFlagUID, 2);
   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, fluidFlagUID);

   // ------------------------------------------------------------------------------------------------------------------
   //   Communication and recursive time step
   // ------------------------------------------------------------------------------------------------------------------
   auto comm = std::make_shared< blockforest::communication::NonUniformBufferedScheme< CommunicationStencil_T > >(blocks);
   auto packInfo = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldId);
   comm->addPackInfo(packInfo);

   lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > timestep(
      blocks, pdfFieldId, sweepCollection, boundaryCollection, comm, packInfo);

   // ------------------------------------------------------------------------------------------------------------------
   //   Helper lambdas
   // ------------------------------------------------------------------------------------------------------------------


   auto checkUniformVelocity = [&](const std::string& phase) {
      real_t maxDev = real_c(0);
      for (auto& block : *blocks)
      {
         sweepCollection.calculateMacroscopicParameters(&block);
         auto* velField = block.getData< VelocityField_T >(velFieldId);
         WALBERLA_FOR_ALL_CELLS_XYZ(velField,
            maxDev = std::max(maxDev, std::abs(velField->get(x, y, z, 0) - u_x));
            maxDev = std::max(maxDev, std::abs(velField->get(x, y, z, 1)));
            maxDev = std::max(maxDev, std::abs(velField->get(x, y, z, 2)));
         )
      }
      mpi::allReduceInplace(maxDev, mpi::MAX);
      WALBERLA_LOG_INFO_ON_ROOT("Phase '" << phase << "': max velocity deviation from uniform flow = " << maxDev)
      WALBERLA_CHECK_LESS(maxDev, velEps, "Velocity deviates from uniform flow during phase '" << phase << "'")
   };

   auto numberOfBlocks = [&]() {
      uint_t n = forest->getNumberOfBlocks();
      mpi::allReduceInplace(n, mpi::SUM);
      return n;
   };

   auto adaptiveRefresh = [&](const AABB& region, const uint_t targetLevel) {
      forest->setRefreshMinTargetLevelDeterminationFunction(RefineInRegion(region, targetLevel));
      forest->refresh();
      for (auto& block : *blocks)
         packInfo->recalculateNonuniformCommData(&block);
   };

   SweepTimeloop timeloop(blocks->getBlockStorage(), 0);
   timeloop.addFuncBeforeTimeStep(timestep, "Recursive LBM time step");

   uint_t vtkCounter = 0;
   auto runTimesteps = [&](const uint_t n) {
      for (uint_t t = 0; t < n; ++t) {
         timeloop.singleStep();
         vtk::writeDomainDecomposition(blocks, "domain_decomposition_" + std::to_string(vtkCounter), "vtk_out");
         ++vtkCounter;
      }

   };

   // ------------------------------------------------------------------------------------------------------------------
   //   Phase 1: uniform grid
   // ------------------------------------------------------------------------------------------------------------------
   const uint_t initialBlocks = numberOfBlocks();
   WALBERLA_CHECK_EQUAL(blocks->getDepth(), uint_c(0), "Domain should start without refinement")
   runTimesteps(uint_c(5));
   checkUniformVelocity("uniform grid");

   // ------------------------------------------------------------------------------------------------------------------
   //   Phase 2: adaptively refine a sub-region
   // ------------------------------------------------------------------------------------------------------------------
   adaptiveRefresh(refinementRegion, refinementDepth);
   const uint_t refinedBlocks = numberOfBlocks();
   WALBERLA_CHECK_EQUAL(blocks->getDepth(), refinementDepth, "Refinement region should have been refined")
   WALBERLA_CHECK_GREATER(refinedBlocks, initialBlocks, "Refinement should have created additional blocks")
   WALBERLA_LOG_INFO_ON_ROOT("After refinement: " << refinedBlocks << " blocks, depth " << blocks->getDepth())
   checkUniformVelocity("right after refinement (data migration only)");
   runTimesteps(uint_c(10));
   checkUniformVelocity("after refinement");

   // ------------------------------------------------------------------------------------------------------------------
   //   Phase 3: coarsen everything back to the uniform grid
   // ------------------------------------------------------------------------------------------------------------------
   adaptiveRefresh(emptyRegion, uint_c(0));
   const uint_t coarsenedBlocks = numberOfBlocks();
   WALBERLA_CHECK_EQUAL(blocks->getDepth(), uint_c(0), "Domain should have been coarsened back to a uniform grid")
   WALBERLA_CHECK_EQUAL(coarsenedBlocks, initialBlocks, "Coarsening should restore the original number of blocks")
   WALBERLA_LOG_INFO_ON_ROOT("After coarsening: " << coarsenedBlocks << " blocks, depth " << blocks->getDepth())
   runTimesteps(uint_c(5));
   checkUniformVelocity("after coarsening");

   WALBERLA_LOG_INFO_ON_ROOT("Adaptive refinement test passed.")
   return EXIT_SUCCESS;
}
