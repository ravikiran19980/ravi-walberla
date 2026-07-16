//!
//! SPDX-License-Identifier: CC-0
//! SPDX-FileCreatedBy:Philipp Suffa   philipp.suffa@fau.de
//! SPDX-FileCreatedUsing: [Claude Opus 4.8]
//!
//! \file AdaptiveMeshRefinementExample.cpp
//!
//! \brief Flow around a sphere with ADAPTIVE (dynamic) mesh refinement for the lbm_generated module.
//!
//! In contrast to the (static) MeshRefinementExample, the block forest is re-partitioned during the
//! simulation: blocks are refined where the velocity field shows strong gradients (the sphere and
//! its developing wake) and coarsened again in calm regions. The refinement is performed via
//! blockforest->refresh(), which migrates all block data (the PDF field, the macroscopic fields and
//! the flag field) between the refinement levels and re-balances the load across processes.
//
//======================================================================================================================

#include "blockforest/SetupBlockForest.h"
#include "blockforest/StructuredBlockForest.h"
#include "blockforest/communication/NonUniformBufferedScheme.h"
#include "blockforest/loadbalancing/DynamicCurve.h"
#include "blockforest/loadbalancing/NoPhantomData.h"
#include "blockforest/loadbalancing/StaticCurve.h"
#include "blockforest/Initialization.h"

#include "core/all.h"

#include "field/all.h"
#include "geometry/all.h"

#include "lbm_generated/communication/NonuniformGeneratedPdfPackInfo.h"
#include "lbm_generated/evaluation/PerformanceEvaluation.h"
#include "lbm_generated/field/AddToStorage.h"
#include "lbm_generated/field/PdfField.h"
#include "lbm_generated/refinement/BasicRecursiveTimeStep.h"

#include "timeloop/all.h"

#include "InfoHeader.h"

namespace walberla
{

constexpr uint_t FieldGhostLayer{ 2 };

using StorageSpecification_T    = lbm::AdaptiveMeshRefinementExampleStorageSpecification;
using LBMCommunicationStencil_T = StorageSpecification_T::CommunicationStencil;
using PdfField_T                = lbm_generated::PdfField< StorageSpecification_T >;

using SweepCollection_T = lbm::AdaptiveMeshRefinementExampleSweepCollection;

using flag_t      = walberla::uint32_t;
using FlagField_T = FlagField< flag_t >;

using BoundaryCollection_T = lbm::AdaptiveMeshRefinementExampleBoundaryCollection< FlagField_T >;

const FlagUID FluidFlagUID("Fluid");
const FlagUID NoSlipFlagUID("NoSlip");
const FlagUID UBBFlagUID("UBB");
const FlagUID OutflowFlagUID("Outflow");
const FlagUID FreeSlipFlagUID("FreeSlip");

//**********************************************************************************************************************
/*! Adaptive refinement criterion.
 *
 *  A block is refined up to maxLevel if it overlaps the sphere (which must always be well resolved)
 *  or if the magnitude of the velocity gradient inside the block exceeds upperLimit. A block is
 *  coarsened if its velocity gradient is everywhere below lowerLimit. Otherwise the block keeps its
 *  current level.
 */
//**********************************************************************************************************************
class VelocityGradientRefinement
{
 public:
   VelocityGradientRefinement(const ConstBlockDataID& velFieldId, const geometry::Sphere& sphere,
                              const real_t lowerLimit, const real_t upperLimit, const uint_t maxLevel)
      : velFieldId_(velFieldId), sphere_(sphere), lowerLimit_(lowerLimit * lowerLimit),
        upperLimit_(upperLimit * upperLimit), maxLevel_(maxLevel)
   {}

   void operator()(std::vector< std::pair< const Block*, uint_t > >& minTargetLevels,
                   std::vector< const Block* >&, const BlockForest&) const
   {
      for (auto& [block, targetLevel] : minTargetLevels)
      {
         const uint_t currentLevel = block->getLevel();

         // Always keep the sphere and its immediate surroundings on the finest level.
         AABB sphereBox = sphere_.boundingBox();
         sphereBox.scale(real_c(1.5));
         if (sphereBox.intersects(block->getAABB()))
         {
            targetLevel = maxLevel_;
            continue;
         }

         const auto* u = block->getData< VelocityField_T >(velFieldId_);
         real_t maxGradientSq = real_c(0);

         const cell_idx_t xSize = cell_idx_c(u->xSize());
         const cell_idx_t ySize = cell_idx_c(u->ySize());
         const cell_idx_t zSize = cell_idx_c(u->zSize());
         for (cell_idx_t z = cell_idx_t(1); z < zSize - cell_idx_t(1); ++z)
            for (cell_idx_t y = cell_idx_t(1); y < ySize - cell_idx_t(1); ++y)
               for (cell_idx_t x = cell_idx_t(1); x < xSize - cell_idx_t(1); ++x)
               {
                  for (uint_t d = 0; d < uint_c(3); ++d)
                  {
                     const real_t dx = u->get(x + cell_idx_t(1), y, z, d) - u->get(x - cell_idx_t(1), y, z, d);
                     const real_t dy = u->get(x, y + cell_idx_t(1), z, d) - u->get(x, y - cell_idx_t(1), z, d);
                     const real_t dz = u->get(x, y, z + cell_idx_t(1), d) - u->get(x, y, z - cell_idx_t(1), d);
                     maxGradientSq = std::max(maxGradientSq, real_c(0.25) * (dx * dx + dy * dy + dz * dz));
                  }
               }

         if (maxGradientSq > upperLimit_ && currentLevel < maxLevel_)
            targetLevel = currentLevel + uint_t(1);
         else if (maxGradientSq < lowerLimit_ && currentLevel > uint_t(0))
            targetLevel = currentLevel - uint_t(1);
         // else: keep current level (targetLevel is already initialised to it)
      }
   }

 private:
   ConstBlockDataID velFieldId_;
   geometry::Sphere sphere_;
   real_t           lowerLimit_;
   real_t           upperLimit_;
   uint_t           maxLevel_;
};

int main(int argc, char** argv)
{
   Environment env(argc, argv);
   if (!env.config()) { WALBERLA_ABORT("No configuration file specified!") }
   mpi::MPIManager::instance()->useWorldComm();

   // ------------------------------------------------------------------------------------------------------------------
   //   Parameters
   // ------------------------------------------------------------------------------------------------------------------
   auto parameters = env.config()->getOneBlock("Parameters");
   const uint_t timesteps             = parameters.getParameter< uint_t >("timesteps");
   const Vector3< uint_t > rootBlocks = parameters.getParameter< Vector3< uint_t > >("rootBlocks");
   const Vector3< uint_t > cellsPerBlock = parameters.getParameter< Vector3< uint_t > >("cellsPerBlock");
   const Vector3< real_t > sphereCenter  = parameters.getParameter< Vector3< real_t > >("sphereCenter");
   const real_t sphereRadius             = parameters.getParameter< real_t >("sphereRadius");
   const uint_t refinementLevels         = parameters.getParameter< uint_t >("refinementLevels");
   const uint_t vtkWriteFrequency        = parameters.getParameter< uint_t >("vtkWriteFrequency");
   const real_t omega                    = parameters.getParameter< real_t >("omega");

   const uint_t refinementFrequency = parameters.getParameter< uint_t >("refinementFrequency", uint_c(100));
   const real_t lowerRefinementLimit = parameters.getParameter< real_t >("lowerRefinementLimit", real_c(1e-5));
   const real_t upperRefinementLimit = parameters.getParameter< real_t >("upperRefinementLimit", real_c(1e-4));

   const Vector3< real_t > initialVel(real_c(0.1), real_c(0), real_c(0));

   const auto domainAABB = AABB(real_c(0), real_c(0), real_c(0), real_c(1), real_c(1), real_c(1));
   geometry::Sphere sphere(sphereCenter, sphereRadius);

   // ------------------------------------------------------------------------------------------------------------------
   //   Block forest (initially uniform; inlet/outlet in x, periodic in y and z)
   // ------------------------------------------------------------------------------------------------------------------
   const uint_t numProcs = uint_c(mpi::MPIManager::instance()->numProcesses());
   SetupBlockForest setupBfs;
   setupBfs.addWorkloadMemorySUIDAssignmentFunction(blockforest::uniformWorkloadAndMemoryAssignment);
   setupBfs.init(domainAABB, rootBlocks[0], rootBlocks[1], rootBlocks[2], false, false, false);
   setupBfs.balanceLoad(blockforest::StaticLevelwiseCurveBalanceWeighted(), numProcs);

   auto forest = std::make_shared< BlockForest >(uint_c(MPIManager::instance()->worldRank()), setupBfs);
   auto blocks = std::make_shared< StructuredBlockForest >(forest, cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2]);
   blocks->createCellBoundingBoxes();

   // Enable dynamic (adaptive) mesh refinement.
   forest->recalculateBlockLevelsInRefresh(true);
   forest->alwaysRebalanceInRefresh(true);
   forest->allowRefreshChangingDepth(true);
   forest->allowMultipleRefreshCycles(true);
   forest->reevaluateMinTargetLevelsAfterForcedRefinement(true);
   forest->checkForEarlyOutInRefresh(true);
   forest->checkForLateOutInRefresh(true);
   forest->setRefreshPhantomBlockMigrationPreparationFunction(
      blockforest::DynamicCurveBalance< blockforest::NoPhantomData >(true, true, false));

   // ------------------------------------------------------------------------------------------------------------------
   //   Fields
   // ------------------------------------------------------------------------------------------------------------------
   const StorageSpecification_T StorageSpec = StorageSpecification_T();
   const BlockDataID pdfFieldId  = lbm_generated::addPdfFieldToStorage(blocks, "pdfs", StorageSpec, FieldGhostLayer, field::fzyx);
   const BlockDataID velFieldId  = field::addToStorage< VelocityField_T >(blocks, "velocity", real_c(0), field::fzyx, FieldGhostLayer);
   const BlockDataID densityId   = field::addToStorage< ScalarField_T >(blocks, "density", real_c(1), field::fzyx, FieldGhostLayer);
   const BlockDataID flagFieldId = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field", FieldGhostLayer);

   SweepCollection_T sweepCollection(blocks, densityId, pdfFieldId, velFieldId, omega);

   // ------------------------------------------------------------------------------------------------------------------
   //   Communication and recursive time step
   // ------------------------------------------------------------------------------------------------------------------
   auto communication = std::make_shared< blockforest::communication::NonUniformBufferedScheme< LBMCommunicationStencil_T > >(blocks);
   auto packInfo      = lbm_generated::setupNonuniformPdfCommunication< PdfField_T >(blocks, pdfFieldId);
   communication->addPackInfo(packInfo);

   // ------------------------------------------------------------------------------------------------------------------
   //   (Re-)initialisation helpers
   // ------------------------------------------------------------------------------------------------------------------

   // Set the boundary flags from the geometry. Has to be redone after every refinement step because
   // newly created blocks start with an empty (or migrated) flag field.
   auto boundariesConfig = env.config()->getOneBlock("Boundaries");
   FlagUID objBoundaryUID("NoSlip");

   auto setupFlagField = [&]() {
      geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldId, boundariesConfig);
      for (auto& block : *blocks)
      {
         auto * flagField = block.getData<FlagField_T>(flagFieldId);
         if ( !flagField->flagExists(objBoundaryUID))
            flagField->registerFlag(objBoundaryUID);
         auto flag = flagField->getFlag(objBoundaryUID);

         WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(flagField,
                                                         Cell cell(x,y,z);
                                                         auto midPoint = blocks->getGlobalCellCenterFromBlockLocalCell(cell, block);
                                                         if (contains(sphere, midPoint)) {
                                                            flagField->addFlag(cell, flag);
                                                         }
         )
      }
      geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldId, FluidFlagUID);
   };

   setupFlagField();

   for (auto& block : *blocks)
   {
      auto* velField = block.getData< VelocityField_T >(velFieldId);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(velField,
         for (uint_t d = 0; d < uint_c(3); ++d) velField->get(x, y, z, d) = initialVel[d];
      )
      sweepCollection.initialise(&block, cell_idx_c(2));
   }

   BoundaryCollection_T boundaryCollection(blocks, flagFieldId, pdfFieldId, FluidFlagUID, initialVel[0]);

   // Refill the index vectors of the boundary collection from the (rebuilt) flag field.
   auto refillBoundaries = [&]() {
      boundaryCollection.OutflowBCObject->fillFromFlagField< FlagField_T >(blocks, flagFieldId, OutflowFlagUID, FluidFlagUID);
      boundaryCollection.UBBBCObject->fillFromFlagField< FlagField_T >(blocks, flagFieldId, UBBFlagUID, FluidFlagUID);
      boundaryCollection.NoSlipBCObject->fillFromFlagField< FlagField_T >(blocks, flagFieldId, NoSlipFlagUID, FluidFlagUID);
      boundaryCollection.FreeSlipBCObject->fillFromFlagField< FlagField_T >(blocks, flagFieldId, FreeSlipFlagUID, FluidFlagUID);
   };

   lbm_generated::BasicRecursiveTimeStep< PdfField_T, SweepCollection_T, BoundaryCollection_T > LBMMeshRefinement(
      blocks, pdfFieldId, sweepCollection, boundaryCollection, communication, packInfo);

   // ------------------------------------------------------------------------------------------------------------------
   //   Adaptive refinement step
   // ------------------------------------------------------------------------------------------------------------------
   VelocityGradientRefinement refinementCriterion(velFieldId, sphere, lowerRefinementLimit, upperRefinementLimit,
                                                  refinementLevels);

   auto adaptiveRefinementStep = [&]() {
      // The refinement criterion evaluates the velocity field, so make sure it is up to date.
      for (auto& block : *blocks)
         sweepCollection.calculateMacroscopicParameters(&block);

      forest->setRefreshMinTargetLevelDeterminationFunction(refinementCriterion);
      forest->refresh();

      // Re-establish all data that depends on the (changed) block structure.
      for (auto& block : *blocks)
         packInfo->recalculateNonuniformCommData(&block);

      setupFlagField();
      refillBoundaries();

      WALBERLA_LOG_INFO_ON_ROOT("Adaptive refinement: depth " << blocks->getDepth() << ", "
                                << forest->getNumberOfBlocks() << " blocks on root process")
   };

   // Refine once before the simulation starts so that the sphere is well resolved from the beginning.
   adaptiveRefinementStep();

   // ------------------------------------------------------------------------------------------------------------------
   //   Time loop
   // ------------------------------------------------------------------------------------------------------------------
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   if (refinementFrequency > 0)
   {
      timeloop.addFuncBeforeTimeStep(
         [&, refinementFrequency]() {
            const uint_t step = timeloop.getCurrentTimeStep();
            if (step > uint_c(0) && step % refinementFrequency == 0)
            {
               adaptiveRefinementStep();
            }
         },
         "Adaptive mesh refinement");
   }

   timeloop.addFuncBeforeTimeStep(LBMMeshRefinement, "Recursive LBM time step");

   timeloop.addFuncAfterTimeStep(timing::RemainingTimeLogger(timeloop.getNrOfTimeSteps(), 10), "remaining time logger");

   // VTK output
   if (vtkWriteFrequency > 0)
   {
      auto vtkOutput = vtk::createVTKOutput_BlockData(*blocks, "vtk", vtkWriteFrequency, 0, true, "vtk_out", "simulation_step", false, true, true, false, 0, false);
      auto velocityWriter = make_shared< field::VTKWriter< VelocityField_T > >(velFieldId, "velocity");
      auto densityWriter  = make_shared< field::VTKWriter< ScalarField_T > >(densityId, "density");
      vtkOutput->addCellDataWriter(velocityWriter);
      vtkOutput->addCellDataWriter(densityWriter);

      field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldId);
      fluidFilter.addFlag(FluidFlagUID);
      vtkOutput->addCellInclusionFilter(fluidFilter);

      vtkOutput->addBeforeFunction([&]() {
         for (auto& block : *blocks)
            sweepCollection.calculateMacroscopicParameters(&block);
      });

      timeloop.addFuncAfterTimeStep(vtk::writeFiles(vtkOutput), "VTK Output");

      auto domainDecompositionOutput = vtk::createVTKOutput_DomainDecomposition(
         *blocks, "domain_decomposition", vtkWriteFrequency, "vtk_out", "simulation_step", false, true, true, false);
      timeloop.addFuncAfterTimeStep(vtk::writeFiles(domainDecompositionOutput), "Domain decomposition output");
   }

   // ------------------------------------------------------------------------------------------------------------------
   //   Run
   // ------------------------------------------------------------------------------------------------------------------
   WcTimer simTimer;
   WcTimingPool timingPool;
   simTimer.start();
   timeloop.run(timingPool);
   simTimer.end();

   lbm_generated::PerformanceEvaluation< FlagField_T > const performance(blocks, flagFieldId, FluidFlagUID);
   performance.logResultOnRoot(timesteps, simTimer.max());
   timingPool.unifyRegisteredTimersAcrossProcesses();
   timingPool.logResultOnRoot(timing::REDUCE_TOTAL, true);

   return EXIT_SUCCESS;
}
} // namespace walberla

int main(int argc, char** argv) { return walberla::main(argc, argv); }
