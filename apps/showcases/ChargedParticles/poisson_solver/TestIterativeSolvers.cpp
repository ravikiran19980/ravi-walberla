#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "boundary/all.h"

#include "core/Environment.h"
#include "core/grid_generator/SCIterator.h"
#include "core/logging/all.h"
#include "core/waLBerlaBuildInfo.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "core/math/all.h"

#include "PoissonSolver.h"

namespace walberla {

enum Testcase { TEST_DIRICHLET_1, TEST_DIRICHLET_2, TEST_NEUMANN };

using ScalarField_T = GhostLayerField< real_t, 1 >;

template < typename PdeField, Testcase testcase >
void applyDirichletFunction(const shared_ptr< StructuredBlockStorage > & blocks, const math::AABB & domainAABB, const stencil::Direction& direction,
                            IBlock* block, PdeField* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz) {

   WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(interval,
      real_t boundaryCoord_x = 0.;
      real_t boundaryCoord_y = 0.;
      real_t boundaryCoord_z = 0.;

      const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x, y, z));
      auto cellCenter     = cellAABB.center();

      // snap cell position to actual domain position
      switch (direction) {
         case stencil::W:
            boundaryCoord_x = domainAABB.xMin();
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::E:
            boundaryCoord_x = domainAABB.xMax();
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::S:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = domainAABB.yMin();
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::N:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = domainAABB.yMax();
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::B:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = domainAABB.zMin();
            break;
         case stencil::T:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = domainAABB.zMax();
            break;
         default:
            WALBERLA_ABORT("Unknown direction");
      }

      // use positions normalized to unit cube
      boundaryCoord_x /= domainAABB.size(0);
      boundaryCoord_y /= domainAABB.size(1);
      boundaryCoord_z /= domainAABB.size(2);

      auto funcVal = real_c(0);
      switch (testcase) {
         case TEST_DIRICHLET_1:
            funcVal = (boundaryCoord_x * boundaryCoord_x) - real_c(0.5) * (boundaryCoord_y * boundaryCoord_y) - real_c(0.5) * (boundaryCoord_z * boundaryCoord_z);
            break;
         case TEST_DIRICHLET_2:
            funcVal = real_c( sin  ( math::pi * boundaryCoord_x ) ) *
                      real_c( sin  ( math::pi * boundaryCoord_y ) ) *
                      real_c( sinh ( math::root_two * math::pi * boundaryCoord_z ) );
            break;
         default:
            WALBERLA_ABORT("Unknown testcase");
      }
      p->get(x, y, z) = real_c(2) * funcVal - p->get(x + cx, y + cy, z + cz);
   )
}

void resetSolution(const shared_ptr< StructuredBlockStorage > & blocks, BlockDataID & solution, BlockDataID & solutionCpy) {
   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* solutionField    = block->getData< ScalarField_T >(solution);
      ScalarField_T* solutionFieldCpy = block->getData< ScalarField_T >(solutionCpy);

      // reset fields
      solutionField->set(real_c(0));
      solutionFieldCpy->set(real_c(0));
   }
}

void resetRHS(const shared_ptr< StructuredBlockStorage > & blocks, BlockDataID & rhs) {
   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* rhsField = block->getData< ScalarField_T >(rhs);

      // reset field
      rhsField->set(real_c(0));
   }
}

// solve two different scenarios (dirichlet scenario and neumann scenario) with different analytical solutions and setups
template < Testcase testcase >
void solve(const shared_ptr< StructuredBlockForest > & blocks,
           const math::AABB & domainAABB, BlockDataID & solution, BlockDataID & solutionCpy, BlockDataID & rhs,
           const uint_t numIter, bool useAbsResThres, real_t resThres, uint_t resCheckFreq) {

   const bool useDirichlet = testcase == TEST_DIRICHLET_1 || testcase == TEST_DIRICHLET_2;

   // set boundary handling depending on scenario
   std::function< void () > boundaryHandling = {};

   if constexpr (useDirichlet) {
      // set dirichlet function per domain face
      auto dirichletFunction = DirichletFunctionDomainBoundary< ScalarField_T >(*blocks, solution);

#define GET_BOUNDARY_LAMBDA(dir) \
         [&blocks, &domainAABB](IBlock* block, ScalarField_T* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz) { \
            applyDirichletFunction< ScalarField_T, testcase >(blocks, domainAABB, dir, block, p, interval, cx, cy, cz); \
         }

      dirichletFunction.setFunction(stencil::W, GET_BOUNDARY_LAMBDA(stencil::W));
      dirichletFunction.setFunction(stencil::E, GET_BOUNDARY_LAMBDA(stencil::E));
      dirichletFunction.setFunction(stencil::S, GET_BOUNDARY_LAMBDA(stencil::S));
      dirichletFunction.setFunction(stencil::N, GET_BOUNDARY_LAMBDA(stencil::N));
      dirichletFunction.setFunction(stencil::B, GET_BOUNDARY_LAMBDA(stencil::B));
      dirichletFunction.setFunction(stencil::T, GET_BOUNDARY_LAMBDA(stencil::T));

      boundaryHandling = dirichletFunction;
   } else {
       boundaryHandling = pde::NeumannDomainBoundary< ScalarField_T >(*blocks, solution);
   }

   // solvers: Jacobi and SOR

   auto poissonSolverJacobi = PoissonSolver< WALBERLA_JACOBI > (solution, solutionCpy, rhs, blocks, boundaryHandling, numIter, useAbsResThres, resThres, resThres, resCheckFreq);
   auto poissonSolverDampedJac = PoissonSolver< DAMPED_JACOBI > (solution, solutionCpy, rhs, blocks, boundaryHandling, numIter, useAbsResThres, resThres, resThres, resCheckFreq);
   auto poissonSolverSOR = PoissonSolver< WALBERLA_SOR > (solution, solutionCpy, rhs, blocks, boundaryHandling, numIter, useAbsResThres, resThres, resThres, resCheckFreq);

   // calc error depending on scenario

   auto computeMaxError = [&blocks, &solution, &domainAABB]() {
      real_t error = real_c(0);

      for (auto block = blocks->begin(); block != blocks->end(); ++block) {
         ScalarField_T* solutionField = block->getData< ScalarField_T >(solution);

         real_t blockResult = real_t(0);
         WALBERLA_FOR_ALL_CELLS_XYZ_OMP(solutionField, omp parallel for schedule(static) reduction(max: blockResult),
                                    const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x,y,z));
                                    auto cellCenter = cellAABB.center();

                                    // use positions normalized to unit cube
                                    real_t scaleX = real_c(1) / domainAABB.size(0);
                                    real_t scaleY = real_c(1) / domainAABB.size(1);
                                    real_t scaleZ = real_c(1) / domainAABB.size(2);

                                    real_t posX = cellCenter[0] * scaleX;
                                    real_t posY = cellCenter[1] * scaleY;
                                    real_t posZ = cellCenter[2] * scaleZ;

                                    real_t analyticalSol;

                                    // analytical solution of problem with neumann/dirichlet boundaries
                                    switch (testcase) {
                                       case TEST_DIRICHLET_1:
                                          analyticalSol =                 (posX * posX)
                                                          - real_c(0.5) * (posY * posY)
                                                          - real_c(0.5) * (posZ * posZ);
                                          break;
                                       case TEST_DIRICHLET_2:
                                          analyticalSol = real_c( sin ( math::pi * posX ) ) *
                                                          real_c( sin ( math::pi * posY ) ) *
                                                          real_c( sinh ( math::root_two * math::pi * posZ ) );
                                          break;
                                       case TEST_NEUMANN:
                                          analyticalSol = real_c( cos ( real_c(2) * math::pi * posX ) ) *
                                                          real_c( cos ( real_c(2) * math::pi * posY ) ) *
                                                          real_c( cos ( real_c(2) * math::pi * posZ ) );
                                          break;
                                       default:
                                          WALBERLA_ABORT("Unknown testcase");
                                    }

                                    real_t currErr = real_c(fabs(solutionField->get(x, y, z) - analyticalSol));
                                    blockResult = std::max(blockResult, currErr);
         )
         error = std::max(error, blockResult);
      }
      mpi::allReduceInplace( error, mpi::MAX );

      return error;
   };

   // init rhs depending on scenario

   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* rhsField = block->getData< ScalarField_T >(rhs);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         rhsField,

         const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x, y, z));
         auto cellCenter = cellAABB.center();

         // use positions normalized to unit cube
         real_t scaleX = real_c(1) / domainAABB.size(0);
         real_t scaleY = real_c(1) / domainAABB.size(1);
         real_t scaleZ = real_c(1) / domainAABB.size(2);

         real_t posX = cellCenter[0] * scaleX;
         real_t posY = cellCenter[1] * scaleY;
         real_t posZ = cellCenter[2] * scaleZ;

         switch (testcase) {
            case TEST_DIRICHLET_1:
               rhsField->get(x, y, z) = real_c(0);
               break;
            case TEST_DIRICHLET_2:
               rhsField->get(x, y, z) = real_c( ( math::pi * math::pi ) * ( ( scaleX * scaleX ) + ( scaleY * scaleY ) - real_c(2) * ( scaleZ * scaleZ ) ) ) *
                                        real_c( sin ( math::pi * posX ) ) *
                                        real_c( sin ( math::pi * posY ) ) *
                                        real_c( sinh ( math::root_two * math::pi * posZ ) );
               break;
            case TEST_NEUMANN:
               rhsField->get(x, y, z) = real_c(4) * math::pi * math::pi *
                                        real_c( (pow(scaleX, 2) + pow(scaleY, 2) + pow(scaleZ, 2)) ) *
                                        real_c( cos(real_c(2) * math::pi * posX) ) *
                                        real_c( cos(real_c(2) * math::pi * posY) ) *
                                        real_c( cos(real_c(2) * math::pi * posZ) );
               break;
            default:
               WALBERLA_ABORT("Unknown testcase");
         }
      )
   }

   auto initError = computeMaxError();
   WALBERLA_LOG_INFO_ON_ROOT("Initial error is: " << initError);

   // solve with jacobi
   WALBERLA_LOG_INFO_ON_ROOT("-- Solve using Jacobi --");
   poissonSolverJacobi();
   auto errJac = computeMaxError();
   WALBERLA_LOG_INFO_ON_ROOT("Error after Jacobi solver is: " << errJac);

   // solve with damped jacobi
   WALBERLA_LOG_INFO_ON_ROOT("-- Solve using (damped) Jacobi --");
   resetSolution(blocks, solution, solutionCpy); // reset solutions and solve anew
   poissonSolverDampedJac();
   auto errDampedJac = computeMaxError();
   WALBERLA_LOG_INFO_ON_ROOT("Error after (damped) Jacobi solver is: " << errDampedJac);

   // solve with SOR
   WALBERLA_LOG_INFO_ON_ROOT("-- Solve using SOR --");
   resetSolution(blocks, solution, solutionCpy); // reset solutions and solve anew
   poissonSolverSOR();
   auto errSOR = computeMaxError();
   WALBERLA_LOG_INFO_ON_ROOT("Error after SOR solver is: " << errSOR);
}

// solve two different charged particle scenarios (dirichlet scenario and neumann scenario) with different setups
template < bool useDirichlet >
void solveChargedParticles(const shared_ptr< StructuredBlockForest > & blocks,
                           const math::AABB & domainAABB, BlockDataID & solution, BlockDataID & solutionCpy, BlockDataID & rhs) {

   // solvers: Jacobi and SOR

   auto numIter = uint_c(20000);
   auto useAbsResThreshold = true;
   auto resThres = real_c(1e-5);
   auto resCheckFreq = uint_c(1000);

   // set boundary handling depending on scenario
   std::function< void () > boundaryHandling = {};
   std::vector< BoundaryCondition > boundaryConditions;

   if constexpr (useDirichlet) {
      for (const auto& e : stencil::D3Q6::dir)
         boundaryConditions.emplace_back(e, "Dirichlet", 0_r);

      boundaryHandling = DirichletDomainBoundary< ScalarField_T >(*blocks, solution, boundaryConditions);
   } else {
      boundaryHandling = pde::NeumannDomainBoundary< ScalarField_T >(*blocks, solution);
   }

   auto poissonSolverJacobi = PoissonSolver< DAMPED_JACOBI > (solution, solutionCpy, rhs, blocks, boundaryHandling, numIter, useAbsResThreshold, resThres, resThres, resCheckFreq);
   auto poissonSolverSOR = PoissonSolver< WALBERLA_SOR > (solution, solutionCpy, rhs, blocks, boundaryHandling, numIter, useAbsResThreshold, resThres, resThres, resCheckFreq);

   // init rhs with two charged particles

   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* rhsField = block->getData< ScalarField_T >(rhs);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         rhsField,

         const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x, y, z));
         auto cellCenter = cellAABB.center();

         const real_t x0 = domainAABB.xMin() + real_c(0.45) * domainAABB.size(0);
         const real_t y0 = domainAABB.yMin() + real_c(0.45) * domainAABB.size(1);
         const real_t z0 = domainAABB.zMin() + real_c(0.45) * domainAABB.size(2);
         const real_t r0 = real_c(0.08) * domainAABB.size(0);
         const real_t s0 = real_c(1);

         const real_t x1 = domainAABB.xMin() + real_c(0.65) * domainAABB.size(0);
         const real_t y1 = domainAABB.yMin() + real_c(0.65) * domainAABB.size(1);
         const real_t z1 = domainAABB.zMin() + real_c(0.65) * domainAABB.size(2);
         const real_t r1 = real_c(0.08) * domainAABB.size(0);
         const real_t s1 = real_c(1);

         real_t posX = cellCenter[0];
         real_t posY = cellCenter[1];
         real_t posZ = cellCenter[2];

         if ( ( real_c( pow( posX - x0, 2 ) ) + real_c( pow( posY - y0, 2 ) ) + real_c( pow( posZ - z0, 2 ) ) ) < real_c( pow( r0, 2 ) ) ) {
            auto relDistPos0 = real_c( sqrt( real_c( pow( ( posX - x0 ) / r0, 2 ) ) + real_c( pow( ( posY - y0 ) / r0, 2 ) ) + real_c( pow( ( posZ - z0 ) / r0, 2 ) ) ) );
            rhsField->get(x, y, z) = -s0 * ( real_c(1) - relDistPos0 );
         } else if ( ( real_c ( pow( posX - x1, 2 ) ) + real_c ( pow( posY - y1, 2 ) ) + real_c ( pow( posZ - z1, 2 ) ) ) < real_c ( pow( r1, 2 ) ) ) {
            auto relDistPos1 = real_c( sqrt( real_c( pow( ( posX - x1 ) / r1, 2 ) ) + real_c( pow( ( posY - y1 ) / r1, 2 ) ) + real_c( pow( ( posZ - z1 ) / r1, 2 ) ) ) );
            rhsField->get(x, y, z) = -s1 * ( real_c(1) - relDistPos1);
         } else {
            rhsField->get(x, y, z) = real_c(0);
         }
      )
   }

   // solve with jacobi
   poissonSolverJacobi();

   // solve with SOR
   resetSolution(blocks, solution, solutionCpy); // reset solutions and solve anew
   poissonSolverSOR();
}

int main(int argc, char** argv)
{
   Environment env(argc, argv);

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));

   logging::Logging::instance()->setLogLevel(logging::Logging::LogLevel::DETAIL);
   logging::Logging::instance()->includeLoggingToFile( "log" );

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   auto cfgFile = env.config();
   if (!cfgFile) WALBERLA_ABORT("Usage: " << argv[0] << " < path-to-configuration-file > \n");

   Config::BlockHandle setup                   = cfgFile->getBlock("Setup");
   const Vector3< uint_t > numBlocksPerDim     = setup.getParameter< Vector3< uint_t > >("numBlocks");
   const Vector3< uint_t > numCellsBlockPerDim = setup.getParameter< Vector3< uint_t > >("numCellsPerBlock");
   const uint_t maxIterations                  = setup.getParameter< uint_t >("maxIterations");
   const bool useAbsResThreshold               = setup.getParameter< bool >("useAbsResThreshold");
   const real_t resThres                       = setup.getParameter< real_t >("resThreshold");
   const uint_t resCheckFreq                   = setup.getParameter< uint_t >("resCheckFreq");

   auto domainAABB = math::AABB(0, 0, 0, 1, 1, 1);
   WALBERLA_LOG_INFO_ON_ROOT("Domain sizes are: x = " << domainAABB.size(0) << ", y = " << domainAABB.size(1) << ", z = " << domainAABB.size(2));

   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      domainAABB,
      numBlocksPerDim[0], numBlocksPerDim[1], numBlocksPerDim[2],
      numCellsBlockPerDim[0], numCellsBlockPerDim[1], numCellsBlockPerDim[2],
      true,
      false, false, false,
      false);

   BlockDataID rhs = field::addToStorage< ScalarField_T >(blocks, "rhs", 0, field::fzyx, 1);
   BlockDataID solution = field::addToStorage< ScalarField_T >(blocks, "solution", 0, field::fzyx, 1);
   BlockDataID solutionCpy = field::addCloneToStorage< ScalarField_T >(blocks, solution, "solution (copy)");

   /* 1. analytical tests */

   // first solve neumann problem...
   WALBERLA_LOG_INFO_ON_ROOT("Run analytical test cases...")
   WALBERLA_LOG_INFO_ON_ROOT("- Solving analytical Neumann problem with Jacobi and SOR... -")
   solve< TEST_NEUMANN > (blocks, domainAABB, solution, solutionCpy, rhs, maxIterations, useAbsResThreshold, resThres, resCheckFreq);

   // ... then solve dirichlet problems
   resetRHS(blocks, rhs); // reset fields and solve anew
   resetSolution(blocks, solution, solutionCpy);
   WALBERLA_LOG_INFO_ON_ROOT("- Solving analytical Dirichlet problem (1) with Jacobi and SOR... -")
   solve< TEST_DIRICHLET_1 > (blocks, domainAABB, solution, solutionCpy, rhs, maxIterations, useAbsResThreshold, resThres, resCheckFreq);

   resetRHS(blocks, rhs); // reset fields and solve anew
   resetSolution(blocks, solution, solutionCpy);
   WALBERLA_LOG_INFO_ON_ROOT("- Solving analytical Dirichlet problem (2) with Jacobi and SOR... -")
   solve< TEST_DIRICHLET_2 > (blocks, domainAABB, solution, solutionCpy, rhs, maxIterations, useAbsResThreshold, resThres, resCheckFreq);

   /* 2. experimental charged particle tests */

   WALBERLA_LOG_INFO_ON_ROOT("Run charged particle test cases...")
   resetRHS(blocks, rhs); // reset fields and solve anew
   resetSolution(blocks, solution, solutionCpy);

   // neumann
   WALBERLA_LOG_INFO_ON_ROOT("- Run charged particles with Neumann boundaries... -")
   solveChargedParticles < false > (blocks, domainAABB, solution, solutionCpy, rhs);

   // dirichlet
   WALBERLA_LOG_INFO_ON_ROOT("- Run charged particles with Dirichlet (val = 0) boundaries... -")
   resetSolution(blocks, solution, solutionCpy); // reset solutions and solve anew
   solveChargedParticles < true > (blocks, domainAABB, solution, solutionCpy, rhs);

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { walberla::main(argc, argv); }

