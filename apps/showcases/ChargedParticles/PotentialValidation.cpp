#include "blockforest/Initialization.h"

#include "core/Environment.h"
#include "core/grid_generator/SCIterator.h"
#include "core/logging/all.h"
#include "core/waLBerlaBuildInfo.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "geometry/bodies/Sphere.h"

#include <core/math/all.h>

#include "PoissonSolver.h"

namespace walberla {

using ScalarField_T = GhostLayerField< real_t, 1 >;

template < typename PdeField >
void applyPotentialBCs(const shared_ptr< StructuredBlockStorage > & blocks, const math::AABB & domainAABB, const stencil::Direction& direction,
                       IBlock* block, PdeField* p,
                       const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz,
                       const real_t x0, const real_t y0, const real_t z0,
                       const real_t q_e, const real_t eps_e) {

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

      const real_t preFactor = real_c(1) / (real_c(4) * math::pi * eps_e);

      const real_t dist = real_c(sqrt(pow(boundaryCoord_x - x0, 2) + pow(boundaryCoord_y - y0, 2) + pow(boundaryCoord_z - z0, 2)));

      auto funcVal = preFactor * (q_e / dist);

      p->get(x, y, z) = real_c(2) * funcVal - p->get(x + cx, y + cy, z + cz);
   )
}

void solveElectrostaticPoisson(const shared_ptr< StructuredBlockForest >& blocks,
                               real_t dx, real_t radiusScaleFactor,
                               const math::AABB& domainAABB,
                               const bool useOverlapFraction,
                               BlockDataID& solution, BlockDataID& solutionCpy,
                               BlockDataID& rhs,
                               BlockDataID& analytical)
{
   auto numIter = uint_c(100000);
   auto resThres = real_c(1e-14);
   auto resCheckFreq = uint_c(1000);

   const real_t x0 = domainAABB.xMin() + real_c(0.5) * domainAABB.size(0);
   const real_t y0 = domainAABB.yMin() + real_c(0.5) * domainAABB.size(1);
   const real_t z0 = domainAABB.zMin() + real_c(0.5) * domainAABB.size(2);
   const real_t e = real_c(1.602176E-19); // ampereseconds
   const real_t eps_e = real_c(78.5 * 8.854187812813E-12); // ampereseconds / voltmeter
   const real_t q_e = real_c(8000) * e; // ampereseconds
   auto R_L = radiusScaleFactor * real_c(6) * dx;

   Vector3< real_t > const position(x0, y0, z0);
   geometry::Sphere const sphere(position, R_L);

   // set dirichlet function per domain face
   auto dirichletFunction = DirichletFunctionDomainBoundary< ScalarField_T >(*blocks, solution);

#define GET_POTENTIAL_BCS_LAMBDA(dir) \
   [&blocks, &domainAABB, &x0, &y0, &z0, &q_e, &eps_e] \
      (IBlock* block, ScalarField_T* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz) { \
         applyPotentialBCs< ScalarField_T >(blocks, domainAABB, dir, block, p, interval, cx, cy, cz, x0, y0, z0, q_e, eps_e); \
      }


   dirichletFunction.setFunction(stencil::W, GET_POTENTIAL_BCS_LAMBDA(stencil::W));
   dirichletFunction.setFunction(stencil::E, GET_POTENTIAL_BCS_LAMBDA(stencil::E));
   dirichletFunction.setFunction(stencil::S, GET_POTENTIAL_BCS_LAMBDA(stencil::S));
   dirichletFunction.setFunction(stencil::N, GET_POTENTIAL_BCS_LAMBDA(stencil::N));
   dirichletFunction.setFunction(stencil::B, GET_POTENTIAL_BCS_LAMBDA(stencil::B));
   dirichletFunction.setFunction(stencil::T, GET_POTENTIAL_BCS_LAMBDA(stencil::T));

   auto poissonSolverSOR = PoissonSolver< WALBERLA_SOR, true > (solution, solutionCpy, rhs, blocks, numIter, resThres, resCheckFreq, dirichletFunction);

   // init rhs with two charged particles

   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* rhsField = block->getData< ScalarField_T >(rhs);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         rhsField,

         const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x, y, z));
         auto cellCenter = cellAABB.center();

         if (useOverlapFraction) {
            const real_t overlap = geometry::overlapFraction(sphere, cellCenter, dx, 3);

            auto densityCell       = overlap * q_e / ((real_c(4) / real_c(3)) * math::pi * real_c(pow(R_L, 3)));
            rhsField->get(x, y, z) = densityCell / eps_e;
         } else {
            const real_t posX = cellCenter[0];
            const real_t posY = cellCenter[1];
            const real_t posZ = cellCenter[2];

            const real_t squareSum = real_c(pow(posX - x0, 2) + pow(posY - y0, 2) + pow(posZ - z0, 2));
            const real_t dist      = real_c(sqrt(squareSum));

            if (dist > R_L)
               rhsField->get(x, y, z) = 0_r;
            else
               rhsField->get(x, y, z) = -(3 * q_e) / (4 * math::pi * real_c(pow(R_L, 3)) * eps_e);
         })
   }

   for (auto block = blocks->begin(); block != blocks->end(); ++block) {
      ScalarField_T* analyticalField = block->getData< ScalarField_T >(analytical);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         analyticalField, const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x, y, z));
         auto cellCenter = cellAABB.center();

         const real_t posX = cellCenter[0];
         const real_t posY = cellCenter[1];
         const real_t posZ = cellCenter[2];

         const real_t dist = real_c(sqrt(pow(posX - x0, 2) + pow(posY - y0, 2) + pow(posZ - z0, 2)));

         const real_t preFactor = real_c(1) / (real_c(4) * math::pi * eps_e);

         if (dist < R_L) {
            analyticalField->get(x, y, z) = preFactor * (q_e / (real_c(2) * R_L)) * (real_c(3) - real_c(pow(dist / R_L, 2)));
         } else {
            analyticalField->get(x, y, z) = preFactor * (q_e / dist);
         }
      )
   }

   // count cells
   auto cells( uint_t(0) );
   for( auto block = blocks->begin( ); block != blocks->end(); ++block ) {
      cells += blocks->getNumberOfXCells(*block) * blocks->getNumberOfYCells(*block) * blocks->getNumberOfZCells(*block);
   }
   mpi::allReduceInplace( cells, mpi::SUM );

   // lambda for L2 error calculation
   auto computeL2Error = [&blocks, &cells, &solution, &analytical]() {
      real_t error = real_c(0);

      for (auto block = blocks->begin(); block != blocks->end(); ++block) {
         auto solutionField = block->getData< ScalarField_T >( solution );
         auto analyticalField = block->getData< ScalarField_T >( analytical );

         auto blockResult(real_t(0));

        WALBERLA_FOR_ALL_CELLS_XYZ_OMP(solutionField, omp parallel for schedule(static) reduction(+: blockResult),
            real_t currErr = solutionField->get(x, y, z) - analyticalField->get(x, y, z);
            blockResult += currErr * currErr;
        ) // WALBERLA_FOR_ALL_CELLS_XYZ_OMP

        error += blockResult;
      }
      mpi::allReduceInplace( error, mpi::SUM );

      return std::sqrt( error / real_c(cells));

   };

   // solve with SOR
   poissonSolverSOR();
   auto errSOR = computeL2Error();
   WALBERLA_LOG_INFO_ON_ROOT("Error after SOR solver is: " << errSOR);
}

int main(int argc, char** argv)
{
   const Environment env(argc, argv);

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));

   logging::Logging::instance()->setLogLevel(logging::Logging::LogLevel::DETAIL);
   logging::Logging::instance()->includeLoggingToFile( "log" );

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   auto useOverlapFraction = true;
   auto dx                 = real_c(1E-4);
   auto numCellsPerDim     = 32;
   auto numBlocksPerDim    = 2;
   auto refCellCount       = 64;
   auto radiusScaleFactor  = real_c(numCellsPerDim) / real_c(refCellCount);
   auto domainAABB         = math::AABB(0, 0, 0, real_c(numCellsPerDim) * dx, real_c(numCellsPerDim) * dx, real_c(numCellsPerDim) * dx);
   WALBERLA_LOG_INFO_ON_ROOT("Domain sizes are: x = " << domainAABB.size(0) << ", y = " << domainAABB.size(1) << ", z = " << domainAABB.size(2));

   const shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      domainAABB,
      uint_c(numBlocksPerDim), uint_c(numBlocksPerDim),uint_c(numBlocksPerDim),
      uint_c(numCellsPerDim), uint_c(numCellsPerDim), uint_c(numCellsPerDim),
      true,
      false, false, false,
      false);

   BlockDataID rhs = field::addToStorage< ScalarField_T >(blocks, "rhs", 0, field::fzyx, 1);
   BlockDataID solution = field::addToStorage< ScalarField_T >(blocks, "solution", 0, field::fzyx, 1);
   BlockDataID solutionCpy = field::addCloneToStorage< ScalarField_T >(blocks, solution, "solution (copy)");
   BlockDataID analytical = field::addCloneToStorage< ScalarField_T >(blocks, solution, "analytical");

   solveElectrostaticPoisson (blocks, dx, radiusScaleFactor, domainAABB, useOverlapFraction, solution, solutionCpy, rhs, analytical);

   auto vtkWriter = vtk::createVTKOutput_BlockData(*blocks, "block_data", uint_c(1), uint_c(0), true, "VTK" );
   vtkWriter->addCellDataWriter(make_shared<field::VTKWriter<ScalarField_T> >(solution, "solution"));
   vtkWriter->addCellDataWriter(make_shared<field::VTKWriter<ScalarField_T> >(rhs, "rhs"));
   vtkWriter->addCellDataWriter(make_shared<field::VTKWriter<ScalarField_T> >(analytical, "analytical"));
   vtk::writeFiles(vtkWriter)();

   return EXIT_SUCCESS;
}

} // namespace walberla

int main(int argc, char** argv) { walberla::main(argc, argv); }


