//
// Created by avnss on 4/21/2023.
//

#ifndef WALBERLA_ERRORNORMS_H
#define WALBERLA_ERRORNORMS_H

namespace walberla {
    real_t computeErrorMax(const shared_ptr<StructuredBlockForest>& blocks, BlockDataID & NumericalSol,  const math::AABB & domainAABB,const real_t radius,const real_t epsilon, const real_t charge,const BlockDataID& potentialErrorID) {
        real_t error = real_c(0);
        real_t normmax = real_c(0);


        for (auto block = blocks->begin(); block != blocks->end(); ++block) {
            ScalarField_T* solutionField = block->getData< ScalarField_T >(NumericalSol);

            real_t blockResult = real_t(0);
            real_t blockResult_analytical = real_t(0);
            WALBERLA_FOR_ALL_CELLS_XYZ_OMP(solutionField, omp parallel for schedule(static) reduction(max: blockResult, max: blockResult_analytical),
                                           const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x,y,z));
                                                   auto cellCenter = cellAABB.center();
                                                   if(cellCenter[0] != 0.5 && cellCenter[0] != 99.5 && cellCenter[1] != 0.5 && cellCenter[1] != 99.5 && cellCenter[2] != 0.5 && cellCenter[2] != 99.5) {
                                                       real_t posX = cellCenter[0];
                                                       real_t posY = cellCenter[1];
                                                       real_t posZ = cellCenter[2];

                                                       real_t analyticalSol;

                                                       // analytical solution of problem with neumann/dirichlet boundaries

                                                       real_t distance = real_c(
                                                               sqrt(pow((domainAABB.center()[0] - posX), 2) +
                                                                    pow((domainAABB.center()[1] - posY), 2) +
                                                                    pow((domainAABB.center()[2] - posZ), 2)));
                                                       if (distance >= radius) {
                                                           analyticalSol =
                                                                   (1 / (4 * math::pi * epsilon)) * (charge / distance);
                                                       } else {
                                                           analyticalSol = (1 / (4 * math::pi * epsilon)) *
                                                                           (charge / (2 * radius)) *
                                                                           (3 - pow((distance / radius), 2));
                                                       }

                                                       real_t currErr = real_c(
                                                               fabs(solutionField->get(x, y, z) - analyticalSol) /
                                                               analyticalSol);
                                                       blockResult = std::max(blockResult, currErr);
                                                       blockResult_analytical = std::max(blockResult_analytical,
                                                                                         analyticalSol);

                                                   }
            )
            error = std::max(error, blockResult);
            normmax = std::max(normmax, blockResult_analytical);

        }

        mpi::allReduceInplace( error, mpi::MAX );
        mpi::allReduceInplace( normmax, mpi::MAX );


        return error;
    }



    real_t computeErrorL2(const shared_ptr<StructuredBlockForest>& blocks, BlockDataID & NumericalSol,  const math::AABB & domainAABB,const real_t radius,const real_t epsilon, const real_t charge,const BlockDataID& potentialErrorID) {
        real_t error = real_c(0);
        real_t normL2 = real_c(0);
        real_t cells = real_t(0);




        for (auto block = blocks->begin(); block != blocks->end(); ++block) {

            ScalarField_T* solutionField = block->getData< ScalarField_T >(NumericalSol);
            ScalarField_T* potentialErrorField = block->getData< ScalarField_T >(potentialErrorID);

            real_t blockResult = real_t(0);
            real_t blockResult_analytical = real_t(0);
            real_t block_local_cells = real_t(0);
            WALBERLA_FOR_ALL_CELLS_XYZ_OMP(solutionField, omp parallel for schedule(static) reduction(+: blockResult, +: blockResult_analytical, +: block_local_cells),
                                           const auto cellAABB = blocks->getBlockLocalCellAABB(*block, Cell(x,y,z));
                                                   auto cellCenter = cellAABB.center();

                                                       real_t posX = cellCenter[0];
                                                       real_t posY = cellCenter[1];
                                                       real_t posZ = cellCenter[2];

                                                       real_t analyticalSol;

                                                       // analytical solution of problem with neumann/dirichlet boundaries

                                                       real_t distance = real_c(
                                                               sqrt(pow((domainAABB.center()[0] - posX), 2) +
                                                                    pow((domainAABB.center()[1] - posY), 2) +
                                                                    pow((domainAABB.center()[2] - posZ), 2)));
                                                       if (distance >= radius) {
                                                           analyticalSol =
                                                                   (1 / (4 * math::pi * epsilon)) * (charge / distance);
                                                       } else {
                                                           analyticalSol = (1 / (4 * math::pi * epsilon)) *
                                                                           (charge / (2 * radius)) *
                                                                           (3 - pow((distance / radius), 2));
                                                       }

                                                       //potentialAnalyticalField->get(x, y, z) = analyticalSol;
                                                       real_t currErr = (solutionField->get(x, y, z) - analyticalSol);
                                                       potentialErrorField ->get(x, y, z) = abs(currErr);
                                                       blockResult += currErr * currErr;
                                                       blockResult_analytical += analyticalSol * analyticalSol;
                                                       block_local_cells += 1;


            )
            cells +=  block_local_cells;
            error += blockResult;
            normL2 += blockResult_analytical;


        }
        mpi::allReduceInplace( error, mpi::SUM );
        mpi::allReduceInplace( normL2, mpi::SUM );
        mpi::allReduceInplace( cells, mpi::SUM );

        WALBERLA_LOG_INFO_ON_ROOT("number of cell is:"<< " " << cells);
        return (std::sqrt(error/(cells))); //(std::sqrt(error/(cells)))/(std::sqrt(normL2/(cells)));
    }
}

#endif //WALBERLA_ERRORNORMS_H
