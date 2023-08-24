#ifndef WALBERLA_POISSONSOLVER_H
#define WALBERLA_POISSONSOLVER_H

#include "pde/all.h"

#include <algorithm>

#include "CustomBoundary.h"
#include "DirichletDomainBoundary.h"
#include "Neumann.h"

enum Enum { WALBERLA_JACOBI, WALBERLA_SOR, DAMPED_JACOBI };

namespace walberla
{
// typedefs
using Stencil_T = stencil::D3Q7;
typedef GhostLayerField< real_t, 1 > ScalarField_T;

template< Enum solver >
class PoissonSolver
{
 public:
   void dampedJacobiSweep(IBlock* const block)
   {
      ScalarField_T* srcField = block->getData< ScalarField_T >(src_);
      ScalarField_T* dstField = block->getData< ScalarField_T >(dst_);
      ScalarField_T* rhsField = block->getData< ScalarField_T >(rhs_);

      const real_t omega          = real_c(0.6);
      const real_t invLaplaceDiag = real_t(1) / laplaceWeights_[Stencil_T::idx[stencil::C]];

      WALBERLA_ASSERT_GREATER_EQUAL(srcField->nrOfGhostLayers(), 1);

      WALBERLA_FOR_ALL_CELLS_XYZ(srcField, real_t stencilTimesSrc = real_c(0);
                                 for (auto dir = Stencil_T::begin(); dir != Stencil_T::end(); ++dir) stencilTimesSrc +=
                                 laplaceWeights_[dir.toIdx()] * srcField->getNeighbor(x, y, z, *dir);

                                 dstField->get(x, y, z) =
                                    srcField->get(x, y, z) +
                                    omega * invLaplaceDiag * (rhsField->get(x, y, z) - stencilTimesSrc);)

      srcField->swapDataPointers(dstField);
   }

   PoissonSolver(const BlockDataID& src, const BlockDataID& dst, const BlockDataID& rhs,
                 const std::shared_ptr< StructuredBlockForest >& blocks,
                 const std::function< void() >& boundaryHandling, uint_t iterations = uint_t(1000),
                 bool useAbsResNormThres = false, real_t absResNormThres = real_c(1e-10),
                 real_t relResNormThres = real_c(1e-4), uint_t resCheckFreq = uint_t(100))
      : src_(src), dst_(dst), rhs_(rhs), blocks_(blocks), boundaryHandling_(boundaryHandling),
        useRelativeResidualThreshold_(!useAbsResNormThres), relativeResidualReductionFactor_(relResNormThres)
   {
      // stencil weights
      laplaceWeights_                             = std::vector< real_t >(Stencil_T::Size, real_c(0));
      laplaceWeights_[Stencil_T::idx[stencil::C]] = real_t(2) / (blocks_->dx() * blocks_->dx()) +
                                                    real_t(2) / (blocks_->dy() * blocks_->dy()) +
                                                    real_t(2) / (blocks_->dz() * blocks_->dz());
      laplaceWeights_[Stencil_T::idx[stencil::T]] = real_t(-1) / (blocks_->dz() * blocks_->dz());
      laplaceWeights_[Stencil_T::idx[stencil::B]] = real_t(-1) / (blocks_->dz() * blocks_->dz());
      laplaceWeights_[Stencil_T::idx[stencil::N]] = real_t(-1) / (blocks_->dy() * blocks_->dy());
      laplaceWeights_[Stencil_T::idx[stencil::S]] = real_t(-1) / (blocks_->dy() * blocks_->dy());
      laplaceWeights_[Stencil_T::idx[stencil::E]] = real_t(-1) / (blocks_->dx() * blocks_->dx());
      laplaceWeights_[Stencil_T::idx[stencil::W]] = real_t(-1) / (blocks_->dx() * blocks_->dx());

      // communication

      commScheme_ = make_shared< blockforest::communication::UniformBufferedScheme< Stencil_T > >(blocks_);
      commScheme_->addPackInfo(make_shared< field::communication::PackInfo< ScalarField_T > >(src_));
      commScheme_->addPackInfo(make_shared< field::communication::PackInfo< ScalarField_T > >(rhs_));

      // res norm computation callback

      residualNorm_ =
         make_shared< pde::ResidualNorm< Stencil_T > >(blocks_->getBlockStorage(), src_, rhs_, laplaceWeights_);

      // handling for absolute/relative residual thresholds
      // if absolute res thres is used -> simply propagate ctor values for absolute res checks to Jacobi iteration
      // else
      //  -> only execute "resCheckFreq" iterations at once and then execute logic for relative exit crit
      //  -> repeat "numExecutions" times to perform the same amount of total iterations
      WALBERLA_ASSERT(iterations % resCheckFreq == 0,
                      "Number of iterations should be divisible by residual check frequency!")

      if (useAbsResNormThres)
      {
         residualNormThreshold_  = absResNormThres;
         residualCheckFrequency_ = resCheckFreq;
         numIterPerExecution_    = iterations;
         numExecutions_          = uint_c(1);
      }
      else
      {
         residualNormThreshold_  = real_c(0);
         residualCheckFrequency_ = uint_c(0);
         numIterPerExecution_    = resCheckFreq;
         numExecutions_          = iterations / resCheckFreq;
      }

      // jacobi

      jacobiFixedSweep_ = make_shared< pde::JacobiFixedStencil< Stencil_T > >(src_, dst_, rhs_, laplaceWeights_);

      // use custom impl with damping or jacobi from waLBerla
      std::function< void(IBlock*) > jacSweep = {};
      if (solver == DAMPED_JACOBI)
      {
         jacSweep = [this](IBlock* block) { dampedJacobiSweep(block); };
      }
      else {
         jacSweep = *jacobiFixedSweep_;
      }

      jacobiIteration_ = std::make_unique< pde::JacobiIteration >(blocks_->getBlockStorage(), numIterPerExecution_,
                                                                  *commScheme_, jacSweep, *residualNorm_,
                                                                  residualNormThreshold_, residualCheckFrequency_);

      jacobiIteration_->addBoundaryHandling(boundaryHandling_);

      // SOR

      real_t omega = real_t(2) / real_t(3);

      sorFixedSweep_ = make_shared< pde::SORFixedStencil< Stencil_T > >(blocks, src_, rhs_, laplaceWeights_, omega);

      sorIteration_ = std::make_unique< pde::RBGSIteration >(
         blocks_->getBlockStorage(), numIterPerExecution_, *commScheme_, sorFixedSweep_->getRedSweep(),
         sorFixedSweep_->getBlackSweep(), *residualNorm_, residualNormThreshold_, residualCheckFrequency_);

      sorIteration_->addBoundaryHandling(boundaryHandling);
   }

   // get approximate solution of electric potential
   void operator()()
   {
      for (uint_t executions = 0; executions < numExecutions_; ++executions)
      {
         // execute solver...
         if constexpr (solver != WALBERLA_SOR)
            (*jacobiIteration_)();
         else
            (*sorIteration_)();

         // .. and check if (relative) res threshold was reached
         if (useRelativeResidualThreshold_)
         {
            auto curRes = (*residualNorm_)();

            WALBERLA_LOG_INFO_ON_ROOT("Residual norm after " << executions * numIterPerExecution_
                                                             << " Jacobi iterations: " << curRes);

            if (curRes <= relativeResidualReductionFactor_ * initRes_)
            {
               WALBERLA_LOG_INFO_ON_ROOT("Aborting Jacobi iteration (residual norm threshold reached):"
                                         "\n  residual norm thres: "
                                         << relativeResidualReductionFactor_ * initRes_
                                         << "\n  residual norm:           " << curRes);

               break;
            }
         }
      }
   }

   void computeInitialResidual()
   {
      // compute initial residual and print
      boundaryHandling_();
      (*commScheme_)();
      initRes_ = (*residualNorm_)();

      WALBERLA_LOG_INFO_ON_ROOT("Initial residual = " << initRes_);
   }

 private:
   BlockDataID src_;
   BlockDataID dst_;
   BlockDataID rhs_;
   std::vector< real_t > laplaceWeights_;
   std::shared_ptr< StructuredBlockForest > blocks_;
   std::shared_ptr< blockforest::communication::UniformBufferedScheme< Stencil_T > > commScheme_;

   std::function< void() > boundaryHandling_;

   std::shared_ptr< pde::ResidualNorm< Stencil_T > > residualNorm_;

   std::shared_ptr< pde::JacobiFixedStencil< Stencil_T > > jacobiFixedSweep_;
   std::unique_ptr< pde::JacobiIteration > jacobiIteration_;

   std::shared_ptr< pde::SORFixedStencil< Stencil_T > > sorFixedSweep_;
   std::unique_ptr< pde::RBGSIteration > sorIteration_;

   // general residual variables
   real_t residualNormThreshold_;
   uint_t residualCheckFrequency_;
   uint_t numExecutions_;
   uint_t numIterPerExecution_;

   // variables for relative residual threshold
   real_t initRes_;
   bool useRelativeResidualThreshold_;
   real_t relativeResidualReductionFactor_;
};

} // namespace walberla

#endif // WALBERLA_POISSONSOLVER_H
