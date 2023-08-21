#ifndef WALBERLA_POISSONSOLVER_H
#define WALBERLA_POISSONSOLVER_H

#include "pde/all.h"

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
                 real_t residualNormThreshold = real_c(1e-4), uint_t residualCheckFrequency = uint_t(100))
      : src_(src), dst_(dst), rhs_(rhs), blocks_(blocks), boundaryHandling_(boundaryHandling)
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

      // res norm

      residualNorm_ =
         make_shared< pde::ResidualNorm< Stencil_T > >(blocks_->getBlockStorage(), src_, rhs_, laplaceWeights_);

      // jacobi

      jacobiFixedSweep_ = make_shared< pde::JacobiFixedStencil< Stencil_T > >(src_, dst_, rhs_, laplaceWeights_);

      // use custom impl with damping or jacobi from waLBerla
      std::function< void(IBlock*) > jacSweep = {};
      if (solver == DAMPED_JACOBI)
      {
         jacSweep = [this](IBlock* block) { dampedJacobiSweep(block); };
      }
      else { jacSweep = *jacobiFixedSweep_; }

      jacobiIteration_ =
         std::make_unique< pde::JacobiIteration >(blocks_->getBlockStorage(), iterations, *commScheme_, jacSweep,
                                                  *residualNorm_, residualNormThreshold, residualCheckFrequency);

      jacobiIteration_->addBoundaryHandling(boundaryHandling_);

      // SOR

      real_t omega = real_t(2) / real_t(3);

      sorFixedSweep_ = make_shared< pde::SORFixedStencil< Stencil_T > >(blocks, src_, rhs_, laplaceWeights_, omega);

      sorIteration_ = std::make_unique< pde::RBGSIteration >(
         blocks_->getBlockStorage(), iterations, *commScheme_, sorFixedSweep_->getRedSweep(),
         sorFixedSweep_->getBlackSweep(), *residualNorm_, residualNormThreshold, residualCheckFrequency);

      sorIteration_->addBoundaryHandling(boundaryHandling);
   }

   // get approximate solution of electric potential
   void operator()()
   {
      if constexpr (solver != WALBERLA_SOR) { (*jacobiIteration_)(); }
      else { (*sorIteration_)(); }
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

   real_t initRes_;
};

} // namespace walberla

#endif // WALBERLA_POISSONSOLVER_H
