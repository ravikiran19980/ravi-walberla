//
// Created by dy94rovu on 3/5/26.
//

#include <blockforest/all.h>
#include <algorithm>
#include "WelfordVelocity.h"
#include "PIDController.h"
#include "utilities/bulkVelocityReductionKernel.h"

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
#include "gpu/Kernel.h"
#endif

/// BULK VELOCITY CALCULATION
namespace walberla
{
using namespace pystencils;

struct ForceCalculatorParameters
{
   Vector3<uint_t> domainSize;
   uint_t wallAxis;

   real_t channelHalfWidth;
   real_t targetBulkVelocity;
   real_t targetFrictionVelocity;
};


template< typename VelocityField_T, typename MaskField_T >
class ForceCalculator
{
 public:
   ForceCalculator(const std::weak_ptr< StructuredBlockStorage >& blocks, const BlockDataID meanVelocityId,
                   const BlockDataID maskFieldId,
                   const ForceCalculatorParameters& forceParams)
      : blocks_(blocks), meanVelocityId_(meanVelocityId), channelHalfWidth_(real_c(forceParams.channelHalfWidth)),
        targetBulkVelocity_(forceParams.targetBulkVelocity), targetFrictionVelocity_(forceParams.targetFrictionVelocity),
        maskFieldId_(maskFieldId), force_(targetFrictionVelocity_ * targetFrictionVelocity_ / channelHalfWidth_)
   {
      const auto& domainSize = forceParams.domainSize;

      Cell maxCell;
      maxCell[forceParams.wallAxis] = int_c(forceParams.channelHalfWidth) - 1;
      maxCell[flowDirection_]     = int_c(domainSize[flowDirection_]) - 1;
      const auto remainingIdx     = 3 - forceParams.wallAxis - flowDirection_;
      maxCell[remainingIdx]       = int_c(domainSize[remainingIdx]) - 1;
      ci_                         = CellInterval(Cell{}, maxCell);

   }

   real_t bulkVelocity() const { return bulkVelocity_; }
   void setBulkVelocity(const real_t bulkVelocity) { bulkVelocity_ = bulkVelocity; }

   void calculateBulkVelocity()
   {
      // reset bulk velocity
      bulkVelocity_ = 0_r;
      real_t numFluidCells = 0_r;

      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks)

      for (auto block = blocks->begin(); block != blocks->end(); ++block)
      {
         auto* meanVelocityField = block->template getData< VelocityField_T >(meanVelocityId_);
         auto* maskField = block->template getData< MaskField_T >(maskFieldId_);
         WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)
         WALBERLA_CHECK_NOT_NULLPTR(maskField)

         auto fieldSize = meanVelocityField->xyzSize();

         for (auto cellIt = fieldSize.begin(); cellIt != fieldSize.end(); ++cellIt)
         {
            const real_t localMean = meanVelocityField->get(*cellIt, 0);
            const real_t B = maskField->get(*cellIt);
            const real_t fluidWeight = (1-B);
            if (fluidWeight > 0)
            {
               bulkVelocity_ += localMean * fluidWeight;
               numFluidCells += fluidWeight;
            }
         }
      }

      mpi::allReduceInplace< real_t >(bulkVelocity_, mpi::SUM);
      mpi::allReduceInplace< real_t >(numFluidCells, mpi::SUM);


      bulkVelocity_ /= numFluidCells;

   }

   real_t calculateDrivingForce () const
   {
      // forcing term as in Malaspinas (2014) "Wall model for large-eddy simulation based on the lattice Boltzmann method"
      const auto force = targetFrictionVelocity_ * targetFrictionVelocity_ / channelHalfWidth_;
               // + (targetBulkVelocity_ - bulkVelocity_) * targetBulkVelocity_ / channelHalfWidth_;
      return force;
   }

   real_t getCurrentDrivingForce() const
   {
      return  force_;
   }

   real_t getBulkVelocity() const
   {
      return  bulkVelocity_;
   }

 private:
   const std::weak_ptr< StructuredBlockStorage > blocks_{};
   const BlockDataID meanVelocityId_{};
   const BlockDataID maskFieldId_{};

   const uint_t flowDirection_{};
   const real_t channelHalfWidth_{};
   const real_t targetBulkVelocity_{};
   const real_t targetFrictionVelocity_{};

   CellInterval ci_{};

   real_t bulkVelocity_{};
   real_t force_{};
   real_t oldforce_{};
};


template< typename VelocityField_T, typename MaskField_T >
class ForceAdjusterPID
{
 public:
   ForceAdjusterPID(
    const shared_ptr< StructuredBlockStorage >& blocks,
    const BlockDataID meanVelocityId,
    const BlockDataID maskFieldId,
    real_t targetVelocity,
    real_t externalForcing,
    real_t proportionalGain,
    real_t derivativeGain,
    real_t integralGain,
    real_t maxRamp,
    real_t minActuatingVariable,
    real_t maxActuatingVariable)
    : blocks_(blocks),
      meanVelocityId_(meanVelocityId),
      maskFieldId_(maskFieldId),
      currentExternalForcing_(externalForcing),
      pid_(targetVelocity, externalForcing, proportionalGain, derivativeGain,
           integralGain, maxRamp, minActuatingVariable, maxActuatingVariable)

   {
      WALBERLA_LOG_INFO_ON_ROOT("Creating PID controller with pg = " << pid_.getProportionalGain()
                                                                     << ", dg = " << pid_.getDerivateGain()
                                                                      << ", ig = " << pid_.getIntegralGain());
      auto blocks_ptr = blocks_.lock();
      auto domain = blocks_ptr->getDomain();
      auto max = domain.max();
      auto min = domain.min();

      const auto aabb = AABB(min, max);
      ci_ = blocks_ptr->getCellBBFromAABB(aabb);


   }

   void operator()(const real_t currentBulkVelocity)
   {
      // compute new forcing value on root (since flow rate only known on root)
      WALBERLA_ROOT_SECTION()
      {
         real_t newExternalForcing = pid_.update(currentBulkVelocity);
         currentExternalForcing_   = newExternalForcing;
      }

      // send updated external forcing to all other processes
      mpi::broadcastObject(currentExternalForcing_);
   }



   real_t bulkVelocity() const { return bulkVelocity_; }
   void setBulkVelocity(const real_t bulkVelocity) { bulkVelocity_ = bulkVelocity; }

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   void calculateBulkVelocity(const int32_t gpuBlockSize0, const int32_t gpuBlockSize1, const int32_t gpuBlockSize2)
   {
      real_t *d_velSum, *d_weightSum;
      WALBERLA_GPU_CHECK(gpuMalloc(&d_velSum, sizeof(real_t)));
      WALBERLA_GPU_CHECK(gpuMalloc(&d_weightSum, sizeof(real_t)));
      WALBERLA_GPU_CHECK(gpuMemset(d_velSum, 0, sizeof(real_t)));
      WALBERLA_GPU_CHECK(gpuMemset(d_weightSum, 0, sizeof(real_t)));

      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks)
      real_t h_velSum=0, h_weightSum=0;
      for (auto block = blocks->begin(); block != blocks->end(); ++block)
      {
         auto meanVelocityField = block-> getData< VelocityField_T >(meanVelocityId_);
         auto maskField         = block-> getData< MaskField_T >(maskFieldId_);

         int N         = meanVelocityField->xSize() * meanVelocityField->ySize() * meanVelocityField->zSize();


         // launched once per waLBerla block; atomicAdd accumulates
         // across all of them into the same two global scalars

         CellInterval localCi;
         blocks->transformGlobalToBlockLocalCellInterval(localCi, *block, ci_);
         localCi.intersect(meanVelocityField->xyzSize());
         if (localCi.empty())
            continue;

         WALBERLA_ASSERT_GREATER_EQUAL(localCi.xMin(), -int_c(gpuField->nrOfGhostLayers()));
         WALBERLA_ASSERT_GREATER_EQUAL(localCi.yMin(), -int_c(gpuField->nrOfGhostLayers()));
         WALBERLA_ASSERT_GREATER_EQUAL(localCi.zMin(), -int_c(gpuField->nrOfGhostLayers()));

         real_t *   _data_velocity_field = meanVelocityField->dataAt(localCi.xMin(), localCi.yMin(), localCi.zMin(), 0);
         real_t *   _data_fraction_field = maskField->dataAt(localCi.xMin(), localCi.yMin(), localCi.zMin(), 0);

         const int64_t _size_x = int64_t(int64_c(localCi.xSize()));
         const int64_t _size_y = int64_t(int64_c(localCi.ySize()));
         const int64_t _size_z = int64_t(int64_c(localCi.zSize()));

         const int64_t _stride_x = int64_t(meanVelocityField->xStride());
         const int64_t _stride_y = int64_t(meanVelocityField->yStride());
         const int64_t _stride_z = int64_t(meanVelocityField->zStride());
         const int64_t _stride_f = int64_t(1 * int64_t(meanVelocityField->fStride()));



         dim3 _block( std::min<uint_t>(gpuBlockSize0, _size_x),
                            std::min<uint_t>(gpuBlockSize1, _size_y),
                            std::min<uint_t>(gpuBlockSize2, _size_z) );

         dim3 _grid( (_size_x + _block.x - 1) / _block.x,
                     (_size_y + _block.y - 1) / _block.y,
                     (_size_z + _block.z - 1) / _block.z );


         size_t threadsPerBlock = _block.x * _block.y * _block.z;
         size_t numBlocks       = _grid.x  * _grid.y  * _grid.z;

         size_t _shmem = 2*threadsPerBlock * sizeof(real_t);


         auto bulkVelocityKernel = gpu::make_kernel(&bulkVelocityReductionKernel);
         bulkVelocityKernel.addParam(_data_velocity_field);
         bulkVelocityKernel.addParam(_data_fraction_field);
         bulkVelocityKernel.addParam(d_velSum);
         bulkVelocityKernel.addParam(d_weightSum);
         bulkVelocityKernel.addParam(_size_x);
         bulkVelocityKernel.addParam(_size_y);
         bulkVelocityKernel.addParam(_size_z);
         bulkVelocityKernel.addParam(_stride_x);
         bulkVelocityKernel.addParam(_stride_y);
         bulkVelocityKernel.addParam(_stride_z);
         bulkVelocityKernel.addParam(_stride_f);
         bulkVelocityKernel.addParam(threadsPerBlock);

         bulkVelocityKernel.configure(_grid,_block, _shmem);
         bulkVelocityKernel();
         WALBERLA_GPU_CHECK(gpuDeviceSynchronize());


         WALBERLA_GPU_CHECK(gpuMemcpy(&h_velSum, d_velSum, sizeof(real_t), gpuMemcpyDeviceToHost));
         WALBERLA_GPU_CHECK(gpuMemcpy(&h_weightSum, d_weightSum, sizeof(real_t), gpuMemcpyDeviceToHost));
         WALBERLA_GPU_CHECK(gpuFree(d_velSum));
         WALBERLA_GPU_CHECK(gpuFree(d_weightSum));

      }

      mpi::allReduceInplace< real_t >(h_velSum, mpi::SUM);
      mpi::allReduceInplace< real_t >(h_weightSum, mpi::SUM);
      bulkVelocity_ = h_velSum / h_weightSum;
      //WALBERLA_LOG_INFO_ON_ROOT("h_velSum: " << h_velSum);
      //WALBERLA_LOG_INFO_ON_ROOT("h_weightSum: " << h_weightSum);


   }

#else
   void calculateBulkVelocity()
   {
      // reset bulk velocity
      bulkVelocity_        = 0_r;
      real_t numFluidCells = 0_r;

      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks)

      for (auto block = blocks->begin(); block != blocks->end(); ++block)
      {
         auto* meanVelocityField = block->template getData< VelocityField_T >(meanVelocityId_);
         auto* maskField         = block->template getData< MaskField_T >(maskFieldId_);
         WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)
         WALBERLA_CHECK_NOT_NULLPTR(maskField)

         auto fieldSize = meanVelocityField->xyzSize();

         for (auto cellIt = fieldSize.begin(); cellIt != fieldSize.end(); ++cellIt)
         {
            const real_t localMean   = meanVelocityField->get(*cellIt, 0);
            const real_t B           = maskField->get(*cellIt);
            const real_t fluidWeight = (1 - B);
            if (fluidWeight > 0)
            {
               bulkVelocity_ += localMean * fluidWeight;
               numFluidCells += fluidWeight;
            }
         }
      }

      mpi::allReduceInplace< real_t >(bulkVelocity_, mpi::SUM);
      mpi::allReduceInplace< real_t >(numFluidCells, mpi::SUM);

      bulkVelocity_ /= numFluidCells;
   }

#endif




   real_t getCurrentDrivingForce() const
   {
      return  currentExternalForcing_;
   }


 private:
   const std::weak_ptr< StructuredBlockStorage > blocks_{};
   const BlockDataID meanVelocityId_{};
   const BlockDataID maskFieldId_{};

   const uint_t flowDirection_{};
   const real_t channelHalfWidth_{};
   const real_t targetBulkVelocity_{};
   const real_t targetFrictionVelocity_{};

   CellInterval ci_{};

   real_t bulkVelocity_{};
   real_t currentExternalForcing_{};
   real_t oldforce_{};

   PIDController pid_;
};




/// PLANE-AVERAGED PROFILES
/**
 * \brief Computes simple plane-averaged values along the wall normal axis.
 *
 * For scalar fields: Returns vector of averaged scalars at each height.
 * For vector fields: Returns vector of wall-normal component at each height.
 *
 * Averaging is done in streamwise and spanwise directions only.
 */
template< typename Field_T >
class PlaneAveragedProfiles
{
 public:

   PlaneAveragedProfiles(const std::shared_ptr< StructuredBlockStorage >& blocks,
                       const BlockDataID fieldId,
                       const uint_t wallAxis,
                       const Vector3<real_t>& domainSize,
                       bool startFromCheckpoint=false,
                       real_t averagingCount = 0,
                       std::vector<real_t> timeAveragedFluidProfile = {},
                       std::vector<real_t> timeAveragedFluidSquaredProfile = {},
                       std::vector<real_t> timeAveragedParticleProfile = {},
                       std::vector<real_t> timeAveragedParticleSquaredProfile = {}
                       )
      : blocks_(blocks), fieldId_(fieldId), wallAxis_(wallAxis), numHeights_(uint_c(domainSize[ wallAxis])), fieldSize_(Field_T::F_SIZE),fieldSizeSquared_(Field_T::F_SIZE*Field_T::F_SIZE)
   {
      fluidRMSProfile_.resize(fieldSizeSquared_*numHeights_, 0_r);
      particleRMSProfile_.resize(fieldSizeSquared_*numHeights_, 0_r);

      if (startFromCheckpoint == false)
      {
         timeAveragedFluidProfile_.resize(fieldSize_*numHeights_, 0_r);
         timeAveragedFluidSquaredProfile_.resize(fieldSizeSquared_*numHeights_, 0_r);

         timeAveragedParticleProfile_.resize(fieldSize_*numHeights_, 0_r);
         timeAveragedParticleSquaredProfile_.resize(fieldSizeSquared_*numHeights_, 0_r);
         timeStepCount_ = averagingCount;
      }

      else
      {
         timeAveragedFluidProfile_ = std::move(timeAveragedFluidProfile);
         timeAveragedFluidSquaredProfile_ = std::move(timeAveragedFluidSquaredProfile);

         timeAveragedParticleProfile_ = std::move(timeAveragedParticleProfile);
         timeAveragedParticleSquaredProfile_ = std::move(timeAveragedParticleSquaredProfile);
         timeStepCount_ = averagingCount;
      }
   }

   /**
    * \brief Compute fluid and particle velocity fluctuation RMS values separately
    *
    * Decomposes velocity into fluid and particle components using B field (solid volume fraction):
    * fluidVelocity = velocity * (1 - B)
    * particleVelocity = velocity * B
    */

   void computeFluidParticleAveragedVectors(const BlockDataID BFieldId)
   {
      // Initialize storage

      std::vector< real_t > fluidProfile(fieldSize_ * numHeights_, 0_r);
      std::vector< real_t > fluidSquaredProfile(fieldSizeSquared_ * numHeights_, 0_r);
      std::vector< real_t > particleProfile(fieldSize_ * numHeights_, 0_r);
      std::vector< real_t > particleSquaredProfile(fieldSizeSquared_ * numHeights_, 0_r);
      std::vector< uint_t > fluidCounts(numHeights_, 0);
      std::vector< uint_t > particleCounts(numHeights_, 0);

      // Accumulate fluid and particle velocity components
      for (auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt)
      {
         auto* Field  = blockIt->template getData< Field_T >(fieldId_);
         auto* BField = blockIt->template getData< GhostLayerField< real_t, 1 > >(BFieldId);
         WALBERLA_CHECK_NOT_NULLPTR(Field);
         WALBERLA_CHECK_NOT_NULLPTR(BField);

         const auto ci = Field->xyzSize();

         for (auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt)
         {
            real_t B = BField->get(*cellIt);
            Cell globalCell(*cellIt);
            blocks_->transformBlockLocalToGlobalCell(globalCell, *blockIt);
            const uint_t idx = uint_c(globalCell.y());

            // fluid phase
            if (B < 1.0_r)
            {

               for (uint_t i = 0; i < fieldSize_; ++i)
               {
                  real_t fluidQuantity_i = Field->get(*cellIt, i);

                  fluidProfile[idx * fieldSize_ + i] += fluidQuantity_i*(1.0_r - B);

                  for (uint_t j = 0; j < fieldSize_; ++j)
                  {
                     real_t fluidQuantity_j = Field->get(*cellIt, j);
                     fluidSquaredProfile[idx * fieldSizeSquared_ + i*fieldSize_ + j] += (1.0_r - B)*fluidQuantity_i * fluidQuantity_j;
                  }
               }
               fluidCounts[idx]++;
            }

            // Particle phase
            if (B > 0_r)
            {

               for (uint_t i = 0; i < fieldSize_; ++i)
               {
                  real_t particleQuantity_i = Field->get(*cellIt, i);
                  particleProfile[idx * fieldSize_ + i] += particleQuantity_i * B;

                  for (uint_t j = 0; j < fieldSize_; ++j)
                  {
                     real_t particleQuantity_j = Field->get(*cellIt, j);
                     particleSquaredProfile[idx * fieldSizeSquared_ + i*fieldSize_ + j] += B * particleQuantity_i * particleQuantity_j;
                  }
               }
               particleCounts[idx]++;
            }

         }
      }

      // MPI reduction
      mpi::allReduceInplace(fluidProfile, mpi::SUM);
      mpi::allReduceInplace(fluidSquaredProfile, mpi::SUM);
      mpi::allReduceInplace(particleProfile, mpi::SUM);
      mpi::allReduceInplace(particleSquaredProfile, mpi::SUM);
      mpi::allReduceInplace(fluidCounts, mpi::SUM);
      mpi::allReduceInplace(particleCounts, mpi::SUM);

      // Average and compute time-averaged RMS
      timeStepCount_++;

      for (uint_t h = 0; h < numHeights_; ++h)
      {
         if (fluidCounts[h] > 0)
         {
            for (uint_t i = 0; i < fieldSize_ ; ++i)
            {
               fluidProfile[h * fieldSize_ + i] /= real_c(fluidCounts[h]);
               timeAveragedFluidProfile_[h * fieldSize_ + i] +=
                  (fluidProfile[h * fieldSize_ + i] - timeAveragedFluidProfile_[h * fieldSize_ + i]) /
                  real_c(timeStepCount_);

               for (uint_t j=0; j < fieldSize_; j++)
               {
                  fluidSquaredProfile[h * fieldSizeSquared_ + i * fieldSize_ + j] /= real_c(fluidCounts[h]);
                  timeAveragedFluidSquaredProfile_[h * fieldSizeSquared_ + i * fieldSize_ + j] +=
                     (fluidSquaredProfile[h * fieldSizeSquared_ + i * fieldSize_ + j] -
                      timeAveragedFluidSquaredProfile_[h * fieldSizeSquared_ + i * fieldSize_ + j]) /
                     real_c(timeStepCount_);
               }


            }
         }

         if (particleCounts[h] > 0)
         {
            for (uint_t i = 0; i < fieldSize_; ++i)
            {
               particleProfile[h * fieldSize_ + i] /= real_c(particleCounts[h]);
               timeAveragedParticleProfile_[h * fieldSize_ + i] +=
                  (particleProfile[h * fieldSize_ + i] - timeAveragedParticleProfile_[h * fieldSize_ + i]) /
                  real_c(timeStepCount_);

               for (uint_t j = 0; j < fieldSize_; j++)
               {
                  particleSquaredProfile[h * fieldSizeSquared_ + i * fieldSize_ + j] /= real_c(particleCounts[h]);
                  timeAveragedParticleSquaredProfile_[h * fieldSizeSquared_ + i * fieldSize_ + j] +=
                     (particleSquaredProfile[h * fieldSizeSquared_ + i * fieldSize_ + j] -
                      timeAveragedParticleSquaredProfile_[h * fieldSizeSquared_ + i * fieldSize_ + j]) /
                     real_c(timeStepCount_);
               }
            }
         }
      }
   }


   void computeFluidParticleRMS()
   {

      for (uint_t h = 0; h < numHeights_; ++h)
      {
         for(uint_t i = 0; i < fieldSize_; ++i)
         {
            for(uint_t j = 0; j < fieldSize_; ++j){

               real_t fluidVariance = timeAveragedFluidSquaredProfile_[h*fieldSizeSquared_ + i*fieldSize_ + j] - timeAveragedFluidProfile_[h*fieldSize_ + i] * timeAveragedFluidProfile_[h*fieldSize_ + j];
               fluidRMSProfile_[h*fieldSizeSquared_ + i*fieldSize_ + j] = fluidVariance;

               real_t particleVariance = timeAveragedParticleSquaredProfile_[h*fieldSizeSquared_ + i*fieldSize_ + j] - timeAveragedParticleProfile_[h*fieldSize_ + i] * timeAveragedParticleProfile_[h*fieldSize_ + j];
               particleRMSProfile_[h*fieldSizeSquared_ + i*fieldSize_ + j] = particleVariance;
            }
         }
      }
   }


   const std::vector< real_t >& getFluidRMSProfile() const { return fluidRMSProfile_; }
   const std::vector< real_t >& getParticleRMSProfile() const { return particleRMSProfile_; }
   const std::vector< real_t >& getFluidAVGProfile() const { return timeAveragedFluidProfile_; }
   const std::vector< real_t >& getParticleAVGProfile() const { return timeAveragedParticleProfile_; }
   const std::vector< real_t >& getFluidAVGSquaredProfile() const { return timeAveragedFluidSquaredProfile_; }
   const std::vector< real_t >& getParticleAVGSquaredProfile() const { return timeAveragedParticleSquaredProfile_; }
   const uint_t getTimeStepCount() const { return timeStepCount_; }


 private:
   const std::shared_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID fieldId_;
   const uint_t wallAxis_;
   const uint_t numHeights_;
   const uint_t fieldSize_;
   const uint_t fieldSizeSquared_;


   std::vector< real_t > timeAveragedFluidProfile_;
   std::vector< real_t > timeAveragedFluidSquaredProfile_;
   std::vector< real_t > fluidRMSProfile_;

   std::vector< real_t > timeAveragedParticleProfile_;
   std::vector< real_t > timeAveragedParticleSquaredProfile_;
   std::vector< real_t > particleRMSProfile_;

   uint_t timeStepCount_;
};

/*
 * Initialises the velocity field with a logarithmic profile and sinusoidal perturbations to trigger turbulence.
 * This initialisation is a modified form of Henrik Asmuth.
 */

template< typename VelocityField_T >
void setVelocityFieldsAsmuth(const std::weak_ptr< StructuredBlockStorage >& forest, const BlockDataID& velocityFieldId,
                             const BlockDataID& meanVelocityFieldId, const real_t frictionVelocity,
                             const uint_t channel_half_width, const real_t B, const real_t kappa,
                             const real_t viscosity, const uint_t wallAxis, const uint_t flowAxis)
{
   auto blocks = forest.lock();
   WALBERLA_CHECK_NOT_NULLPTR(blocks)

   const auto domainSize = blocks->getDomain().max();
   const auto delta      = real_c(channel_half_width);
   const auto remAxis    = 3 - wallAxis - flowAxis;

   // ─────────────────────────────────────────────
   // Physical targets in wall units
   // Derived from your experiments across domains:
   //   sweet spot: λ_x⁺ ≈ 150-200, λ_z⁺ ≈ 100
   // ─────────────────────────────────────────────
   const real_t target_lambda_x_plus = 128 / (domainSize[codegen::flow_axis]) * 150.0_r; // middle of sweet spot
   const real_t target_lambda_z_plus = 64 / (domainSize[codegen::remaining_axis]) * 150.0_r;

   // Wall unit conversion factor
   const real_t utau_over_nu = frictionVelocity / viscosity;

   // Target wavelengths in lattice cells
   const real_t lambda_x = target_lambda_x_plus / utau_over_nu;
   const real_t lambda_z = target_lambda_z_plus / utau_over_nu;

   // Factors for sin()  →  factor = 2 * L / lambda
   const uint_t factor_x = uint_c(2.0_r * real_c(domainSize[flowAxis]) / lambda_x);
   const uint_t factor_z = uint_c(2.0_r * real_c(domainSize[remAxis]) / lambda_z);

   // ─────────────────────────────────────────────
   // Sanity check log - verify these look reasonable
   // ─────────────────────────────────────────────
   WALBERLA_LOG_INFO_ON_ROOT("=== Generalized Initialization ===");
   WALBERLA_LOG_INFO_ON_ROOT("u_tau/nu          = " << utau_over_nu);
   WALBERLA_LOG_INFO_ON_ROOT("lambda_x (cells)  = " << lambda_x << "  lambda_x+= " << target_lambda_x_plus);
   WALBERLA_LOG_INFO_ON_ROOT("lambda_z (cells)  = " << lambda_z << "  lambda_z+= " << target_lambda_z_plus);
   WALBERLA_LOG_INFO_ON_ROOT("factor_x          = " << factor_x);
   WALBERLA_LOG_INFO_ON_ROOT("factor_z          = " << factor_z);

   // Warn if less than half a wave fits - perturbation will be very weak
   if (factor_x < 1.0_r)
      WALBERLA_LOG_WARNING_ON_ROOT("factor_x < 1: domain too short in x "
                                   "for target lambda_x+. Consider reducing "
                                   "target_lambda_x_plus.");
   if (factor_z < 1.0_r)
      WALBERLA_LOG_WARNING_ON_ROOT("factor_z < 1: domain too short in z "
                                   "for target lambda_z+. Consider reducing "
                                   "target_lambda_z_plus.");

   for (auto block = blocks->begin(); block != blocks->end(); ++block)
   {
      auto* velocityField     = block->template getData< VelocityField_T >(velocityFieldId);
      auto* meanVelocityField = block->template getData< VelocityField_T >(meanVelocityFieldId);
      WALBERLA_CHECK_NOT_NULLPTR(velocityField)
      WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)

      const auto ci = velocityField->xyzSizeWithGhostLayer();
      for (auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt)
      {
         Cell globalCell(*cellIt);
         blocks->transformBlockLocalToGlobalCell(globalCell, *block);
         Vector3< real_t > cellCenter;
         blocks->getCellCenter(cellCenter, globalCell);

         // Wall normal position
         const auto y     = cellCenter[wallAxis];
         const auto pos   = std::max(delta - std::abs(y - delta - 1_r), 0.05_r);
         const auto rel_y = pos / delta;

         // Normalized coords for x and z
         const auto rel_x = cellCenter[flowAxis] / real_c(domainSize[flowAxis]);
         const auto rel_z = cellCenter[remAxis] / real_c(domainSize[remAxis]);

         // Streamwise: log-law mean profile
         const real_t initialVel = frictionVelocity * (std::log(frictionVelocity * pos / viscosity) / kappa + B);

         Vector3< real_t > vel;
         vel[flowAxis] = initialVel;

         // Spanwise perturbation
         // factor_x anchored to λ_x+ sweet spot
         vel[remAxis] = 2_r * frictionVelocity / kappa * std::sin(math::pi * 16 * rel_x) *
                        std::sin(math::pi * 8_r * rel_y) // wall-normal envelope
                        / (std::pow(rel_y, 2_r) + 1_r);

         // Wall-normal perturbation
         // factor_z anchored to λ_z+ = 100
         // factor_x same physical scale as spanwise
         vel[wallAxis] = 8_r * frictionVelocity / kappa *
                         (std::sin(math::pi * (factor_z) *rel_z) * std::sin(math::pi * 16_r * rel_y) +
                          std::sin(math::pi * (factor_x) *rel_x)) /
                         (std::pow(rel_y, 2_r) + 1_r);
         /// (std::pow(0.5_r * delta - pos, 2_r) + 1_r);

         for (uint_t d = 0; d < 3; ++d)
         {
            velocityField->get(*cellIt, d)     = vel[d];
            meanVelocityField->get(*cellIt, d) = vel[d];
         }
      }
   }
}

template<typename FieldMean_T, typename FieldSoS_T, typename WelfordSweep_T>
class reduceWelfordFields
{
 public:
   reduceWelfordFields(const std::shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID meanfieldId,const BlockDataID sosfieldId,
                              const uint_t wallAxis, const uint_t numHeights, WelfordSweep_T welfordSweep)
      : blocks_(blocks), meanfieldId_(meanfieldId),sosfieldId_(sosfieldId), wallAxis_(wallAxis), numHeights_(numHeights), fieldSize_(uint_c(FieldMean_T::F_SIZE)),
        planeMeans_(numHeights_ * fieldSize_, real_t(0)),planeSOSMeans_(numHeights_ * fieldSize_*fieldSize_, real_t(0)), planeCounts_(numHeights_, uint_t(0)), welfordSweep_(welfordSweep)
   {}

   void operator()()
   {
      std::fill(planeMeans_.begin(), planeMeans_.end(), real_t(0));
      std::fill(planeSOSMeans_.begin(), planeSOSMeans_.end(), real_t(0));
      std::fill(planeCounts_.begin(), planeCounts_.end(), uint_t(0));

      for (auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt)
      {
         auto* meanfield = blockIt->template getData< FieldMean_T >(meanfieldId_);
         auto* sosfield =  blockIt->template getData< FieldSoS_T >(sosfieldId_);
         WALBERLA_CHECK_NOT_NULLPTR(meanfield)

         const auto ci = meanfield->xyzSize();

         for (auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt)
         {
            Cell globalCell(*cellIt);
            blocks_->transformBlockLocalToGlobalCell(globalCell, *blockIt);
            const uint_t heightIdx = uint_c(globalCell[wallAxis_]);
            WALBERLA_CHECK_LESS(heightIdx, numHeights_)

            for (uint_t i = 0; i < fieldSize_; ++i)
            {
               planeMeans_[heightIdx * fieldSize_ + i] += meanfield->get(*cellIt, i);
               for (uint_t j = 0; j < fieldSize_; ++j)
               {
                  //WALBERLA_LOG_INFO_ON_ROOT("sos value is  " << sosfield->get(*cellIt,i*fieldSize_ + j));
                  planeSOSMeans_[heightIdx * fieldSize_* fieldSize_ + i*fieldSize_ + j] += sosfield->get(*cellIt,i*fieldSize_ + j)/welfordSweep_.getCounter();
               }
            }
            ++planeCounts_[heightIdx];
         }
      }

      mpi::allReduceInplace(planeMeans_, mpi::SUM);
      mpi::allReduceInplace(planeSOSMeans_, mpi::SUM);
      mpi::allReduceInplace(planeCounts_, mpi::SUM);

      for (uint_t h = 0; h < numHeights_; ++h)
      {
         if (planeCounts_[h] == 0)
         {
            continue;
         }

         for (uint_t i = 0; i < fieldSize_; ++i)
         {
            planeMeans_[h * fieldSize_ + i] /= real_c(planeCounts_[h]);
            for (uint_t j = 0; j < fieldSize_; ++j)
            {
               planeSOSMeans_[h * fieldSize_* fieldSize_ + i*fieldSize_ + j] /= real_c(planeCounts_[h]);
            }
         }
      }
   }

   const std::vector< real_t >& getPlaneMeans() const { return planeMeans_; }
   const std::vector< real_t >& getPlaneSoSMeans() const { return planeSOSMeans_; }
   uint_t getNumHeights() const { return numHeights_; }
   uint_t getFieldSize() const { return fieldSize_; }

 private:
   const std::shared_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID meanfieldId_;
   const BlockDataID sosfieldId_;
   const uint_t wallAxis_;
   const uint_t numHeights_;
   const uint_t fieldSize_;

   std::vector< real_t > planeMeans_;
   std::vector< real_t > planeSOSMeans_;
   std::vector< uint_t > planeCounts_;
   WelfordSweep_T welfordSweep_;
};

template<typename FieldMean_T>
const std::vector< real_t > computeViscousStress(const std::shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID & meanVelocityFieldId ,const Vector3<real_t>& domainSize)
{
   std::vector< real_t > viscousStress(uint_c(domainSize[codegen::wall_axis]), 0_r);
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      auto& block = *blockIt;
      auto* meanfield = blockIt->template getData< FieldMean_T >(meanVelocityFieldId);
      WALBERLA_FOR_ALL_CELLS_XYZ(

         meanfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));

         const uint_t j = uint_c(cell.y()); if (j > 0 && j < uint_c(domainSize[codegen::wall_axis]) - 1) {
            if (y == 0)
            {
               const real_t U0 = meanfield->get(x, y, z, 0);
               const real_t U1 = meanfield->get(x, y+1, z, 0);
               const real_t U2 = meanfield->get(x, y+2, z,0);
               viscousStress[j] += (-3.0 *U0 + 4.0 * U1 - U2) / (2.0);
            }
            else if (y == cell_idx_c(blocks->getNumberOfYCells(block) - 1))
            {
               const real_t U0 = meanfield->get(x, y, z,0);
               const real_t U1 = meanfield->get(x, y-1, z,0);
               const real_t U2 = meanfield->get(x, y-2, z,0);
               viscousStress[j] += (3.0 * U0 - 4.0 * U1 + U2) / (2.0);
            }
            else
            {
               const real_t U0 = meanfield->get(x, y-1, z,0);
               const real_t U1 = meanfield->get(x, y+1, z,0);
               viscousStress[j] += (U1 - U0) / (2.0);
               ;
            }
         } else {
            // wall flux (one-sided derivative), average over x,z on the two extreme planes

            if (j == 0)
            {
               const real_t U0 = meanfield->get(x, y, z,0);
               const real_t U1 = meanfield->get(x, y+1, z,0);
               const real_t U2 = meanfield->get(x, y+2, z,0);
               viscousStress[j] += (-3.0 * U0 + 4.0 * U1 - U2) / (2.0);
               ;
            }
            if (j == uint_c(domainSize[codegen::wall_axis]) - 1)
            {
               const real_t U0 = meanfield->get(x, y, z,0);
               const real_t U1 = meanfield->get(x, y-1, z,0);
               const real_t U2 = meanfield->get(x, y-2, z,0);
               viscousStress[j] += (3.0 * U0 - 4.0 * U1 + U2) / (2.0);
            }
         }

      )
   }
   WALBERLA_MPI_SECTION()
   {
      mpi::allReduceInplace(viscousStress, mpi::SUM);

   }
   for (size_t k = 0; k < viscousStress.size(); ++k)
   {
      viscousStress[k] /= domainSize[codegen::flow_axis] * domainSize[codegen::remaining_axis];
   }

   return viscousStress;
}

}  // namespace walberla
