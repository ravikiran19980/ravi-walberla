//
// Created by dy94rovu on 3/5/26.
//

#include <blockforest/all.h>
#include <algorithm>
#include "WelfordVelocity.h"

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
        maskFieldId_(maskFieldId)
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
         //CellInterval localCi;
         //blocks->transformGlobalToBlockLocalCellInterval(localCi, *block, ci_);
         //fieldSize.intersect(localCi);

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

   void calculateDrivingForce()
   {
      // forcing term as in Malaspinas (2014) "Wall model for large-eddy simulation based on the lattice Boltzmann method"
      force_ = targetFrictionVelocity_ * targetFrictionVelocity_ / channelHalfWidth_;
               // + (targetBulkVelocity_ - bulkVelocity_) * targetBulkVelocity_ / channelHalfWidth_;
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
                       const Vector3<real_t>& domainSize)
      : blocks_(blocks), fieldId_(fieldId), wallAxis_(wallAxis), numHeights_(uint_c(domainSize[ wallAxis])), fieldSize_(Field_T::F_SIZE)
   {

      timeAveragedFluidProfile_.resize(fieldSize_*numHeights_, 0_r);
      timeAveragedFluidSquaredProfile_.resize(fieldSize_*fieldSize_*numHeights_, 0_r);
      fluidRMSProfile_.resize(fieldSize_*fieldSize_*numHeights_, 0_r);

      timeAveragedParticleProfile_.resize(fieldSize_*numHeights_, 0_r);
      timeAveragedParticleSquaredProfile_.resize(fieldSize_*fieldSize_*numHeights_, 0_r);
      particleRMSProfile_.resize(fieldSize_*fieldSize_*numHeights_, 0_r);

      timeStepCount_ = 0;  // Field_T::F_Size == 3
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
      std::vector< real_t > fluidSquaredProfile(fieldSize_*fieldSize_ * numHeights_, 0_r);
      std::vector< real_t > particleProfile(fieldSize_ * numHeights_, 0_r);
      std::vector< real_t > particleSquaredProfile(fieldSize_*fieldSize_ * numHeights_, 0_r);
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
            const uint_t idx = uint_c(globalCell.z());

            // fluid phase
            if (B < 1.0_r)
            {

               for (uint_t i = 0; i < fieldSize_; ++i)
               {
                  real_t fluidQuantity_i = Field->get(*cellIt, i);

                  fluidProfile[idx * fieldSize_ + i] += fluidQuantity_i*(1.0_r - B);
                  for (uint_t j = 0; j < Field_T::F_SIZE; ++j)
                  {
                     real_t fluidQuantity_j = Field->get(*cellIt, j);
                     fluidSquaredProfile[idx * fieldSize_* fieldSize_ + i*fieldSize_ + j] += (1.0_r - B)*fluidQuantity_i * fluidQuantity_j;
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
                     particleSquaredProfile[idx * fieldSize_* fieldSize_ + i*fieldSize_ + j] += B * particleQuantity_i * particleQuantity_j;
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
                  fluidSquaredProfile[h * fieldSize_* fieldSize_ + i * fieldSize_ + j] /= real_c(fluidCounts[h]);
                  timeAveragedFluidSquaredProfile_[h * fieldSize_* fieldSize_ + i * fieldSize_ + j] +=
                     (fluidSquaredProfile[h * fieldSize_* fieldSize_ + i * fieldSize_ + j] -
                      timeAveragedFluidSquaredProfile_[h * fieldSize_* fieldSize_ + i * fieldSize_ + j]) /
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
                  particleSquaredProfile[h * fieldSize_* fieldSize_ + i * fieldSize_ + j] /= real_c(particleCounts[h]);
                  timeAveragedParticleSquaredProfile_[h * fieldSize_* fieldSize_ + i * fieldSize_ + j] +=
                     (particleSquaredProfile[h * fieldSize_* fieldSize_ + i * fieldSize_ + j] -
                      timeAveragedParticleSquaredProfile_[h * fieldSize_* fieldSize_ + i * fieldSize_ + j]) /
                     real_c(timeStepCount_);
               }
            }
         }
      }
   }


   void computeFluidParticleRMS()
   {
      //fluidRMSProfile_.resize(fieldSize_*numHeights_, 0_r);
      //particleRMSProfile_.resize(fieldSize_*numHeights_, 0_r);

      for (uint_t h = 0; h < numHeights_; ++h)
      {
         for(uint_t i = 0; i < fieldSize_; ++i)
         {
            for(uint_t j = 0; j < fieldSize_; ++j){

               real_t fluidVariance = timeAveragedFluidSquaredProfile_[h*fieldSize_* fieldSize_ + i*fieldSize_ + j] - timeAveragedFluidProfile_[h*fieldSize_ + i] * timeAveragedFluidProfile_[h*fieldSize_ + j];
               fluidRMSProfile_[h*fieldSize_* fieldSize_ + i*fieldSize_ + j] = fluidVariance;//std::sqrt(abs(fluidVariance));

               real_t particleVariance = timeAveragedParticleSquaredProfile_[h*fieldSize_* fieldSize_ + i*fieldSize_ + j] - timeAveragedParticleProfile_[h*fieldSize_ + i] * timeAveragedParticleProfile_[h*fieldSize_ + j];
               particleRMSProfile_[h*fieldSize_* fieldSize_ + i*fieldSize_ + j] = std::sqrt(std::max(particleVariance, 0_r));
            }
         }
      }
   }


   const std::vector< real_t >& getFluidRMSProfile() const { return fluidRMSProfile_; }
   const std::vector< real_t >& getParticleRMSProfile() const { return particleRMSProfile_; }
   const std::vector< real_t >& getFluidAVGProfile() const { return timeAveragedFluidProfile_; }
   const std::vector< real_t >& getParticleAVGProfile() const { return timeAveragedParticleProfile_; }


 private:
   const std::shared_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID fieldId_;
   const uint_t wallAxis_;
   const uint_t numHeights_;
   const uint_t fieldSize_;


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
    * This initialisation is provided by Henrik Asmuth.
    */
   template<typename VelocityField_T>
   void setVelocityFieldsAsmuth( const std::weak_ptr<StructuredBlockStorage>& forest,
                                 const BlockDataID & velocityFieldId, const BlockDataID & meanVelocityFieldId,
                                 const real_t frictionVelocity, const uint_t channel_half_width,
                                 const real_t B, const real_t kappa, const real_t viscosity,
                                 const uint_t wallAxis, const uint_t flowAxis ) {

      auto blocks = forest.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks)

      const auto domainSize = blocks->getDomain().max();
      const auto delta = real_c(channel_half_width);
      const auto remAxis = 3 - wallAxis - flowAxis;

      for( auto block = blocks->begin(); block != blocks->end(); ++block ) {

         auto * velocityField = block->template getData<VelocityField_T>(velocityFieldId);
         WALBERLA_CHECK_NOT_NULLPTR(velocityField)

         auto * meanVelocityField = block->template getData<VelocityField_T>(meanVelocityFieldId);
         WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)

         const auto ci = velocityField->xyzSizeWithGhostLayer();
         for(auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt) {

            Cell globalCell(*cellIt);
            blocks->transformBlockLocalToGlobalCell(globalCell, *block);
            Vector3<real_t> cellCenter;
            blocks->getCellCenter(cellCenter, globalCell);

            const auto y = cellCenter[wallAxis];
            const auto rel_x = cellCenter[flowAxis] / domainSize[flowAxis];
            const auto rel_z = cellCenter[remAxis] / domainSize[remAxis];

            const real_t pos = std::max(delta - std::abs(y - delta - 1_r), 0.05_r);
            const auto rel_y = pos / delta;

            auto initialVel = frictionVelocity * (std::log(frictionVelocity * pos / viscosity) / kappa + B);

            Vector3<real_t> vel;
            vel[flowAxis] = initialVel;

            vel[remAxis] = 2_r * frictionVelocity / kappa * std::sin(math::pi * 16_r * rel_x) *
                           std::sin(math::pi * 8_r * rel_y) / (std::pow(rel_y, 2_r) + 1_r);

            vel[wallAxis] = 8_r * frictionVelocity / kappa *
                            (std::sin(math::pi * 8_r * rel_z) * std::sin(math::pi * 8_r * rel_y) +
                             std::sin(math::pi * 8_r * rel_x)) / (std::pow(0.5_r * delta - pos, 2_r) + 1_r);

            for(uint_t d = 0; d < 3; ++d) {
               velocityField->get(*cellIt, d) = vel[d];
               meanVelocityField->get(*cellIt, d) = vel[d];
            }

         }
      }

   } // function setVelocityFieldsHenrik

template<typename FieldMean_T, typename FieldSoS_T>
class reduceWelfordFields
{
 public:
   reduceWelfordFields(const std::shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID meanfieldId,const BlockDataID sosfieldId,
                              const uint_t wallAxis, const uint_t numHeights, WelfordVelocity welfordSweep)
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
      WelfordVelocity welfordSweep_;
};

}  // namespace walberla

