//
// Created by dy94rovu on 3/5/26.
//

#include <blockforest/all.h>

/// BULK VELOCITY CALCULATION
namespace walberla
{

struct ForceCalculatorParameters
{
   Vector3<uint_t> domainSize;
   uint_t wallAxis;

   real_t channelHalfWidth;
   real_t targetBulkVelocity;
   real_t targetFrictionVelocity;
};


template< typename VelocityField_T >
class ForceCalculator
{
 public:
   ForceCalculator(const std::weak_ptr< StructuredBlockStorage >& blocks, const BlockDataID meanVelocityId,
                   const ForceCalculatorParameters& forceParams)
      : blocks_(blocks), meanVelocityId_(meanVelocityId), channelHalfWidth_(real_c(forceParams.channelHalfWidth)),
        targetBulkVelocity_(forceParams.targetBulkVelocity), targetFrictionVelocity_(forceParams.targetFrictionVelocity)
   {
      const auto& domainSize = forceParams.domainSize;

      Cell maxCell;
      maxCell[forceParams.wallAxis] = int_c(forceParams.channelHalfWidth) - 1;
      maxCell[flowDirection_]     = int_c(domainSize[flowDirection_]) - 1;
      const auto remainingIdx     = 3 - forceParams.wallAxis - flowDirection_;
      maxCell[remainingIdx]       = int_c(domainSize[remainingIdx]) - 1;
      ci_                         = CellInterval(Cell{}, maxCell);

      numCells_ = real_c(forceParams.channelHalfWidth * domainSize[flowDirection_] * domainSize[remainingIdx]);
   }

   real_t bulkVelocity() const { return bulkVelocity_; }
   void setBulkVelocity(const real_t bulkVelocity) { bulkVelocity_ = bulkVelocity; }

   void calculateBulkVelocity()
   {
      // reset bulk velocity
      bulkVelocity_ = 0_r;

      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks)

      for (auto block = blocks->begin(); block != blocks->end(); ++block)
      {
         auto* meanVelocityField = block->template getData< VelocityField_T >(meanVelocityId_);
         WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)

         auto fieldSize = meanVelocityField->xyzSize();
         CellInterval localCi;
         blocks->transformGlobalToBlockLocalCellInterval(localCi, *block, ci_);
         fieldSize.intersect(localCi);

         auto* slicedField = meanVelocityField->getSlicedField(fieldSize);
         WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)

         for (auto fieldIt = slicedField->beginXYZ(); fieldIt != slicedField->end(); ++fieldIt)
         {
            const auto localMean = fieldIt[flowDirection_];
            bulkVelocity_ += localMean;
         }
      }

      mpi::allReduceInplace< real_t >(bulkVelocity_, mpi::SUM);
      bulkVelocity_ /= numCells_;
   }

   void calculateDrivingForce()
   {
      force_ = oldforce_ + (targetBulkVelocity_ - bulkVelocity_) * targetBulkVelocity_ / channelHalfWidth_;
      oldforce_ = force_;
   }

   real_t getCurrentDrivingForce() const
   {
      return  force_;
   }

 private:
   const std::weak_ptr< StructuredBlockStorage > blocks_{};
   const BlockDataID meanVelocityId_{};

   const uint_t flowDirection_{};
   const real_t channelHalfWidth_{};
   const real_t targetBulkVelocity_{};
   const real_t targetFrictionVelocity_{};

   CellInterval ci_{};

   real_t numCells_{};
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
                       const BlockDataID velocityfieldId,
                       const uint_t wallAxis,
                       const Vector3<real_t>& domainSize)
      : blocks_(blocks), velocityfieldId_(velocityfieldId), wallAxis_(wallAxis), numHeights_(uint_c(domainSize[ wallAxis]))
   {
      profile_.resize(numHeights_, 0_r);
      squaredProfile_.resize(numHeights_, 0_r);
      counts_.resize(numHeights_, 0);
      timeAveragedProfile_.resize(numHeights_, 0_r);
      timeAveragedSquaredProfile_.resize(numHeights_, 0_r);
      rmsProfile_.resize(numHeights_, 0_r);

      timeAveragedFluidProfile_.resize(numHeights_, 0_r);
      timeAveragedFluidSquaredProfile_.resize(numHeights_, 0_r);
      fluidRMSProfile_.resize(numHeights_, 0_r);

      timeAveragedParticleProfile_.resize(numHeights_, 0_r);
      timeAveragedParticleSquaredProfile_.resize(numHeights_, 0_r);
      particleRMSProfile_.resize(numHeights_, 0_r);

      timeStepCount_ = 0;
   }

   void computeAveragedProfiles()
   {
      std::fill(profile_.begin(), profile_.end(), 0_r);
      std::fill(squaredProfile_.begin(), profile_.end(), 0_r);
      std::fill(counts_.begin(), counts_.end(), 0);

      for (auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt)
      {
         auto* field = blockIt->template getData< Field_T >(velocityfieldId_);
         WALBERLA_CHECK_NOT_NULLPTR(field);

         const auto ci = field->xyzSize();

         for (auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt)
         {
            Cell globalCell(*cellIt);
            blocks_->transformBlockLocalToGlobalCell(globalCell, *blockIt);

            uint_t heightIdx = uint_c(globalCell[wallAxis_]);

            if (heightIdx < numHeights_)
            {
               real_t value = 0_r;

               if (Field_T::F_Size == 3)
               {
                  // For velocity: get wall-normal component (wallAxis direction)
                  value = field->get(*cellIt, wallAxis_);
               }
               else
               {
                  // For scalar: get the value
                  value = field->get(*cellIt);
               }

               profile_[heightIdx] += value;
               squaredProfile_[heightIdx] += value*value;
               counts_[heightIdx]++;
            }
         }
      }

      mpi::allReduceInplace(profile_, mpi::SUM);
      mpi::allReduceInplace(squaredProfile_, mpi::SUM);
      mpi::allReduceInplace(counts_, mpi::SUM);

      timeStepCount_++;
      for (uint_t h = 0; h < numHeights_; ++h)
      {
         if (counts_[h] > 0)
         {
            profile_[h] /= real_c(counts_[h]);
            squaredProfile_[h] /= real_c(counts_[h]);
            timeAveragedProfile_[h] += (profile_[h] - timeAveragedProfile_[h])/real_c(timeStepCount_);
            timeAveragedSquaredProfile_[h] += (squaredProfile_[h] - timeAveragedSquaredProfile_[h])/real_c(timeStepCount_);
         }
      }
   }

   /**
    * \brief Compute fluid and particle velocity fluctuation RMS values separately
    *
    * Decomposes velocity into fluid and particle components using B field (solid volume fraction):
    * fluidVelocity = velocity * (1 - B)
    * particleVelocity = velocity * B
    */

   void computeFluidParticleFluctuationRMS(const BlockDataID BFieldId)
   {
      // Initialize storage
      std::vector< real_t > fluidProfile(numHeights_, 0_r);
      std::vector< real_t > fluidSquaredProfile(numHeights_, 0_r);
      std::vector< real_t > particleProfile(numHeights_, 0_r);
      std::vector< real_t > particleSquaredProfile(numHeights_, 0_r);
      std::vector< uint_t > fluidCounts(numHeights_, 0);
      std::vector< uint_t > particleCounts(numHeights_, 0);

      // Accumulate fluid and particle velocity components
      for (auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt)
      {
         auto* velocityField = blockIt->template getData< Field_T >(velocityfieldId_);
         auto* BField = blockIt->template getData< GhostLayerField< real_t, 1 > >(BFieldId);
         WALBERLA_CHECK_NOT_NULLPTR(velocityField);
         WALBERLA_CHECK_NOT_NULLPTR(BField);

         const auto ci = velocityField->xyzSize();

         for (auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt)
         {
            Cell globalCell(*cellIt);
            blocks_->transformBlockLocalToGlobalCell(globalCell, *blockIt);

            uint_t heightIdx = uint_c(globalCell[wallAxis_]);

            if (heightIdx < numHeights_)
            {
               real_t B = BField->get(*cellIt);
               real_t velocity = velocityField->get(*cellIt, wallAxis_);

               // Fluid phase: (1-B) * u_fluid = u, so u_fluid = u / (1-B)
               if (B < 1.0_r)
               {
                  real_t fluidVel = velocity * (1.0_r - B);
                  fluidProfile[heightIdx] += fluidVel;
                  fluidSquaredProfile[heightIdx] += fluidVel * fluidVel;
                  fluidCounts[heightIdx]++;
               }

               // Particle phase: B * u_particle = u, so u_particle = u / B
               if (B > 0.0_r)
               {
                  real_t particleVel = velocity * B;
                  particleProfile[heightIdx] += particleVel;
                  particleSquaredProfile[heightIdx] += particleVel * particleVel;
                  particleCounts[heightIdx]++;
               }
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
      fluidRMSProfile_.resize(numHeights_, 0_r);
      particleRMSProfile_.resize(numHeights_, 0_r);

      for (uint_t h = 0; h < numHeights_; ++h)
      {
         if (fluidCounts[h] > 0)
         {
            fluidProfile[h] /= real_c(fluidCounts[h]);
            fluidSquaredProfile[h] /= real_c(fluidCounts[h]);

            timeAveragedFluidProfile_[h] += (fluidProfile[h] - timeAveragedFluidProfile_[h]) / real_c(timeStepCount_);
            timeAveragedFluidSquaredProfile_[h] += (fluidSquaredProfile[h] - timeAveragedFluidSquaredProfile_[h]) / real_c(timeStepCount_);
         }

         if (particleCounts[h] > 0)
         {
            particleProfile[h] /= real_c(particleCounts[h]);
            particleSquaredProfile[h] /= real_c(particleCounts[h]);

            timeAveragedParticleProfile_[h] += (particleProfile[h] - timeAveragedParticleProfile_[h]) / real_c(timeStepCount_);
            timeAveragedParticleSquaredProfile_[h] += (particleSquaredProfile[h] - timeAveragedParticleSquaredProfile_[h]) / real_c(timeStepCount_);
         }
      }
   }


   void computeFluidParticleRMSFromTimeAverage()
   {
      fluidRMSProfile_.resize(numHeights_, 0_r);
      particleRMSProfile_.resize(numHeights_, 0_r);

      for (uint_t h = 0; h < numHeights_; ++h)
      {
         real_t fluidVariance = timeAveragedFluidSquaredProfile_[h] - timeAveragedFluidProfile_[h] * timeAveragedFluidProfile_[h];
         fluidRMSProfile_[h] = std::sqrt(std::max(fluidVariance, 0_r));

         real_t particleVariance = timeAveragedParticleSquaredProfile_[h] - timeAveragedParticleProfile_[h] * timeAveragedParticleProfile_[h];
         particleRMSProfile_[h] = std::sqrt(std::max(particleVariance, 0_r));
      }
   }

   /**
    * \brief Get fluid RMS value at height
    */
   real_t getFluidRMSValue(const uint_t heightIdx) const
   {
      WALBERLA_ASSERT_LESS(heightIdx, numHeights_);
      return fluidRMSProfile_[heightIdx];
   }

   /**
    * \brief Get particle RMS value at height
    */
   real_t getParticleRMSValue(const uint_t heightIdx) const
   {
      WALBERLA_ASSERT_LESS(heightIdx, numHeights_);
      return particleRMSProfile_[heightIdx];
   }

   /**
    * \brief Get entire fluid RMS profile
    */
   const std::vector< real_t >& getFluidRMSProfile() const { return fluidRMSProfile_; }

   /**
    * \brief Get entire particle RMS profile
    */
   const std::vector< real_t >& getParticleRMSProfile() const { return particleRMSProfile_; }

   /**
    * \brief Get time-averaged fluid profile
    */
   const std::vector< real_t >& getTimeAveragedFluidProfile() const { return timeAveragedFluidProfile_; }

   /**
    * \brief Get time-averaged particle profile
    */
   const std::vector< real_t >& getTimeAveragedParticleProfile() const { return timeAveragedParticleProfile_; }

   /**
    * \brief Get RMS value at height
    */
   real_t getRMSValue(const uint_t heightIdx) const
   {
      WALBERLA_ASSERT_LESS(heightIdx, numHeights_);
      return rmsProfile_[heightIdx];
   }

   /**
    * \brief Get entire RMS profile
    */
   const std::vector< real_t >& getRMSProfile() const { return rmsProfile_; }

   /**
    * \brief Get time-averaged profile
    */
   const std::vector< real_t >& getTimeAveragedProfile() const { return timeAveragedProfile_; }

   /**
    * \brief Get time-averaged squared profile
    */
   const std::vector< real_t >& getTimeAveragedSquaredProfile() const { return timeAveragedSquaredProfile_; }

 private:
   const std::shared_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID velocityfieldId_;
   const uint_t wallAxis_;
   const uint_t numHeights_;

   std::vector< real_t > profile_;
   std::vector< real_t > squaredProfile_;
   std::vector< uint_t > counts_;
   std::vector< real_t > timeAveragedProfile_;
   std::vector< real_t > timeAveragedSquaredProfile_;
   std::vector< real_t > rmsProfile_;

   std::vector< real_t > timeAveragedFluidProfile_;
   std::vector< real_t > timeAveragedFluidSquaredProfile_;
   std::vector< real_t > fluidRMSProfile_;

   std::vector< real_t > timeAveragedParticleProfile_;
   std::vector< real_t > timeAveragedParticleSquaredProfile_;
   std::vector< real_t > particleRMSProfile_;

   uint_t timeStepCount_;
};






}  // namespace walberla

