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
class PlaneAveragedProfile
{
 public:

   PlaneAveragedProfile(const std::shared_ptr< StructuredBlockStorage >& blocks,
                       const BlockDataID fieldId,
                       const uint_t wallAxis,
                       const Vector3<real_t>& domainSize)
      : blocks_(blocks), fieldId_(fieldId), wallAxis_(wallAxis), numHeights_(uint_c(domainSize[ wallAxis]))
   {
      profile_.resize(numHeights_, 0_r);
      counts_.resize(numHeights_, 0);
      timeAveragedProfile_.resize(numHeights_, 0_r);
      timeStepCount_ = 0;
   }

   void computeAveragedProfiles()
   {
      std::fill(profile_.begin(), profile_.end(), 0_r);
      std::fill(counts_.begin(), counts_.end(), 0);

      for (auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt)
      {
         auto* field = blockIt->template getData< Field_T >(fieldId_);
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
               counts_[heightIdx]++;
            }
         }
      }

      mpi::allReduceInplace(profile_, mpi::SUM);
      mpi::allReduceInplace(counts_, mpi::SUM);

      timeStepCount_++;
      for (uint_t h = 0; h < numHeights_; ++h)
      {
         if (counts_[h] > 0)
         {
            profile_[h] /= real_c(counts_[h]);
            timeAveragedProfile_[h] += (profile_[h] - timeAveragedProfile_[h])/real_c(timeStepCount_);
         }
      }
   }

 private:
   const std::shared_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID fieldId_;
   const uint_t wallAxis_;
   const uint_t numHeights_;

   std::vector< real_t > profile_;
   std::vector< uint_t > counts_;
   std::vector< real_t > timeAveragedProfile_;
   uint_t timeStepCount_;
};

}  // namespace walberla

