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
}