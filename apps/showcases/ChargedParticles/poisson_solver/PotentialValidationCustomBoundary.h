#pragma once
#include "ApplyPotentialValidationBoundaryConditions.h"

namespace walberla
{
template< typename PdeField >
class PotentialCustomBoundary
{
 public:
   using ApplyFunction = std::function< void(IBlock* block, PdeField* p, const CellInterval& interval,
                                             const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz,
                                             BoundaryCondition& bcobject, StructuredBlockStorage& blocks,
                                             const math::AABB& domainAABB, const real_t charge, const real_t epsilon) >;

   PotentialCustomBoundary(StructuredBlockStorage& blocks, const BlockDataID& fieldId,
                           const std::vector< BoundaryCondition >& boundaryconditions, const math::AABB& domainAABB,
                           const real_t charge, const real_t epsilon)
      : blocks_(blocks), fieldId_(fieldId), boundaryconditions_(boundaryconditions), domainAABB_(domainAABB),
        charge_(charge), epsilon_(epsilon)
   {
      for (auto e : boundaryconditions_)
      {
         this->includeBoundary(e.getDirection());
         this->setValue(e.getDirection(), e.getValue());

         if (e.getType() == "Neumann")
         {
            if (e.getDirection() == stencil::E) { dx_[stencil::D3Q6::idx[stencil::E]] = blocks.dx(); }
            else if (e.getDirection() == stencil::W) { dx_[stencil::D3Q6::idx[stencil::W]] = blocks.dx(); }
            else if (e.getDirection() == stencil::S) { dx_[stencil::D3Q6::idx[stencil::S]] = blocks.dy(); }
            else if (e.getDirection() == stencil::N) { dx_[stencil::D3Q6::idx[stencil::N]] = blocks.dy(); }
            else if (e.getDirection() == stencil::T) { dx_[stencil::D3Q6::idx[stencil::T]] = blocks.dz(); }
            else { dx_[stencil::D3Q6::idx[stencil::B]] = blocks.dz(); }
         }
      }

#define GET_BOUNDARY_LAMBDA \
   [this](IBlock* block, ScalarField_T* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, \
          const cell_idx_t cz, BoundaryCondition& bcobject, StructuredBlockStorage& blocks__, \
          const math::AABB& domainAABB__, const real_t charge__, const real_t epsilon__) { \
      applyFunction< ScalarField_T >(block, p, interval, cx, cy, cz, bcobject, blocks__, domainAABB__, charge__, \
                                     epsilon__); \
   }

      this->setFunction(stencil::W, GET_BOUNDARY_LAMBDA);
      this->setFunction(stencil::E, GET_BOUNDARY_LAMBDA);
      this->setFunction(stencil::S, GET_BOUNDARY_LAMBDA);
      this->setFunction(stencil::N, GET_BOUNDARY_LAMBDA);
      this->setFunction(stencil::B, GET_BOUNDARY_LAMBDA);
      this->setFunction(stencil::T, GET_BOUNDARY_LAMBDA);
   }

   void includeBoundary(const stencil::Direction& direction) { includeBoundary_[stencil::D3Q6::idx[direction]] = true; }
   void excludeBoundary(const stencil::Direction& direction)
   {
      includeBoundary_[stencil::D3Q6::idx[direction]] = false;
   }

   void setFunction(const ApplyFunction func)
   {
      for (uint_t i = 0; i != stencil::D3Q6::Size; ++i)
         applyFunctions_[i] = func;
   }
   void setFunction(const stencil::Direction& direction, const ApplyFunction func)
   {
      applyFunctions_[stencil::D3Q6::idx[direction]] = func;
   }

   void setValue(const real_t value)
   {
      for (uint_t i = 0; i != stencil::D3Q6::Size; ++i)
         values_[i] = value;
   }
   void setValue(const stencil::Direction& direction, const real_t value)
   {
      values_[stencil::D3Q6::idx[direction]] = value;
   }

   template< typename pdeField >
   void applyFunction(IBlock* block, pdeField* p, const CellInterval& interval, const cell_idx_t cx,
                      const cell_idx_t cy, const cell_idx_t cz, BoundaryCondition& e, StructuredBlockStorage& blocks,
                      const math::AABB& domainAABB, const real_t charge, const real_t epsilon)
   {
      if (e.getType() == "Neumann")
      {
         dx_[stencil::D3Q6::idx[stencil::E]] = blocks.dx();
         dx_[stencil::D3Q6::idx[stencil::W]] = blocks.dx();
         dx_[stencil::D3Q6::idx[stencil::S]] = blocks.dy();
         dx_[stencil::D3Q6::idx[stencil::N]] = blocks.dy();
         dx_[stencil::D3Q6::idx[stencil::T]] = blocks.dz();
         dx_[stencil::D3Q6::idx[stencil::B]] = blocks.dz();
      }

      applyPotentialBoundaryValue(block, p, interval, cx, cy, cz, e, blocks, domainAABB, charge, epsilon, dx_);
   }

   void operator()();

 protected:
   StructuredBlockStorage& blocks_;
   BlockDataID fieldId_;
   const std::vector< BoundaryCondition > boundaryconditions_;
   const math::AABB& domainAABB_;
   const real_t charge_;
   const real_t epsilon_;
   bool includeBoundary_[stencil::D3Q6::Size];

   // user-defined apply function
   ApplyFunction applyFunctions_[stencil::D3Q6::Size];

   real_t values_[stencil::D3Q6::Size];
   real_t dx_[stencil::D3Q6::Size];
   uint_t order_[stencil::D3Q6::Size];

}; // class FunctionBoundary

template< typename PdeField >
void PotentialCustomBoundary< PdeField >::operator()()
{
   for (auto blockIt = blocks_.begin(); blockIt != blocks_.end(); ++blockIt)
   {
      auto* block = static_cast< blockforest::Block* >(&(*blockIt));

      PdeField* p = block->template getData< PdeField >(fieldId_);

      for (auto e : boundaryconditions_)
      {
         if (applyFunctions_[stencil::D3Q6::idx[stencil::W]] && includeBoundary_[stencil::D3Q6::idx[stencil::W]] &&
             blocks_.atDomainXMinBorder(*block) && e.getDirection() == stencil::W)
         {
            applyFunctions_[stencil::D3Q6::idx[stencil::W]](
               block, p,
               CellInterval(cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(1), cell_idx_t(0), cell_idx_t(0), e, blocks_, domainAABB_, charge_, epsilon_);
         }
         if (applyFunctions_[stencil::D3Q6::idx[stencil::E]] && includeBoundary_[stencil::D3Q6::idx[stencil::E]] &&
             blocks_.atDomainXMaxBorder(*block) && e.getDirection() == stencil::E)
         {
            applyFunctions_[stencil::D3Q6::idx[stencil::E]](
               block, p,
               CellInterval(cell_idx_c(p->xSize()), cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->xSize()),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), e, blocks_, domainAABB_, charge_, epsilon_);
         }

         if (applyFunctions_[stencil::D3Q6::idx[stencil::S]] && includeBoundary_[stencil::D3Q6::idx[stencil::S]] &&
             blocks_.atDomainYMinBorder(*block) && e.getDirection() == stencil::S)
         {
            applyFunctions_[stencil::D3Q6::idx[stencil::S]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), cell_idx_c(p->xSize()) - cell_idx_t(1),
                            cell_idx_t(-1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(1), cell_idx_t(0), e, blocks_, domainAABB_, charge_, epsilon_);
         }
         if (applyFunctions_[stencil::D3Q6::idx[stencil::N]] && includeBoundary_[stencil::D3Q6::idx[stencil::N]] &&
             blocks_.atDomainYMaxBorder(*block) && e.getDirection() == stencil::N)
         {
            applyFunctions_[stencil::D3Q6::idx[stencil::N]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_c(p->ySize()), cell_idx_t(0),
                            cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()),
                            cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), e, blocks_, domainAABB_, charge_, epsilon_);
         }

         if (applyFunctions_[stencil::D3Q6::idx[stencil::B]] && includeBoundary_[stencil::D3Q6::idx[stencil::B]] &&
             blocks_.atDomainZMinBorder(*block) && e.getDirection() == stencil::B)
         {
            applyFunctions_[stencil::D3Q6::idx[stencil::B]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), cell_idx_c(p->xSize()) - cell_idx_t(1),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_t(-1)),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(1), e, blocks_, domainAABB_, charge_, epsilon_);
         }
         if (applyFunctions_[stencil::D3Q6::idx[stencil::T]] && includeBoundary_[stencil::D3Q6::idx[stencil::T]] &&
             blocks_.atDomainZMaxBorder(*block) && e.getDirection() == stencil::T)
         {
            applyFunctions_[stencil::D3Q6::idx[stencil::T]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->zSize()),
                            cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()) - cell_idx_t(1),
                            cell_idx_c(p->zSize())),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), e, blocks_, domainAABB_, charge_, epsilon_);
         }
      }
   }
}
} // namespace walberla
