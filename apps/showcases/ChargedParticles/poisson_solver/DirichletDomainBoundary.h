#ifndef WALBERLA_DIRICHLETDOMAINBOUNDARY_H
#define WALBERLA_DIRICHLETDOMAINBOUNDARY_H

#include "BoundaryCondition.h"
#include "stencil/D3Q6.h"

namespace walberla
{
template< typename PdeField >
class DirichletDomainBoundary
{
 public:
   DirichletDomainBoundary(StructuredBlockStorage& blocks, const BlockDataID& fieldId,
                           const std::vector< BoundaryCondition >& boundaryConditions)
      : blocks_(blocks), fieldId_(fieldId), boundaryconditions_(boundaryConditions)
   {
      // ravi implementation

      for (auto e : boundaryConditions)
      {
         if (e.getType() == "Neumann") { this->excludeBoundary(e.getDirection()); } // swap conditions
         else
         {
            this->includeBoundary(e.getDirection());
            this->setValue(e.getDirection(), e.getValue());
         }
      }
   }

   void includeBoundary(const stencil::Direction& direction) { includeBoundary_[stencil::D3Q6::idx[direction]] = true; }
   void excludeBoundary(const stencil::Direction& direction)
   {
      includeBoundary_[stencil::D3Q6::idx[direction]] = false;
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

   virtual void operator()();

   virtual ~DirichletDomainBoundary() = default;

 protected:
   // first-order dirichlet boundary conditions with scalar value
   void apply(PdeField* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz,
              const real_t value) const;

   StructuredBlockStorage& blocks_;
   BlockDataID fieldId_;
   std::vector< BoundaryCondition > boundaryconditions_;

   bool includeBoundary_[stencil::D3Q6::Size];

   real_t values_[stencil::D3Q6::Size];

}; // class DirichletDomainBoundary

template< typename PdeField >
void DirichletDomainBoundary< PdeField >::operator()()
{
   for (auto block = blocks_.begin(); block != blocks_.end(); ++block)
   {
      PdeField* p = block->template getData< PdeField >(fieldId_);

      if (includeBoundary_[stencil::D3Q6::idx[stencil::W]] && blocks_.atDomainXMinBorder(*block))
      {
         apply(p,
               CellInterval(cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(1), cell_idx_t(0), cell_idx_t(0), values_[stencil::D3Q6::idx[stencil::W]]);
      }
      if (includeBoundary_[stencil::D3Q6::idx[stencil::E]] && blocks_.atDomainXMaxBorder(*block))
      {
         apply(p,
               CellInterval(cell_idx_c(p->xSize()), cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->xSize()),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), values_[stencil::D3Q6::idx[stencil::E]]);
      }

      if (includeBoundary_[stencil::D3Q6::idx[stencil::S]] && blocks_.atDomainYMinBorder(*block))
      {
         apply(p,
               CellInterval(cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), cell_idx_c(p->xSize()) - cell_idx_t(1),
                            cell_idx_t(-1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(1), cell_idx_t(0), values_[stencil::D3Q6::idx[stencil::S]]);
      }
      if (includeBoundary_[stencil::D3Q6::idx[stencil::N]] && blocks_.atDomainYMaxBorder(*block))
      {
         apply(p,
               CellInterval(cell_idx_t(0), cell_idx_c(p->ySize()), cell_idx_t(0),
                            cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()),
                            cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), values_[stencil::D3Q6::idx[stencil::N]]);
      }

      if (includeBoundary_[stencil::D3Q6::idx[stencil::B]] && blocks_.atDomainZMinBorder(*block))
      {
         apply(p,
               CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), cell_idx_c(p->xSize()) - cell_idx_t(1),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_t(-1)),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(1), values_[stencil::D3Q6::idx[stencil::B]]);
      }
      if (includeBoundary_[stencil::D3Q6::idx[stencil::T]] && blocks_.atDomainZMaxBorder(*block))
      {
         apply(p,
               CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->zSize()),
                            cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()) - cell_idx_t(1),
                            cell_idx_c(p->zSize())),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), values_[stencil::D3Q6::idx[stencil::T]]);
      }
   }
}

template< typename PdeField >
void DirichletDomainBoundary< PdeField >::apply(PdeField* p, const CellInterval& interval, const cell_idx_t cx,
                                                const cell_idx_t cy, const cell_idx_t cz, const real_t value) const
{
   WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(interval,
                                          p->get(x, y, z) = real_c(2) * value - p->get(x + cx, y + cy, z + cz);)
}

template< typename PdeField >
class DirichletFunctionDomainBoundary
{
 public:
   using ApplyFunction = std::function< void(IBlock* block, PdeField* p, const CellInterval& interval,
                                             const cell_idx_t cx, const cell_idx_t cy, const cell_idx_t cz) >;

   DirichletFunctionDomainBoundary(StructuredBlockStorage& blocks, const BlockDataID& fieldId)
      : blocks_(blocks), fieldId_(fieldId)
   {
      for (uint_t i = 0; i != stencil::D3Q6::Size; ++i)
      {
         includeBoundary_[i] = true;

         applyFunctions_[i] = {};
      }
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

   void operator()();

 protected:
   StructuredBlockStorage& blocks_;
   BlockDataID fieldId_;

   bool includeBoundary_[stencil::D3Q6::Size];

   // user-defined apply function
   ApplyFunction applyFunctions_[stencil::D3Q6::Size];

}; // class DirichletFunctionDomainBoundary

template< typename PdeField >
void DirichletFunctionDomainBoundary< PdeField >::operator()()
{
   for (auto blockIt = blocks_.begin(); blockIt != blocks_.end(); ++blockIt)
   {
      auto* block = static_cast< blockforest::Block* >(&(*blockIt));

      PdeField* p = block->template getData< PdeField >(fieldId_);

      if (applyFunctions_[stencil::D3Q6::idx[stencil::W]] && includeBoundary_[stencil::D3Q6::idx[stencil::W]] &&
          blocks_.atDomainXMinBorder(*block))
      {
         applyFunctions_[stencil::D3Q6::idx[stencil::W]](
            block, p,
            CellInterval(cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1),
                         cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
            cell_idx_t(1), cell_idx_t(0), cell_idx_t(0));
      }
      if (applyFunctions_[stencil::D3Q6::idx[stencil::E]] && includeBoundary_[stencil::D3Q6::idx[stencil::E]] &&
          blocks_.atDomainXMaxBorder(*block))
      {
         applyFunctions_[stencil::D3Q6::idx[stencil::E]](
            block, p,
            CellInterval(cell_idx_c(p->xSize()), cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->xSize()),
                         cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
            cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0));
      }

      if (applyFunctions_[stencil::D3Q6::idx[stencil::S]] && includeBoundary_[stencil::D3Q6::idx[stencil::S]] &&
          blocks_.atDomainYMinBorder(*block))
      {
         applyFunctions_[stencil::D3Q6::idx[stencil::S]](
            block, p,
            CellInterval(cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), cell_idx_c(p->xSize()) - cell_idx_t(1),
                         cell_idx_t(-1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
            cell_idx_t(0), cell_idx_t(1), cell_idx_t(0));
      }
      if (applyFunctions_[stencil::D3Q6::idx[stencil::N]] && includeBoundary_[stencil::D3Q6::idx[stencil::N]] &&
          blocks_.atDomainYMaxBorder(*block))
      {
         applyFunctions_[stencil::D3Q6::idx[stencil::N]](
            block, p,
            CellInterval(cell_idx_t(0), cell_idx_c(p->ySize()), cell_idx_t(0), cell_idx_c(p->xSize()) - cell_idx_t(1),
                         cell_idx_c(p->ySize()), cell_idx_c(p->zSize()) - cell_idx_t(1)),
            cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0));
      }

      if (applyFunctions_[stencil::D3Q6::idx[stencil::B]] && includeBoundary_[stencil::D3Q6::idx[stencil::B]] &&
          blocks_.atDomainZMinBorder(*block))
      {
         applyFunctions_[stencil::D3Q6::idx[stencil::B]](
            block, p,
            CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), cell_idx_c(p->xSize()) - cell_idx_t(1),
                         cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_t(-1)),
            cell_idx_t(0), cell_idx_t(0), cell_idx_t(1));
      }
      if (applyFunctions_[stencil::D3Q6::idx[stencil::T]] && includeBoundary_[stencil::D3Q6::idx[stencil::T]] &&
          blocks_.atDomainZMaxBorder(*block))
      {
         applyFunctions_[stencil::D3Q6::idx[stencil::T]](
            block, p,
            CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->zSize()), cell_idx_c(p->xSize()) - cell_idx_t(1),
                         cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize())),
            cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1));
      }
   }
}

} // namespace walberla

#endif // WALBERLA_DIRICHLETDOMAINBOUNDARY_H
