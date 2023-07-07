//
// Created by avnss on 4/8/2023.
//

#ifndef WALBERLA_CUSTOMFUNCTIONBOUNDARY_H
#define WALBERLA_CUSTOMFUNCTIONBOUNDARY_H

namespace walberla
{
template< typename PdeField >
class CustomFunctionBoundary
{
 public:
   using ApplyFunction =
      std::function< void(IBlock* block, PdeField* p, const CellInterval& interval, const cell_idx_t cx,
                          const cell_idx_t cy, const cell_idx_t cz, BoundaryCondition& bcobject) >;

   CustomFunctionBoundary(StructuredBlockStorage& blocks, const BlockDataID& fieldId,
                          const std::vector< BoundaryCondition >& boundaryconditions)
      : blocks_(blocks), fieldId_(fieldId), boundaryconditions_(boundaryconditions)
   {
      for (auto e : boundaryconditions_)
      {
         this->includeBoundary(e.direction);
         this->setValue(e.direction, e.value);
         // std::cout << values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction << std::endl;
         if (e.type == "Neumann")
         {
               dx_[stencil::D3Q6::idx[stencil::E]] = blocks.dx();
               dx_[stencil::D3Q6::idx[stencil::W]] = blocks.dx();
               dx_[stencil::D3Q6::idx[stencil::S]] = blocks.dy();
               dx_[stencil::D3Q6::idx[stencil::N]] = blocks.dy();
               dx_[stencil::D3Q6::idx[stencil::T]] = blocks.dz();
               dx_[stencil::D3Q6::idx[stencil::B]] = blocks.dz();

         }
      }

#define GET_BOUNDARY_LAMBDA \
   [this](IBlock* block, ScalarField_T* p, const CellInterval& interval, const cell_idx_t cx, const cell_idx_t cy, \
          const cell_idx_t cz, const BoundaryCondition& bcobject) { \
      applyFunction< ScalarField_T >(block, p, interval, cx, cy, cz, bcobject); \
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

   template< typename pdefield >
   void applyFunction(IBlock* block, pdefield* p, const CellInterval& interval, const cell_idx_t cx,
                      const cell_idx_t cy, const cell_idx_t cz, const BoundaryCondition& e)
   {

      this->setValue(e.direction, e.value); // working if called here.. Not in the constructor .... strange!
      if (e.type == "Dirichlet")
      {
         WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(interval,
                                                p->get(x, y, z) = real_c(2) * values_[stencil::D3Q6::idx[e.direction]] -
                                                                  p->get(x + cx, y + cy, z + cz);)
         //if(e.direction == stencil::W){ std::cout <<
         // values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction << std::endl;
         //}
      }

      else
      {

          dx_[stencil::D3Q6::idx[stencil::E]] = blocks_.dx();
          dx_[stencil::D3Q6::idx[stencil::W]] = blocks_.dx();
          dx_[stencil::D3Q6::idx[stencil::S]] = blocks_.dy();
          dx_[stencil::D3Q6::idx[stencil::N]] = blocks_.dy();
          dx_[stencil::D3Q6::idx[stencil::T]] = blocks_.dz();
          dx_[stencil::D3Q6::idx[stencil::B]] = blocks_.dz();

         if (isIdentical(values_[stencil::D3Q6::idx[e.direction]], real_t(0)))
         {
            WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(
               interval, p->get(x, y, z) = p->get(x + cx, y + cy, z + cz); // (dp / dx) == 0 _on_ the boundary
            )
         }
         else
         {
            const real_t vdx =
               (values_[stencil::D3Q6::idx[e.direction]]) * dx_[stencil::D3Q6::idx[e.direction]]; // value * dx
            WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(
               interval, p->get(x, y, z) = vdx + p->get(x + cx, y + cy, z + cz); // (dp / dx) == value _on_ the boundary
            )
         }
      }
   }

   void operator()();

 protected:
   StructuredBlockStorage& blocks_;
   BlockDataID fieldId_;
   const std::vector< BoundaryCondition > boundaryconditions_;
   bool includeBoundary_[stencil::D3Q6::Size];

   // user-defined apply function
   ApplyFunction applyFunctions_[stencil::D3Q6::Size];

   real_t values_[stencil::D3Q6::Size];
   real_t dx_[stencil::D3Q6::Size];
   uint_t order_[stencil::D3Q6::Size];

}; // class CustomFunctionBoundary

template< typename PdeField >
void CustomFunctionBoundary< PdeField >::operator()()
{
   for (auto blockIt = blocks_.begin(); blockIt != blocks_.end(); ++blockIt)
   {
      auto* block = static_cast< blockforest::Block* >(&(*blockIt));

      PdeField* p = block->template getData< PdeField >(fieldId_);

      for (auto e : boundaryconditions_)
      {
         if (applyFunctions_[stencil::D3Q6::idx[stencil::W]] && includeBoundary_[stencil::D3Q6::idx[stencil::W]] &&
             blocks_.atDomainXMinBorder(*block) && e.direction == stencil::W)
         {
            // std::cout << values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction <<
            // std::endl;
            applyFunctions_[stencil::D3Q6::idx[stencil::W]](
               block, p,
               CellInterval(cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(1), cell_idx_t(0), cell_idx_t(0), e);
         }
         if (applyFunctions_[stencil::D3Q6::idx[stencil::E]] && includeBoundary_[stencil::D3Q6::idx[stencil::E]] &&
             blocks_.atDomainXMaxBorder(*block) && e.direction == stencil::E)
         {
            // std::cout << values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction <<
            // std::endl;
            applyFunctions_[stencil::D3Q6::idx[stencil::E]](
               block, p,
               CellInterval(cell_idx_c(p->xSize()), cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->xSize()),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(-1), cell_idx_t(0), cell_idx_t(0), e);
         }

         if (applyFunctions_[stencil::D3Q6::idx[stencil::S]] && includeBoundary_[stencil::D3Q6::idx[stencil::S]] &&
             blocks_.atDomainYMinBorder(*block) && e.direction == stencil::S)
         {
            // std::cout << values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction <<
            // std::endl;
            applyFunctions_[stencil::D3Q6::idx[stencil::S]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), cell_idx_c(p->xSize()) - cell_idx_t(1),
                            cell_idx_t(-1), cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(1), cell_idx_t(0), e);
         }
         if (applyFunctions_[stencil::D3Q6::idx[stencil::N]] && includeBoundary_[stencil::D3Q6::idx[stencil::N]] &&
             blocks_.atDomainYMaxBorder(*block) && e.direction == stencil::N)
         {
            // std::cout << values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction <<
            // std::endl;
            applyFunctions_[stencil::D3Q6::idx[stencil::N]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_c(p->ySize()), cell_idx_t(0),
                            cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()),
                            cell_idx_c(p->zSize()) - cell_idx_t(1)),
               cell_idx_t(0), cell_idx_t(-1), cell_idx_t(0), e);
         }

         if (applyFunctions_[stencil::D3Q6::idx[stencil::B]] && includeBoundary_[stencil::D3Q6::idx[stencil::B]] &&
             blocks_.atDomainZMinBorder(*block) && e.direction == stencil::B)
         {
            // std::cout << values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction <<
            // std::endl;
            applyFunctions_[stencil::D3Q6::idx[stencil::B]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), cell_idx_c(p->xSize()) - cell_idx_t(1),
                            cell_idx_c(p->ySize()) - cell_idx_t(1), cell_idx_t(-1)),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(1), e);
         }
         if (applyFunctions_[stencil::D3Q6::idx[stencil::T]] && includeBoundary_[stencil::D3Q6::idx[stencil::T]] &&
             blocks_.atDomainZMaxBorder(*block) && e.direction == stencil::T)
         {
            // std::cout << values_[stencil::D3Q6::idx[e.direction]] << " " <<  e.type <<  " "<< e.direction <<
            // std::endl;
            applyFunctions_[stencil::D3Q6::idx[stencil::T]](
               block, p,
               CellInterval(cell_idx_t(0), cell_idx_t(0), cell_idx_c(p->zSize()),
                            cell_idx_c(p->xSize()) - cell_idx_t(1), cell_idx_c(p->ySize()) - cell_idx_t(1),
                            cell_idx_c(p->zSize())),
               cell_idx_t(0), cell_idx_t(0), cell_idx_t(-1), e);
         }
      }
   }
}
} // namespace walberla

#endif // WALBERLA_CUSTOMFUNCTIONBOUNDARY_H
