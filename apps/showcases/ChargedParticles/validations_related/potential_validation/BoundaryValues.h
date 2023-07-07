//
// Created by avnss on 4/22/2023.
//

#ifndef WALBERLA_BOUNDARYVALUES_H
#define WALBERLA_BOUNDARYVALUES_H

namespace walberla
{
template< typename PdeField>
void applyBoundaryValue(IBlock* block, PdeField* p, const CellInterval& interval, const cell_idx_t cx,
    const cell_idx_t cy, const cell_idx_t cz, const BoundaryCondition& e,StructuredBlockStorage& blocks,const math::AABB & domainAABB,real_t charge,real_t epsilon,real_t dx_[])
{
    real_t funcVal;
   WALBERLA_FOR_ALL_CELLS_IN_INTERVAL_XYZ(
      interval, real_t boundaryCoord_x = 0.; real_t boundaryCoord_y = 0.; real_t boundaryCoord_z = 0.;

      const auto cellAABB = blocks.getBlockLocalCellAABB(*block, Cell(x, y, z)); auto cellCenter = cellAABB.center();

      // snap cell position to actual domain position
      switch (e.direction) {
         case stencil::W:
            boundaryCoord_x = domainAABB.xMin();
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::E:
            boundaryCoord_x = domainAABB.xMax();
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::S:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = domainAABB.yMin();
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::N:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = domainAABB.yMax();
            boundaryCoord_z = cellCenter[2];
            break;
         case stencil::B:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = domainAABB.zMin();
            break;
         case stencil::T:
            boundaryCoord_x = cellCenter[0];
            boundaryCoord_y = cellCenter[1];
            boundaryCoord_z = domainAABB.zMax();
            break;
         default:
            WALBERLA_ABORT("Unknown direction");
      }


      funcVal = real_c(0);
      if(e.type == "Dirichlet") {

          real_t distance = std::sqrt(pow((boundaryCoord_x - domainAABB.center()[0]),2) + pow((boundaryCoord_y -  domainAABB.center()[1]),2) + pow((boundaryCoord_z -  domainAABB.center()[2]),2));
          //WALBERLA_LOG_INFO_ON_ROOT("charge :" << charge << " " << "epsilon "<< " " << epsilon);
          //WALBERLA_LOG_INFO_ON_ROOT("xcoord" << " " << boundaryCoord_x <<" "<< "direction"<< e.direction);
          funcVal =  (charge /( 4 * math::pi * epsilon*distance));

          p->get(x, y, z) = real_c(2) * funcVal - p->get(x + cx, y + cy, z + cz);
      }
      else if(e.type == "Neumann"){
          funcVal = e.value;

          if (isIdentical(funcVal, real_t(0))) {
              p->get(x, y, z) = p->get(x + cx, y + cy, z + cz); // (dp / dx) == 0 _on_ the boundary
          } else {
              const real_t vdx = (funcVal * dx_[stencil::D3Q6::idx[e.direction]]); // value * dx

              p->get(x, y, z) = vdx + p->get(x + cx, y + cy, z + cz); // (dp / dx) == value _on_ the boundary
          }

      }


   )

}
} // namespace walberla

#endif // WALBERLA_BOUNDARYVALUES_H
