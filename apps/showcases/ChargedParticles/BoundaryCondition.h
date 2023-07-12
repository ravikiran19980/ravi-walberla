//
// Created by avnss on 4/7/2023.
//

#ifndef WALBERLA_BOUNDARYCONDITION_H
#define WALBERLA_BOUNDARYCONDITION_H

namespace walberla
{
struct BoundaryCondition
{
   stencil::Direction direction;
   std::string type;
   double value;

   BoundaryCondition(stencil::Direction direction, std::string type, double value)
      : direction(direction), type(type), value(value)
   {}
}; // the struct defined here needs to be used in different files,

} // namespace walberla
#endif // WALBERLA_BOUNDARYCONDITION_H
