#include <utility>

//
// Created by avnss on 4/7/2023.
//

#ifndef WALBERLA_BOUNDARYCONDITION_H
#define WALBERLA_BOUNDARYCONDITION_H

namespace walberla
{
struct BoundaryCondition
{
   stencil::Direction getDirection() {
      return direction_;
   }

   std::string getType() {
      return type_;
   }

   double getValue() {
      return value_;
   }

   BoundaryCondition(stencil::Direction direction, std::string type, double value)
      : direction_(direction), type_(std::move(type)), value_(value)
   {}

 protected:
   stencil::Direction direction_;
   std::string type_;
   double value_;
}; // the struct defined here needs to be used in different files,

} // namespace walberla
#endif // WALBERLA_BOUNDARYCONDITION_H
