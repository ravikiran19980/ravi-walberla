#include <utility>

//
// Created by avnss on 4/7/2023.
//

#ifndef WALBERLA_BOUNDARYCONDITION_H
#define WALBERLA_BOUNDARYCONDITION_H

namespace walberla
{
class BoundaryCondition
{
public:
   stencil::Direction getDirection() { return direction_; }

   std::string getType() { return type_; }

   real_t getValue() { return value_; }

   BoundaryCondition(stencil::Direction direction, std::string type, real_t value)
      : direction_(direction), type_(std::move(type)), value_(value)
   {}

protected:
   stencil::Direction direction_;
   std::string type_;
   real_t value_;
}; // the struct defined here needs to be used in different files,

} // namespace walberla
#endif // WALBERLA_BOUNDARYCONDITION_H
