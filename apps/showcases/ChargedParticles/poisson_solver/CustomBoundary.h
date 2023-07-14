#ifndef WALBERLA_CUSTOMBOUNDARY_H
#define WALBERLA_CUSTOMBOUNDARY_H

#include "DirichletDomainBoundary.h"
#include "Neumann.h"

namespace walberla
{
typedef GhostLayerField< real_t, 1 > ScalarField_T;

template< typename PdeField >
class CustomBoundary : public DirichletDomainBoundary< PdeField >, public NeumannDomainBoundary< PdeField >
{
 public:
   CustomBoundary(StructuredBlockForest& blocks, const BlockDataID& fieldId,
                  std::vector< BoundaryCondition > boundaryconditions)
      : DirichletDomainBoundary< PdeField >(blocks, fieldId, boundaryconditions), NeumannDomainBoundary< PdeField >(
                                                                                     blocks, fieldId,
                                                                                     boundaryconditions)
   {}

   void operator()();
};

template< typename PdeField >
void CustomBoundary< PdeField >::operator()()
{
   DirichletDomainBoundary< PdeField >::operator()();
   NeumannDomainBoundary< PdeField >::operator()();
}

} // namespace walberla

#endif // WALBERLA_CUSTOMBOUNDARY_H
