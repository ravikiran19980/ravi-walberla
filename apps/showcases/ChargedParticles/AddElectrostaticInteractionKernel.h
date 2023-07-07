//
// Created by avnss on 3/18/2023.
// This file is for electrostatic coupling and hence has to be included in a different namespace and directory

#ifndef WALBERLA_ADDELECTROSTATICINTERACTIONKERNEL_H
#define WALBERLA_ADDELECTROSTATICINTERACTIONKERNEL_H

#pragma once

#include "mesa_pd/common/ParticleFunctions.h"
#include "mesa_pd/data/IAccessor.h"

namespace walberla
{
/*
 * Kernel that adds the current electrostatic forces and torques onto the particles as forces and torques
 *
 * When reducing hyd. force/torque before, this should usually be carried out only on local particles. (recommended)
 * If not, it must be carried out on local and ghost particles.
 */
class AddElectrostaticInteractionKernel
{
 public:
   AddElectrostaticInteractionKernel() = default;

   template< typename ParticleAccessor_T >
   void operator()(const size_t idx, ParticleAccessor_T& ac) const
   {
      static_assert(std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value,
                    "Provide a valid accessor as template");

      mesa_pd::addForceAtomic(idx, ac, ac.getElectrostaticForce(idx));
   }
};

} // namespace walberla

#endif // WALBERLA_ADDELECTROSTATICINTERACTIONKERNEL_H
