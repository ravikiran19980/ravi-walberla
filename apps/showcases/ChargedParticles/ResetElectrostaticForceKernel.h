//
// Created by avnss on 3/18/2023.
// This file is for electrostatic coupling and hence has to be included in a different namespace and directory

#ifndef WALBERLA_RESETELECTROSTATICFORCEKERNEL_H
#define WALBERLA_RESETELECTROSTATICFORCEKERNEL_H

#pragma once

#include "core/math/Vector3.h"

#include "mesa_pd/data/IAccessor.h"

namespace walberla
{
/*
 * Kernel that resets the values of electrostaticForcee currently stored to zero
 */
class ResetElectrostaticForceKernel
{
 public:
   ResetElectrostaticForceKernel() = default;

   template< typename ParticleAccessor_T >
   void operator()(const size_t idx, ParticleAccessor_T& ac) const
   {
      static_assert(std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value,
                    "Provide a valid accessor as template");

      ac.setElectrostaticForce(idx, Vector3< real_t >(real_t(0)));
   }
};

} // namespace walberla

#endif // WALBERLA_RESETELECTROSTATICFORCEKERNEL_H
