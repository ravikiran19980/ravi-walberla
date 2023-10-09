//======================================================================================================================
//
//  This file is part of waLBerla. waLBerla is free software: you can
//  redistribute it and/or modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation, either version 3 of
//  the License, or (at your option) any later version.
//
//  waLBerla is distributed in the hope that it will be useful, but WITHOUT
//  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
//  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
//  for more details.
//
//  You should have received a copy of the GNU General Public License along
//  with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file OverlapFraction.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \brief Functor that provides overlap fraction computations for different MESA-PD shapes (used by SingleCast)
//
//======================================================================================================================

#pragma once

#include "mesa_pd/data/shape/Sphere.h"

#include <cmath>

#include "ParticleAndVolumeFractionMappingKernels.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace gpu
{

// Deprecated
/*struct OverlapFractionFunctor
{
   template< typename ParticleAccessor_T, typename Shape_T, int Weighting_T >
   void operator()(const size_t *//*particleIdx*//*, const Shape_T& *//*shape*//*,
                   const shared_ptr< ParticleAccessor_T >& *//*ac*//*, const IBlock& *//*blockIt*//*,
                   const ParticleAndVolumeFractionSoA_T< Weighting_T >& *//*particleAndVolumeFractionSoA*//*,
                   const real_t *//*omega*//*, const size_t *//*particleIdxMapped*//*)
   {
      WALBERLA_ABORT("OverlapFraction not implemented!");
   }

   template< typename ParticleAccessor_T, int Weighting_T >
   void operator()(const size_t particleIdx, const mesa_pd::data::Sphere& *//*sphere*//*,
                   const shared_ptr< ParticleAccessor_T >& ac, const IBlock& blockIt,
                   const ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                   const real_t omega, const size_t particleIdxMapped)
   {
      WALBERLA_STATIC_ASSERT((std::is_base_of< mesa_pd::data::IAccessor, ParticleAccessor_T >::value));

      Vector3< real_t > particlePosition = ac->getPosition(particleIdx);

      auto nOverlappingParticlesField =
         blockIt.getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA.nOverlappingParticlesFieldID);
      auto BsField  = blockIt.getData< BsFieldGPU_T >(particleAndVolumeFractionSoA.BsFieldID);
      auto idxField = blockIt.getData< idxFieldGPU_T >(particleAndVolumeFractionSoA.idxFieldID);
      auto BField   = blockIt.getData< BFieldGPU_T >(particleAndVolumeFractionSoA.BFieldID);

      auto myKernel = walberla::gpu::make_kernel(&(linearApproximation< Weighting_T >) );
      myKernel.addFieldIndexingParam(
         walberla::gpu::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));          // FieldAccessor
      myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< real_t >::xyz(*BsField)); // FieldAccessor
      myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< id_t >::xyz(*idxField));  // FieldAccessor
      myKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< real_t >::xyz(*BField));  // FieldAccessor
      myKernel.addParam(omega);
      Vector3< real_t > blockStart = blockIt.getAABB().minCorner();
      myKernel.addParam(double3{ particlePosition[0], particlePosition[1], particlePosition[2] }); // spherePosition
      real_t radius = static_cast< mesa_pd::data::Sphere* >(ac->getShape(particleIdx))->getRadius();
      myKernel.addParam(radius); // sphereRadius
      real_t Va =
         real_t((1.0 / 12.0 - radius * radius) * atan((0.5 * sqrt(radius * radius - 0.5)) / (0.5 - radius * radius)) +
                1.0 / 3.0 * sqrt(radius * radius - 0.5) +
                (radius * radius - 1.0 / 12.0) * atan(0.5 / sqrt(radius * radius - 0.5)) -
                4.0 / 3.0 * radius * radius * radius * atan(0.25 / (radius * sqrt(radius * radius - 0.5))));
      myKernel.addParam(Va - radius + real_t(0.5));
      myKernel.addParam(double3{ blockStart[0], blockStart[1], blockStart[2] });                  // blockStart
      myKernel.addParam(blockIt.getAABB().xSize() / real_t(nOverlappingParticlesField->xSize())); // dx
      // myKernel.addParam(int3{ 16, 16, 16 });                                                            // nSamples
      myKernel.addParam(particleIdxMapped);
      myKernel();
   }
};*/

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
