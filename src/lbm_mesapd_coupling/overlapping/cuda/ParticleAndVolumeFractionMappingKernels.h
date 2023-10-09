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
//! \file ParticleAndVolumeFractionMappingKernels.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"
#include "core/math/Vector3.h"

#include "gpu/FieldAccessor.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace gpu
{

template< int Weighting_T >
__global__ void
   superSampling(walberla::gpu::FieldAccessor< uint_t > nOverlappingParticlesField,
                 walberla::gpu::FieldAccessor< real_t > BsField, walberla::gpu::FieldAccessor< id_t > idxField,
                 walberla::gpu::FieldAccessor< real_t > BField, real_t omega,
                 real_t* __restrict__ const spherePositions, real_t* __restrict__ const sphereRadii, double3 blockStart,
                 real_t dx, int3 nSamples, size_t* __restrict__ const numParticlesSubBlocks,
                 size_t* __restrict__ const particleIDsSubBlocks, const size_t subBlocksPerDim);

template< int Weighting_T >
__global__ void linearApproximation(walberla::gpu::FieldAccessor< uint_t > nOverlappingParticlesField,
                                    walberla::gpu::FieldAccessor< real_t > BsField,
                                    walberla::gpu::FieldAccessor< id_t > idxField,
                                    walberla::gpu::FieldAccessor< real_t > BField, real_t omega,
                                    real_t* __restrict__ const spherePositions, real_t* __restrict__ const sphereRadii,
                                    real_t* __restrict__ const f_rs, double3 blockStart, real_t dx,
                                    size_t* __restrict__ const numParticlesSubBlocks,
                                    size_t* __restrict__ const particleIDsSubBlocks, const size_t subBlocksPerDim);

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
