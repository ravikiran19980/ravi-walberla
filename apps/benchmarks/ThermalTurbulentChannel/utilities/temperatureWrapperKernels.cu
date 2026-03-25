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
//! \file temperatureWrapperKernels.cu
//! \ingroup lbm_mesapd_coupling
//! \author Ravi Ayyala <ravi.k.ayyala@fau.de>
//! \brief Provide two kernels that need to be called before and after the PSM kernel
//
//======================================================================================================================
#include "lbm_mesapd_coupling/partially_saturated_cells_method/codegen/PSMUtilityGPU.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/codegen/PSMWrapperKernels.h"
#include "temperatureWrapperKernels.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace gpu
{
__global__ void SetParticleTemperatures(walberla::gpu::FieldAccessor< uint_t > nOverlappingParticlesField,
                                        walberla::gpu::FieldAccessor< uint_t > idxField,
                                        walberla::gpu::FieldAccessor< real_t > particleTemperaturesField,
                                        const real_t* __restrict__ temperatures
)
{
   const uint3 blockIdx_uint3  = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
   const uint3 threadIdx_uint3 = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

   nOverlappingParticlesField.set(blockIdx_uint3, threadIdx_uint3);
   idxField.set(blockIdx_uint3, threadIdx_uint3);
   particleTemperaturesField.set(blockIdx_uint3, threadIdx_uint3);

   // Compute the particle velocity at the cell center for all overlapping particles
   for (uint_t p = 0; p < nOverlappingParticlesField.get(); p++)
   {
      particleTemperaturesField.get(p) = temperatures[idxField.get( p)];
   }
}

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
