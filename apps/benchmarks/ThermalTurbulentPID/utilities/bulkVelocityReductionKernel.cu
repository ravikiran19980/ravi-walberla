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
//  You should have received a copy of the GNU General Public License
//  along with waLBerla (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.
//
//! \file bulkVelocityReductionKernel.cu
//! \ingroup lbm_mesapd_coupling
//
//======================================================================================================================

#include "core/DataTypes.h"
#include "gpu/GPUWrapper.h"

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT

namespace walberla
{

#define BLOCK_SIZE 256

__global__ void bulkVelocityReductionKernel(
    gpu::FieldAccessor< real_t > meanVelocity,
    gpu::FieldAccessor< real_t > maskField,
    real_t* globalVelocitySum,
    real_t* globalWeightSum,
    int N)
{
   __shared__ real_t sVelSum[BLOCK_SIZE];
   __shared__ real_t sWeightSum[BLOCK_SIZE];

   const uint3 blockIdx_uint3  = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
   const uint3 threadIdx_uint3 = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

   meanVelocity.set(blockIdx_uint3, threadIdx_uint3);
   maskField.set(blockIdx_uint3, threadIdx_uint3);

   real_t localVel = 0_r;
   real_t localWeight = 0_r;

   if (idx < N)
   {
      real_t B = maskField.get();
      real_t fluidWeight = 1_r - B;
      if (fluidWeight > 0_r)
      {
         localVel = meanVelocity.get(0) * fluidWeight;
         localWeight = fluidWeight;
      }
   }

   sVelSum[tid]    = localVel;
   sWeightSum[tid] = localWeight;
   __syncthreads();

   for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
   {
      if (tid < stride)
      {
         sVelSum[tid] += sVelSum[tid + stride];
         sWeightSum[tid] += sWeightSum[tid + stride];
      }
      __syncthreads();
   }

   if (tid == 0)
   {
      atomicAdd(globalVelocitySum, sVelSum[0]);
      atomicAdd(globalWeightSum, sWeightSum[0]);
   }
}

} // namespace walberla

#endif
