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

#include "core/logging/Logging.h"

#include "gpu/GPUWrapper.h"

#include "bulkVelocityReductionKernel.h"

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT

namespace walberla
{

__global__ void bulkVelocityReductionKernel(real_t * __restrict__ _data_velocity_field,real_t * __restrict__ _data_fraction_field,real_t* globalVelocitySum,
                                            real_t* globalWeightSum, int64_t const _size_x, int64_t const _size_y,
                                            int64_t const _size_z,int64_t _stride_x,int64_t _stride_y,int64_t _stride_z,int64_t _stride_f,
                                            size_t threadsPerBlock)
{
   extern __shared__ real_t sharedMem[];
   real_t * VelSum = sharedMem;
   real_t * WeightSum = sharedMem + threadsPerBlock;

   const unsigned int tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;
   const unsigned int bid = blockIdx.z *  (gridDim.y  * gridDim.x)  + blockIdx.y  * gridDim.x  + blockIdx.x;
   const unsigned int n   = blockDim.x * blockDim.y * blockDim.z;

   // Every thread must initialize its shared-memory entries.
    VelSum[tid] = real_t(0);
    WeightSum[tid]   = real_t(0);
   const int64_t x = blockIdx.x * blockDim.x + threadIdx.x;
   const int64_t y = blockIdx.y * blockDim.y + threadIdx.y;
   const int64_t z = blockIdx.z * blockDim.z + threadIdx.z;

   real_t localVel    = 0;
   real_t localWeight = 0;


   if (x < _size_x && y < _size_y && z < _size_z) {
      localWeight =  _data_fraction_field[ x * _stride_x + y * _stride_y + z * _stride_z ];

      if ( (1 - localWeight) > 0)
      {
         localVel   = _data_velocity_field[ x * _stride_x + y * _stride_y + z * _stride_z + 0 * _stride_f ];
         VelSum[tid] = localVel * (1 - localWeight);
         WeightSum[tid] = (1 - localWeight);
      }
   }
   __syncthreads();

   for ( int stride = n / 2; stride > 0; stride >>= 1 )
   {
      if ( tid < stride )
      {
         VelSum[tid] += VelSum[tid + stride];
         WeightSum[tid] += WeightSum[tid + stride];

      }
      __syncthreads();
   }

   // look closely at this part later for now its fine
   if ( tid == 0 )
   {
      atomicAdd( globalVelocitySum, VelSum[0] );
      atomicAdd( globalWeightSum, WeightSum[0] );
   }
}

} // namespace walberla

#endif
