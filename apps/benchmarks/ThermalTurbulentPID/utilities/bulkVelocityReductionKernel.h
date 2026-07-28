#pragma once

#include "gpu/FieldAccessor.h"

namespace walberla
{

__global__ void bulkVelocityReductionKernel(real_t * __restrict__ _data_velocity_field,real_t * __restrict__ _data_fraction_field,real_t* globalVelocitySum,
                                            real_t* globalWeightSum, int64_t const _size_x, int64_t const _size_y,
                                            int64_t const _size_z,int64_t _stride_x,int64_t _stride_y,int64_t _stride_z,int64_t _stride_f,
                                            size_t threadsPerBlock);
} // namespace walberla
