#pragma once

#include "gpu/FieldAccessor.h"


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
                                        const real_t* __restrict__ temperatures);

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
