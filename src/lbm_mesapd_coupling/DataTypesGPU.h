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
//! \file DataTypesGPU.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"

#include "field/GhostLayerField.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/GPUField.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace gpu
{

// Maximum number of particles that may overlap with a cell. For fully resolved particles, 2 should normally be
// sufficient (for a sufficiently high stiffness in the DEM).
const uint MaxParticlesPerCell = 2;

// nOverlappingParticlesField is used to store the amount of overlapping particles per cell
// B denotes the local weighting factor and is calculated by taking the sum of all local particle
// weighting factor Bs. The naming of the variables is based on the following paper:
// https://doi.org/10.1016/j.compfluid.2017.05.033
// idxField is used to store the indices of the overlapping particles
// particleVelocitiesField is used to store the velocities of the overlapping particles evaluated at the cell center
// particleForcesField is used to store the hydrodynamic forces of the cell acting on the overlapping particles

using nOverlappingParticlesField_T    = GhostLayerField< uint_t, 1 >;
using nOverlappingParticlesFieldGPU_T = walberla::gpu::GPUField< uint_t >;
using BsField_T                       = GhostLayerField< real_t, MaxParticlesPerCell >;
using BsFieldGPU_T                    = walberla::gpu::GPUField< real_t >;
using idxField_T                      = GhostLayerField< size_t, MaxParticlesPerCell >;
using idxFieldGPU_T                   = walberla::gpu::GPUField< size_t >;
using BField_T                        = GhostLayerField< real_t, 1 >;
using BFieldGPU_T                     = walberla::gpu::GPUField< real_t >;
using particleVelocitiesFieldGPU_T    = walberla::gpu::GPUField< real_t >;
using particleForcesFieldGPU_T        = walberla::gpu::GPUField< real_t >;

// The ParticleAndVolumeFractionSoA encapsulates the data needed by the routines involved in the coupling
template< int Weighting_T >
struct ParticleAndVolumeFractionSoA_T
{
   BlockDataID nOverlappingParticlesFieldID;
   BlockDataID BsFieldID;
   BlockDataID idxFieldID;
   BlockDataID BFieldID;
   BlockDataID particleVelocitiesFieldID;
   BlockDataID particleForcesFieldID;
   // relaxation rate omega is used for Weighting_T != 1
   real_t omega_;
   // UIDs of the particles are stored during mapping, and it is checked that they are the same during the PSM kernel.
   // This prevents running into troubles due to changed indices
   std::vector< walberla::id_t > mappingUIDs;
   // Store positions globally to avoid copying them from CPU to GPU in multiple sweeps
   real_t* positions = nullptr;

   // nrOfGhostLayers is also 1 for the fields that do not need a ghost layer since the generated sweeps can only handle
   // fields with the same number of ghost layerserated kernels)
   ParticleAndVolumeFractionSoA_T(const shared_ptr< StructuredBlockStorage >& bs, const real_t omega)
   {
      nOverlappingParticlesFieldID = walberla::gpu::addGPUFieldToStorage< nOverlappingParticlesFieldGPU_T >(
         bs, "number of overlapping particles field GPU", uint_t(1), field::fzyx, uint_t(1), true);
      BsFieldID  = walberla::gpu::addGPUFieldToStorage< BsFieldGPU_T >(bs, "Bs field GPU", MaxParticlesPerCell,
                                                                       field::fzyx, uint_t(1), true);
      idxFieldID = walberla::gpu::addGPUFieldToStorage< idxFieldGPU_T >(bs, "idx field GPU", MaxParticlesPerCell,
                                                                         field::fzyx, uint_t(1), true);
      BFieldID =
         walberla::gpu::addGPUFieldToStorage< BFieldGPU_T >(bs, "B field GPU", 1, field::fzyx, uint_t(1), true);
      particleVelocitiesFieldID = walberla::gpu::addGPUFieldToStorage< particleVelocitiesFieldGPU_T >(
         bs, "particle velocities field GPU", uint_t(MaxParticlesPerCell * 3), field::fzyx, uint_t(1), true);
      particleForcesFieldID = walberla::gpu::addGPUFieldToStorage< particleForcesFieldGPU_T >(
         bs, "particle forces field GPU", uint_t(MaxParticlesPerCell * 3), field::fzyx, uint_t(1), true);
      omega_ = omega;
   }

   ~ParticleAndVolumeFractionSoA_T()
   {
      if (positions != nullptr) { cudaFree(positions); }
   }
};

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
