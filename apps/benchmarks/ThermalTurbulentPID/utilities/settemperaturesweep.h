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
//! \file PSMWrapperSweepsCPU.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "domain_decomposition/StructuredBlockStorage.h"

#include "field/GhostLayerField.h"

#include "lbm/sweeps/StreamPull.h"
#include "lbm/sweeps/SweepBase.h"

#include "lbm_mesapd_coupling/DataTypesCodegen.h"
#include "lbm_mesapd_coupling/utility/ParticleFunctions.h"

#include "mesa_pd/common/ParticleFunctions.h"

#include "timeloop/SweepTimeloop.h"

#include <cassert>
#include "temperatureWrapperKernels.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace gpu
{

// GPU


using particleTemperaturesField_T  = GhostLayerField< real_t, MaxParticlesPerCell*1 >;

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT

using particleTemperaturesFieldGPU_T        = walberla::gpu::GPUField< real_t >;
#endif

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
template< typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class SetParticleTemperaturesSweepp
{
 public:
   SetParticleTemperaturesSweepp(const shared_ptr< StructuredBlockStorage >& bs,
                                const shared_ptr< ParticleAccessor_T >& ac,
                                const ParticleSelector_T& mappingParticleSelector,
                                ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                                BlockDataID & densityConcentrationFieldCPUGPUID,BlockDataID &particleTemperaturesFieldCPUGPUID, bool uniformParticleTemperature = false)
      : bs_(bs), ac_(ac), mappingParticleSelector_(mappingParticleSelector),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA), densityConcentrationFieldCPUGPUID_(densityConcentrationFieldCPUGPUID),
        particleTemperatureFieldCPUGPUID_(particleTemperaturesFieldCPUGPUID),
        uniformParticleTemperature_(uniformParticleTemperature)
   {}
   void operator()(IBlock* block)
   {
      // Check that uids of the particles have not changed since the last mapping to avoid incorrect indices
      std::vector< walberla::id_t > currentUIDs;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { currentUIDs.push_back(ac_->getUid(idx)); }
      }
      WALBERLA_ASSERT(particleAndVolumeFractionSoA_.mappingUIDs == currentUIDs);

      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }

      if (numMappedParticles == uint_t(0)) return;

      size_t arraySizes = numMappedParticles * sizeof(real_t) * 1;

      // Allocate unified memory for the particle information required for computing the temperature (used in
      // the solid collision operator for temperature field (only for the case where there is dirichlet boundary for particles temperatures)
      real_t* temperatures_h = (real_t*) malloc(arraySizes);
      memset(temperatures_h, 0, arraySizes);

      // Store particle information inside memory
      size_t idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {
            temperatures_h[idxMapped] = ac_->getTemperature(idx);
            idxMapped++;
         }
      }

      real_t* temperatures;
      WALBERLA_GPU_CHECK(gpuMalloc(&temperatures, arraySizes));
      WALBERLA_GPU_CHECK(gpuMemcpy(temperatures, temperatures_h, arraySizes, gpuMemcpyHostToDevice));
      auto nOverlappingParticlesField =
         block->getData< nOverlappingParticlesFieldGPU_T >(particleAndVolumeFractionSoA_.nOverlappingParticlesFieldID);
      auto idxField = block->getData< idxFieldGPU_T >(particleAndVolumeFractionSoA_.idxFieldID);
      auto particleTemperaturesField =
         block->getData< particleTemperaturesFieldGPU_T >(particleTemperatureFieldCPUGPUID_);
      auto BsField =
         block->getData< BsFieldGPU_T >(particleAndVolumeFractionSoA_.BsFieldID);

      // For every cell, compute the particle velocities of the overlapping particles evaluated at the cell center
      auto temperaturesKernel = walberla::gpu::make_kernel(&(SetParticleTemperatures));
      temperaturesKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< uint_t >::xyz(*nOverlappingParticlesField));
      temperaturesKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< id_t >::xyz(*idxField));
      temperaturesKernel.addFieldIndexingParam(walberla::gpu::FieldIndexing< real_t >::xyz(*particleTemperaturesField));
      temperaturesKernel.addParam(temperatures);

      const double3 blockStart = { block->getAABB().minCorner()[0], block->getAABB().minCorner()[1],
                                   block->getAABB().minCorner()[2] };
      //temperaturesKernel.addParam(blockStart);
      //temperaturesKernel.addParam(block->getAABB().xSize() / real_t(nOverlappingParticlesField->xSize()));
      temperaturesKernel();
      WALBERLA_GPU_CHECK(gpuFree(temperatures));
      free(temperatures_h);;

   }

 private:
   shared_ptr< StructuredBlockStorage > bs_;
   const shared_ptr< ParticleAccessor_T > ac_;
   const ParticleSelector_T& mappingParticleSelector_;
   ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA_;
   const BlockDataID &densityConcentrationFieldCPUGPUID_;
   const BlockDataID &particleTemperatureFieldCPUGPUID_;
   const bool uniformParticleTemperature_;
};

#else

// CPU
template< typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class SetParticleTemperaturesSweepp
{
 public:
   SetParticleTemperaturesSweepp(const shared_ptr< StructuredBlockStorage >& bs,
                                 const shared_ptr< ParticleAccessor_T >& ac,
                                 const ParticleSelector_T& mappingParticleSelector,
                                 ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                                 BlockDataID& densityConcentrationFieldCPUGPUID,BlockDataID &particleTemperaturesFieldCPUGPUID,
                                 bool uniformParticleTemperature = false)
      : bs_(bs), ac_(ac), mappingParticleSelector_(mappingParticleSelector),
        particleAndVolumeFractionSoA_(particleAndVolumeFractionSoA),
        densityConcentrationFieldCPUGPUID_(densityConcentrationFieldCPUGPUID),
        particleTemperatureFieldCPUGPUID_(particleTemperaturesFieldCPUGPUID),
        uniformParticleTemperature_(uniformParticleTemperature)
   {}
   void operator()(IBlock* block)
   {
      // Check that uids of the particles have not changed since the last mapping to avoid incorrect indices
      std::vector< walberla::id_t > currentUIDs;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { currentUIDs.push_back(ac_->getUid(idx)); }
      }
      WALBERLA_ASSERT(particleAndVolumeFractionSoA_.mappingUIDs == currentUIDs);

      size_t numMappedParticles = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_)) { numMappedParticles++; }
      }

      if (numMappedParticles == uint_t(0)) return;

      size_t arraySizes = numMappedParticles * sizeof(real_t) * 1;

      // Allocate unified memory for the particle information required for computing the temperature (used in
      // the solid collision operator for temperature field)
      real_t* temperatures = (real_t*) malloc(arraySizes);
      memset(temperatures, 0, arraySizes);

      // Store particle information inside memory
      size_t idxMapped = 0;
      for (size_t idx = 0; idx < ac_->size(); ++idx)
      {
         if (mappingParticleSelector_(idx, *ac_))
         {

            temperatures[idxMapped] = ac_->getTemperature(idx);
            idxMapped++;
         }
      }

      auto nOverlappingParticlesField =
         block->getData< nOverlappingParticlesField_T >(particleAndVolumeFractionSoA_.nOverlappingParticlesFieldID);
      auto idxField = block->getData< idxField_T >(particleAndVolumeFractionSoA_.idxFieldID);
      auto particleTemperaturesField =
         block->getData< particleTemperaturesField_T >(particleTemperatureFieldCPUGPUID_);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         particleTemperaturesField,

         for (uint_t p = 0; p < nOverlappingParticlesField->get(x, y, z);
              p++) { particleTemperaturesField->get(x, y, z, p) = temperatures[idxField->get(x, y, z, p)]; })

      free(temperatures);
   }

 private:
   shared_ptr< StructuredBlockStorage > bs_;
   const shared_ptr< ParticleAccessor_T > ac_;
   const ParticleSelector_T& mappingParticleSelector_;
   ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA_;
   const BlockDataID& densityConcentrationFieldCPUGPUID_;
   const BlockDataID &particleTemperatureFieldCPUGPUID_;
   const bool uniformParticleTemperature_;
};

#endif

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
