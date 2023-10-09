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
//! \file PSMSweepCollection.h
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/PSMWrapperSweepsGPU.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/ParticleAndVolumeFractionMappingGPU.h"

namespace walberla
{
namespace lbm_mesapd_coupling
{
namespace psm
{
namespace gpu
{

template< typename ParticleAccessor_T, typename ParticleSelector_T, int Weighting_T >
class PSMSweepCollectionGPU
{
 public:
   PSMSweepCollectionGPU(const shared_ptr< StructuredBlockStorage >& bs, const shared_ptr< ParticleAccessor_T >& ac,
                         const ParticleSelector_T& ps,
                         ParticleAndVolumeFractionSoA_T< Weighting_T >& particleAndVolumeFractionSoA,
                         const size_t numberOfParticleSubBlocksPerDim)
      : particleMappingSweep(ParticleAndVolumeFractionMappingGPU< ParticleAccessor_T, ParticleSelector_T, Weighting_T >(
           bs, ac, ps, particleAndVolumeFractionSoA, numberOfParticleSubBlocksPerDim)),
        setParticleVelocitiesSweep(SetParticleVelocitiesSweep< ParticleAccessor_T, ParticleSelector_T, Weighting_T >(
           bs, ac, ps, particleAndVolumeFractionSoA)),
        reduceParticleForcesSweep(ReduceParticleForcesSweep< ParticleAccessor_T, ParticleSelector_T, Weighting_T >(
           bs, ac, ps, particleAndVolumeFractionSoA))
   {}

   ParticleAndVolumeFractionMappingGPU< ParticleAccessor_T, ParticleSelector_T, Weighting_T > particleMappingSweep;
   SetParticleVelocitiesSweep< ParticleAccessor_T, ParticleSelector_T, Weighting_T > setParticleVelocitiesSweep;
   ReduceParticleForcesSweep< ParticleAccessor_T, ParticleSelector_T, Weighting_T > reduceParticleForcesSweep;
};

template< typename SweepCollection, typename PSMSweep >
void addPSMSweepsToTimeloop(SweepTimeloop& timeloop, SweepCollection& psmSweepCollection, PSMSweep& psmSweep,
                            bool synchronize = true)
{
   if (synchronize)
   {
      timeloop.add() << Sweep(psmSweepCollection.particleMappingSweep, "Particle mapping");
      timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollection.setParticleVelocitiesSweep),
                              "Set particle velocities");
      timeloop.add() << Sweep(deviceSyncWrapper(psmSweep), "PSM sweep");
      timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollection.reduceParticleForcesSweep),
                              "Reduce particle forces");
   }
   else
   {
      timeloop.add() << Sweep(psmSweepCollection.particleMappingSweep, "Particle mapping");
      timeloop.add() << Sweep(psmSweepCollection.setParticleVelocitiesSweep, "Set particle velocities");
      timeloop.add() << Sweep(psmSweep, "PSM sweep");
      timeloop.add() << Sweep(psmSweepCollection.reduceParticleForcesSweep, "Reduce particle forces");
   };
}

template< typename SweepCollection, typename PSMSweep, typename Communication >
void addPSMSweepsToTimeloops(SweepTimeloop& commTimeloop, SweepTimeloop& timeloop, Communication& comm,
                             SweepCollection& psmSweepCollection, PSMSweep& psmSweep, bool synchronize = true)
{
   if (synchronize)
   {
      commTimeloop.add() << BeforeFunction([&]() { comm.startCommunication(); })
                         << Sweep(deviceSyncWrapper(psmSweepCollection.particleMappingSweep), "Particle mapping");
      commTimeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollection.setParticleVelocitiesSweep),
                                  "Set particle velocities");
      commTimeloop.add() << Sweep(deviceSyncWrapper([&](IBlock* block) { psmSweep.inner(block); }), "PSM inner sweep")
                         << AfterFunction([&]() { comm.wait(); }, "LBM Communication (wait)");
      timeloop.add() << Sweep(deviceSyncWrapper([&](IBlock* block) { psmSweep.outer(block); }), "PSM outer sweep");
      timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollection.reduceParticleForcesSweep),
                              "Reduce particle forces");
   }
   else
   {
      commTimeloop.add() << BeforeFunction([&]() { comm.startCommunication(); })
                         << Sweep(psmSweepCollection.particleMappingSweep, "Particle mapping");
      commTimeloop.add() << Sweep(psmSweepCollection.setParticleVelocitiesSweep, "Set particle velocities");
      commTimeloop.add() << Sweep([&](IBlock* block) { psmSweep.inner(block); }, "PSM inner sweep")
                         << AfterFunction([&]() { comm.wait(); }, "LBM Communication (wait)");
      timeloop.add() << Sweep([&](IBlock* block) { psmSweep.outer(block); }, "PSM outer sweep");
      timeloop.add() << Sweep(psmSweepCollection.reduceParticleForcesSweep, "Reduce particle forces");
   };
}

} // namespace gpu
} // namespace psm
} // namespace lbm_mesapd_coupling
} // namespace walberla
