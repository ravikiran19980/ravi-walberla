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
//! \file   Utility.h
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//
//======================================================================================================================

#pragma once

#include "lbm_mesapd_coupling/DataTypes.h"

#include <mesa_pd/data/ParticleAccessor.h>
#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>

namespace walberla
{
namespace charged_particles
{
// using uint_t = size_t;
template< typename BlockStorage_T, typename ParticleAccessor_T >
void computeChargeDensity(const shared_ptr< BlockStorage_T >& blocks,
                          const BlockDataID& particleAndVolumeFractionFieldID, const BlockDataID& chargeDensityFieldID,
                          const shared_ptr< ParticleAccessor_T >& ac, const real_t epsilon) // const ParticleAccessor_T& ac)
{
   // TODO: compute physically correct charge density here using the particle charges, have a look at
   // src/lbm_mesapd_coupling/partially_saturated_cells_method/ParticleAndVolumeFractionMapping.h for how to iterate
   // over particles and cells
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T* particleAndVolumeFractionField =
         blockIt->template getData< lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T >(
            particleAndVolumeFractionFieldID);
      GhostLayerField< real_t, 1 >* chargeDensityField =
         blockIt->template getData< GhostLayerField< real_t, 1 > >(chargeDensityFieldID);

      real_t charge         = 0;
      real_t particleVolume = 0;
         WALBERLA_FOR_ALL_CELLS_XYZ(particleAndVolumeFractionField, chargeDensityField->get(x, y, z) = 0.0; real_t cell_fractions_sum = 0.0;
            if (particleAndVolumeFractionField->get(x, y, z).size() != size_t(0)) {
                for (auto &e: particleAndVolumeFractionField->get(x, y, z)) {

                    const size_t idx = ac->uidToIdx(e.first); // passing unique uid of the particle
                    charge = ac->getCharge(idx);    // charges could be different
                    particleVolume =
                            ac->getShape(
                                    idx)->getVolume(); // all the particle have same radius and so volume is same for all idx
                    cell_fractions_sum +=
                            real_c(1 / particleVolume) * charge * e.second; // particle have different charges
                }
                //std::cout << epsilon << std::endl;
                chargeDensityField->get(x, y, z) =
                        cell_fractions_sum / epsilon; // - sign is there because rhs is negative of charge
                // density in poisson equation , correct me if iam wrong
            }
      )


   }
}

} // namespace charged_particles
} // namespace walberla
