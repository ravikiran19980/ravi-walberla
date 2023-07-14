//
// Created by RichardAngersbach on 16.01.2023.
//

#ifndef WALBERLA_CHARGEFORCE_H
#define WALBERLA_CHARGEFORCE_H

#include "stencil/D3Q7.h"

namespace walberla
{
using Stencil_T = stencil::D3Q7;
typedef GhostLayerField< real_t, 1 > ScalarField_T;
typedef GhostLayerField< real_t, 3 > VectorField_T;

template< typename ParticleAccessor_T >
class ChargeForceUpdate
{
 public:
    ChargeForceUpdate(const std::shared_ptr<StructuredBlockForest> &blocks,
                      const BlockDataID &potential, const BlockDataID &chargeForce,
                      const BlockDataID &particleAndVolumeFractionFieldID,
                      const BlockDataID &chargeDensityFieldID,
                      const shared_ptr<ParticleAccessor_T> &ac, const real_t epsilon)

                      : blocks_(blocks), \
                      potential_(potential), chargeForce_(chargeForce), \
                      particleAndVolumeFractionFieldID_(particleAndVolumeFractionFieldID), \
                      chargeDensityFieldID_(chargeDensityFieldID), \
                      accessor_(ac), epsilon_(epsilon) {}

   void operator()()
   {
      // get charge force with FD gradient from electric potential
      for (auto block = blocks_->begin(); block != blocks_->end(); ++block)
      {
         VectorField_T* chargeForce = block->getData< VectorField_T >(chargeForce_);
         ScalarField_T* potential   = block->getData< ScalarField_T >(potential_);

         lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T* particleAndVolumeFractionField =
            block->getData< lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T >(
               particleAndVolumeFractionFieldID_);

         GhostLayerField< real_t, 1 >* chargeDensityField =
            block->template getData< GhostLayerField< real_t, 1 > >(chargeDensityFieldID_); // i added this

         WALBERLA_FOR_ALL_CELLS_XYZ(
            potential, chargeForce->get(x, y, z, 0) = -(real_c(1) / (real_c(2) * (blocks_->dx()))) *
                                                      (potential->get(x + 1, y, z) - potential->get(x - 1, y, z)) *
                                                      chargeDensityField->get(x, y, z) * 1 * epsilon_;

            chargeForce->get(x, y, z, 1) = -(real_c(1) / (real_c(2) * (blocks_->dy()))) *
                                           (potential->get(x, y + 1, z) - potential->get(x, y - 1, z)) *
                                           chargeDensityField->get(x, y, z) * 1 * epsilon_;

            chargeForce->get(x, y, z, 2) = -(real_c(1) / (real_c(2) * (blocks_->dz()))) *
                                           (potential->get(x, y, z + 1) - potential->get(x, y, z - 1)) *
                                           chargeDensityField->get(x, y, z) * 1 * epsilon_;

            if (particleAndVolumeFractionField->get(x, y, z).size() != size_t(0)) {
               for (auto particleFracIt = particleAndVolumeFractionField->get(x, y, z).begin();
                    particleFracIt != particleAndVolumeFractionField->get(x, y, z).end(); ++particleFracIt)
               {
                  Vector3< real_t > chargeforceOnParticle(real_c(0));
                  const size_t idx = accessor_->uidToIdx(particleFracIt->first);
                  WALBERLA_ASSERT_UNEQUAL(idx, accessor_->getInvalidIdx(), "Index of particle is invalid!");

                  // add the electrostatic force on particle here.
                  chargeforceOnParticle[0] =
                     chargeForce->get(x, y, z, 0); // Force(x,y,z) = Electric field(x,y,z) * charge in the cell(x,y,z)

                  chargeforceOnParticle[1] = chargeForce->get(
                     x, y, z, 1); // charge in cell(x,y,z) = charge density in cell(x,y,z)* volume of cell

                  chargeforceOnParticle[2] = chargeForce->get(x, y, z, 2); // volume of cell = 1 (lattice units)

                  // scaling to be done like lbm in psmsweep.h?
                  // std::cout <<  chargeforceOnParticle[0] <<" "  << chargeforceOnParticle[1] << " "<<
                  // chargeforceOnParticle[2] << std::endl;
                  lbm_mesapd_coupling::addElectrostaticForceAtomic(idx, *accessor_, chargeforceOnParticle);
               }
            }

         )
      }
   }

 private:
    std::shared_ptr<StructuredBlockForest> blocks_;

    BlockDataID potential_;
    BlockDataID chargeForce_;
    const BlockDataID particleAndVolumeFractionFieldID_;
    const BlockDataID chargeDensityFieldID_;

    const shared_ptr<ParticleAccessor_T> accessor_;

    const real_t epsilon_;
};

} // namespace walberla

#endif // WALBERLA_CHARGEFORCE_H
