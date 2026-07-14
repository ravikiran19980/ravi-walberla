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
//! \file QCriterion.h
//! \ingroup lbm
//! \author Lukas Werner <lks.werner@fau.de>
//
//======================================================================================================================

#pragma once

#include "core/DataTypes.h"
#include "core/math/Vector3.h"

// Back-end for calculating macroscopic values
// You should never use these functions directly, always refer to the member functions
// of PdfField or the free functions that can be found in MacroscopicValueCalculation.h

namespace walberla
{
namespace lbm
{
struct QCriterion
{
   template< typename VelocityField_T, typename Filter_T >
   static inline real_t get(const VelocityField_T& velocityField, const Filter_T& filter, const cell_idx_t x,
                            const cell_idx_t y, const cell_idx_t z, real_t dx = 1_r, real_t dy = 1_r,
                            real_t dz = 1_r)
   {
      const auto one = cell_idx_t{1};

      auto f(velocityField.flattenedShallowCopy());

      if (filter(x, y, z) && filter(x + one, y, z) && filter(x - one, y, z) && filter(x, y + one, z) &&
          filter(x, y - one, z) && filter(x, y, z + one) && filter(x, y, z - one))
      {
         Vector3< real_t > xa(f->get(x + one, y, z, 0), f->get(x + one, y, z, 1), f->get(x + one, y, z, 2));
         Vector3< real_t > xb(f->get(x - one, y, z, 0), f->get(x - one, y, z, 1), f->get(x - one, y, z, 2));
         Vector3< real_t > ya(f->get(x, y + one, z, 0), f->get(x, y + one, z, 1), f->get(x, y + one, z, 2));
         Vector3< real_t > yb(f->get(x, y - one, z, 0), f->get(x, y - one, z, 1), f->get(x, y - one, z, 2));
         Vector3< real_t > za(f->get(x, y, z + one, 0), f->get(x, y, z + one, 1), f->get(x, y, z + one, 2));
         Vector3< real_t > zb(f->get(x, y, z - one, 0), f->get(x, y, z - one, 1), f->get(x, y, z - one, 2));

         return calculate(xa, xb, ya, yb, za, zb, dx, dy, dz);
      }
      return 0_r;
   }

   static inline real_t calculate(const Vector3< real_t > xa, const Vector3< real_t > xb, const Vector3< real_t > ya,
                                  const Vector3< real_t > yb, const Vector3< real_t > za, const Vector3< real_t > zb,
                                  const real_t dx, const real_t dy, const real_t dz)
   {
      const auto halfInvDx = 0.5_r / dx;
      const auto halfInvDy = 0.5_r / dy;
      const auto halfInvDz = 0.5_r / dz;

      const real_t duxdx = (xa[0] - xb[0]) * halfInvDx;
      const real_t duxdy = (ya[0] - yb[0]) * halfInvDy;
      const real_t duxdz = (za[0] - zb[0]) * halfInvDz;

      const real_t duydx = (xa[1] - xb[1]) * halfInvDx;
      const real_t duydy = (ya[1] - yb[1]) * halfInvDy;
      const real_t duydz = (za[1] - zb[1]) * halfInvDz;

      const real_t duzdx = (xa[2] - xb[2]) * halfInvDx;
      const real_t duzdy = (ya[2] - yb[2]) * halfInvDy;
      const real_t duzdz = (za[2] - zb[2]) * halfInvDz;

      // Q = 1/2 * (||W||² - ||S||²)
      real_t sNormSq = duxdx * duxdx + duydy * duydy + duzdz * duzdz + 0.5_r * (duxdy + duydx) * (duxdy + duydx) +
                       0.5_r * (duydz + duzdy) * (duydz + duzdy) +
                       0.5_r * (duxdz + duzdx) * (duxdz + duzdx);

      real_t omegaNormSq = 0.5_r * (duxdz - duzdx) * (duxdz - duzdx) +
                           0.5_r * (duxdy - duydx) * (duxdy - duydx) +
                           0.5_r * (duydz - duzdy) * (duydz - duzdy);

      return 0.5_r * (omegaNormSq - sNormSq);
   }
};

} // namespace lbm
} // namespace walberla
