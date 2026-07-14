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
//! \file
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

#include <mesa_pd/collision_detection/Support.h>
#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/shape/Ellipsoid.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <cmath>
#include <iostream>

namespace walberla {
namespace mesa_pd {

inline
real_t contour(const Vec3& point, const Vec3& semiAxes)
{
   return (point[0] * point[0]) / (semiAxes[0] * semiAxes[0]) +
          (point[1] * point[1]) / (semiAxes[1] * semiAxes[1]) +
          (point[2] * point[2]) / (semiAxes[2] * semiAxes[2]);
}

void check( )
{
   using namespace walberla::mesa_pd::collision_detection;

   Vec3 pos      = Vec3(1_r,2_r,3_r);
   Vec3 semiAxes = Vec3(2_r,3_r,1_r);

   auto el = data::Ellipsoid(semiAxes);
   Support e0(pos, Rot3(), el);
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(1_r,0_r,0_r)), Vec3(3_r,2_r,3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,1_r,0_r)), Vec3(1_r,5_r,3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,0_r,1_r)), Vec3(1_r,2_r,4_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(-1_r,0_r,0_r)), Vec3(-1_r,2_r,3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,-1_r,0_r)), Vec3(1_r,-1_r,3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,0_r,-1_r)), Vec3(1_r,2_r,2_r));

   WALBERLA_CHECK_FLOAT_EQUAL( contour(e0.support(Vec3(-1_r,-2_r,-3_r).getNormalized()) - pos, semiAxes), 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( contour(e0.support(Vec3(-2_r,-3_r,-1_r).getNormalized()) - pos, semiAxes), 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( contour(e0.support(Vec3(-3_r,-1_r,-2_r).getNormalized()) - pos, semiAxes), 1_r );

   Rot3 rot = Rot3(Vec3(1,3,2).getNormalized(), 1.56_r);
   Support e1(pos, rot, el);
   WALBERLA_CHECK_FLOAT_EQUAL( contour( rot.getMatrix().getTranspose() * (e1.support(Vec3(-1_r,-2_r,-3_r).getNormalized()) - pos), semiAxes), 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( contour( rot.getMatrix().getTranspose() * (e1.support(Vec3(-2_r,-3_r,-1_r).getNormalized()) - pos), semiAxes), 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( contour( rot.getMatrix().getTranspose() * (e1.support(Vec3(-3_r,-1_r,-2_r).getNormalized()) - pos), semiAxes), 1_r );
}

} //namespace mesa_pd
} //namespace walberla

int main( int argc, char ** argv )
{
   walberla::Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   walberla::mesa_pd::check();

   return EXIT_SUCCESS;
}
