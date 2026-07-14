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
#include <mesa_pd/data/shape/Sphere.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <cmath>
#include <iostream>

namespace walberla {
namespace mesa_pd {

inline
real_t contour(const Vec3& point, const real_t& radius)
{
   return ((point[0] * point[0]) + (point[1] * point[1]) + (point[2] * point[2])) / (radius * radius);
}

void check( )
{
   using namespace walberla::mesa_pd::collision_detection;

   Vec3 pos      = Vec3(1_r,2_r,3_r);
   real_t radius = 2.356_r;

   auto sp = data::Sphere(radius);
   Support e0(pos, Rot3(), sp);
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(1_r,0_r,0_r)),  Vec3(real_t(1+2.356),2_r,3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,1_r,0_r)),  Vec3(1_r,real_t(2+2.356),3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,0_r,1_r)),  Vec3(1_r,2_r,real_t(3+2.356)));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(-1_r,0_r,0_r)), Vec3(real_t(1-2.356),2_r,3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,-1_r,0_r)), Vec3(1_r,real_t(2-2.356),3_r));
   WALBERLA_CHECK_FLOAT_EQUAL(e0.support(Vec3(0_r,0_r,-1_r)), Vec3(1_r,2_r,real_t(3-2.356)));

   WALBERLA_CHECK_FLOAT_EQUAL( contour(e0.support(Vec3(-1_r,-2_r,-3_r).getNormalized()) - pos, radius), 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( contour(e0.support(Vec3(-2_r,-3_r,-1_r).getNormalized()) - pos, radius), 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( contour(e0.support(Vec3(-3_r,-1_r,-2_r).getNormalized()) - pos, radius), 1_r );
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
