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

#include <mesa_pd/collision_detection/AnalyticCollisionFunctions.h>
#include <mesa_pd/data/DataTypes.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <cmath>
#include <iostream>

namespace walberla {
namespace mesa_pd {

void checkSphereSphereCollision( )
{
   using namespace walberla::mesa_pd::collision_detection::analytic;

   const real_t radius  = 0.5_r;
   auto dir = Vec3(1_r, 2_r, 3_r).getNormalized();
   real_t shift = real_c(0.75);

   Vec3   contactPoint;
   Vec3   contactNormal;
   real_t penetrationDepth;

   //check two spheres in contact
   WALBERLA_CHECK( detectSphereSphereCollision( Vec3(0_r, 0_r, 0_r),
                                                radius,
                                                dir * shift,
                                                radius,
                                                contactPoint,
                                                contactNormal,
                                                penetrationDepth,
                                                0_r ) );
   WALBERLA_CHECK_FLOAT_EQUAL( contactPoint, dir * shift * 0.5_r);
   WALBERLA_CHECK_FLOAT_EQUAL( contactNormal, -dir );
   WALBERLA_CHECK_FLOAT_EQUAL( penetrationDepth, shift - 1_r );

   //no collision
   WALBERLA_CHECK( !detectSphereSphereCollision( Vec3(0_r, 0_r, 0_r),
                                                 radius,
                                                 dir * 1.1_r,
                                                 radius,
                                                 contactPoint,
                                                 contactNormal,
                                                 penetrationDepth,
                                                 0_r ) );
}

void checkSphereCylindricalBoundaryCollision( )
{
   using namespace walberla::mesa_pd::collision_detection::analytic;

   const real_t radius  = 0.5_r;
   auto pos = Vec3(6_r, -2_r, 2_r);
   auto dir = Vec3(1_r, 1_r, 1_r).getNormalized();

   Vec3   contactPoint;
   Vec3   contactNormal;
   real_t penetrationDepth;

   //check two spheres in contact
   WALBERLA_CHECK( detectSphereCylindricalBoundaryCollision( pos,
                                                             radius,
                                                             Vec3(0_r, 0_r, 0_r),
                                                             5_r,
                                                             dir,
                                                             contactPoint,
                                                             contactNormal,
                                                             penetrationDepth,
                                                             0_r ) );
   auto nearestPoint = dot(pos, dir) * dir;
   WALBERLA_CHECK_FLOAT_EQUAL( contactPoint, (pos - nearestPoint).getNormalized() * 5_r + nearestPoint);
   WALBERLA_CHECK_FLOAT_EQUAL( contactNormal, (nearestPoint - pos).getNormalized() );
   WALBERLA_CHECK_FLOAT_EQUAL( penetrationDepth, -((pos - nearestPoint).length() + radius - 5_r) );

   //no collision
   WALBERLA_CHECK( !detectSphereCylindricalBoundaryCollision( Vec3(0_r, 0_r, 0_r),
                                                              radius,
                                                              Vec3(0_r, 0_r, 0_r),
                                                              5_r,
                                                              Vec3(1_r, 1_r, 1_r).getNormalized(),
                                                              contactPoint,
                                                              contactNormal,
                                                              penetrationDepth,
                                                              0_r ) );
}

void checkSphereBoxCollision( )
{
   using namespace walberla::mesa_pd::collision_detection::analytic;

   Vec3   contactPoint;
   Vec3   contactNormal;
   real_t penetrationDepth;

   //check two spheres in contact
   WALBERLA_CHECK( detectSphereBoxCollision( Vec3(2_r, 2_r, 2_r),
                                             real_c(std::sqrt(3)) + 0.1_r,
                                             Vec3(0_r, 0_r, 0_r),
                                             Vec3(2_r, 2_r, 2_r),
                                             Rot3(),
                                             contactPoint,
                                             contactNormal,
                                             penetrationDepth,
                                             0_r ) );
   WALBERLA_CHECK_FLOAT_EQUAL( contactPoint, Vec3(1_r, 1_r, 1_r));
   WALBERLA_CHECK_FLOAT_EQUAL( contactNormal, Vec3(2_r, 2_r, 2_r).getNormalized() );
   WALBERLA_CHECK_FLOAT_EQUAL( penetrationDepth, -0.1_r );
}

void checkSphereHalfSpaceCollision( )
{
   using namespace walberla::mesa_pd::collision_detection::analytic;

   const real_t radius  = 1.0_r;
   auto dir = Vec3(1_r, 2_r, 3_r).getNormalized();
   real_t shift = real_c(0.75);

   Vec3   contactPoint;
   Vec3   contactNormal;
   real_t penetrationDepth;

   //check sphere - half space contact
   WALBERLA_CHECK( detectSphereHalfSpaceCollision( Vec3(0_r, 0_r, 0_r),
                                                   radius,
                                                   dir * shift,
                                                   -dir,
                                                   contactPoint,
                                                   contactNormal,
                                                   penetrationDepth,
                                                   0_r ) );
   WALBERLA_CHECK_FLOAT_EQUAL( contactPoint, dir * shift * 1.0_r);
   WALBERLA_CHECK_FLOAT_EQUAL( contactNormal, -dir );
   WALBERLA_CHECK_FLOAT_EQUAL( penetrationDepth, shift - 1_r );


   auto pos = Vec3(shift, 0_r, 0_r);
   WALBERLA_CHECK( detectSphereHalfSpaceCollision( Vec3(0_r, 0_r, 0_r),
                                                   radius,
                                                   pos,
                                                   -Vec3(1_r, 0_r, 0_r),
                                                   contactPoint,
                                                   contactNormal,
                                                   penetrationDepth,
                                                   0_r ) );
   WALBERLA_CHECK_FLOAT_EQUAL( contactPoint, pos);
   WALBERLA_CHECK_FLOAT_EQUAL( contactNormal, -Vec3(1_r, 0_r, 0_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( penetrationDepth, pos[0] - 1_r );
}

} //namespace mesa_pd
} //namespace walberla

int main( int argc, char ** argv )
{
   walberla::Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   walberla::mesa_pd::checkSphereSphereCollision();
   walberla::mesa_pd::checkSphereCylindricalBoundaryCollision();
   walberla::mesa_pd::checkSphereBoxCollision();
   walberla::mesa_pd::checkSphereHalfSpaceCollision();

   return EXIT_SUCCESS;
}
