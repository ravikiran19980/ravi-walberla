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

#include <mesa_pd/collision_detection/GeneralContactDetection.h>
#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>
#include <mesa_pd/kernel/DoubleCast.h>

#include <core/Abort.h>
#include <core/Environment.h>

#include <memory>

namespace walberla {
namespace mesa_pd {

void generalContactDetection()
{
   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(100);
   auto ss = std::make_shared<data::ShapeStorage>();
   data::ParticleAccessorWithShape accessor(ps, ss);

   auto e0 = ps->create();
   e0->setPosition(Vec3(0_r,0_r,0_r));
   e0->setShapeID(ss->create<data::Ellipsoid>(Vec3(1_r,2_r,3_r)));

   auto e1 = ps->create();
   e1->setPosition(Vec3(1.9_r,0_r,0_r));
   e1->setShapeID(ss->create<data::Ellipsoid>(Vec3(1_r,2_r,3_r)));

   auto p1 = ps->create();
   p1->setPosition(Vec3(-0.9_r,0_r,0_r));
   p1->setShapeID(ss->create<data::HalfSpace>(Vec3(1_r,0_r,0_r)));

   auto cb1 = ps->create();
   cb1->setPosition(Vec3(0_r,0_r,0_r));
   cb1->setShapeID(ss->create<data::CylindricalBoundary>(3_r, Vec3(0_r,0_r,1_r)));

   collision_detection::GeneralContactDetection gcd;
   kernel::DoubleCast double_cast;

   WALBERLA_CHECK(double_cast(0, 1, accessor, gcd, accessor));
   WALBERLA_CHECK_FLOAT_EQUAL( gcd.getContactPoint(), Vec3(0.95_r,0_r,0_r) );
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( gcd.getContactNormal(), Vec3(-1_r,0_r,0_r), 1e-3_r );
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( gcd.getPenetrationDepth(), -0.1_r, 1e-3_r  );
   e1->setPosition(Vec3(2.1_r,0_r,0_r));
   WALBERLA_CHECK(!double_cast(0, 1, accessor, gcd, accessor));

   WALBERLA_CHECK(double_cast(0, 2, accessor, gcd, accessor));
   WALBERLA_CHECK_FLOAT_EQUAL( gcd.getContactPoint(), Vec3(-0.95_r,0_r,0_r) );
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( gcd.getContactNormal(), Vec3(-1_r,0_r,0_r), 1e-3_r );
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( gcd.getPenetrationDepth(), -0.1_r, 1e-3_r );
   WALBERLA_CHECK(!double_cast(1, 2, accessor, gcd, accessor));

   WALBERLA_CHECK(double_cast(1, 3, accessor, gcd, accessor));
   WALBERLA_CHECK_FLOAT_EQUAL( gcd.getContactPoint(), Vec3(3.05_r,0_r,0_r) );
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( gcd.getContactNormal(), Vec3(real_t(+1),0_r,0_r), 1e-3_r );
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON( gcd.getPenetrationDepth(), -0.1_r, 1e-3_r );
   WALBERLA_CHECK(!double_cast(0, 3, accessor, gcd, accessor));
}

} // namespace mesa_pd
} // namespace walberla

int main( int argc, char* argv[] )
{
   walberla::Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   walberla::mesa_pd::generalContactDetection();
   return EXIT_SUCCESS;
}
