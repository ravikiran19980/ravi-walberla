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

#include <mesa_pd/collision_detection/GJK.h>
#include <mesa_pd/collision_detection/Support.h>
#include <mesa_pd/data/shape/Sphere.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <cmath>
#include <iostream>
#include <memory>

namespace walberla {
namespace mesa_pd {

void check( )
{
   using namespace walberla::mesa_pd::collision_detection;

   auto sp0 = data::Sphere(0.6_r);
   auto sp1 = data::Sphere(0.4_r);

   GJK gjk;
   WALBERLA_CHECK( gjk.doGJKmargin( Support(Vec3(1_r,2_r,3_r), Rot3(), sp0),
                                    Support(Vec3(2_r,2_r,3_r), Rot3(), sp0) ));
   WALBERLA_CHECK( !gjk.doGJKmargin( Support(Vec3(1_r,2_r,3_r), Rot3(), sp1),
                                     Support(Vec3(2_r,2_r,3_r), Rot3(), sp1) ));
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
