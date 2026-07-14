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

#include <mesa_pd/domain/BlockForestDomain.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <iostream>

namespace walberla {

using namespace walberla::mesa_pd;

void main( int argc, char ** argv )
{
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);
   mpi::MPIManager::instance()->useWorldComm();

   using namespace walberla::mesa_pd;

   WALBERLA_CHECK_FLOAT_EQUAL( sqDistanceLineToPoint(0.7_r, 0.9_r, 1.2_r) + 1_r,
                               0.04_r + 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( sqDistanceLineToPoint(1.0_r, 0.9_r, 1.2_r) + 1_r,
                               0_r + 1_r );
   WALBERLA_CHECK_FLOAT_EQUAL( sqDistanceLineToPoint(1.5_r, 0.9_r, 1.2_r) + 1_r,
                               0.09_r + 1_r );

   WALBERLA_CHECK_FLOAT_EQUAL( sqDistancePointToAABB( Vec3(1,1,1),
                                                      math::AABB(2_r,2_r,2_r,3_r,3_r,3_r) ) + 1_r,
                               3_r + 1_r );

   WALBERLA_CHECK_FLOAT_EQUAL( sqDistancePointToAABB( Vec3(0.5_r,0.5_r,0.5_r),
                                                      math::AABB(2_r,2_r,2_r,3_r,3_r,3_r) ) + 1_r,
                               6.75_r + 1_r );

   WALBERLA_CHECK_FLOAT_EQUAL( sqDistancePointToAABBPeriodic( Vec3(0.5_r,0.5_r,0.5_r),
                                                              math::AABB(2_r,2_r,2_r,3_r,3_r,3_r),
                                                              math::AABB(1_r,1_r,1_r,3_r,3_r,3_r),
                                                              std::array< bool, 3 >{{false, false, false}}) + 1_r,
                               6.75_r + 1_r );

   WALBERLA_CHECK_FLOAT_EQUAL( sqDistancePointToAABBPeriodic( Vec3(0.5_r,0.5_r,0.5_r),
                                                              math::AABB(2_r,2_r,2_r,3_r,3_r,3_r),
                                                              math::AABB(1_r,1_r,1_r,3_r,3_r,3_r),
                                                              std::array< bool, 3 >{{true, true, true}}) + 1_r,
                               0_r + 1_r );

   WALBERLA_CHECK_FLOAT_EQUAL( sqDistancePointToAABBPeriodic( Vec3(1.1_r,1.1_r,1.1_r),
                                                              math::AABB(2_r,2_r,2_r,3_r,3_r,3_r),
                                                              math::AABB(1_r,1_r,1_r,3_r,3_r,3_r),
                                                              std::array< bool, 3 >{{true, true, true}}) + 1_r,
                               0.03_r + 1_r );
}

}

int main( int argc, char ** argv )
{
   walberla::main(argc, argv);

   return EXIT_SUCCESS;
}
