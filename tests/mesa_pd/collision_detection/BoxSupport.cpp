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
#include <mesa_pd/data/shape/Box.h>

#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <cmath>
#include <iostream>

namespace walberla {
namespace mesa_pd {

void checkContour(const Vec3& pt, const Vec3& edgeLength )
{
   WALBERLA_CHECK_FLOAT_EQUAL(std::abs(pt[0]), edgeLength[0] * 0.5_r, pt);
   WALBERLA_CHECK_FLOAT_EQUAL(std::abs(pt[1]), edgeLength[1] * 0.5_r, pt);
   WALBERLA_CHECK_FLOAT_EQUAL(std::abs(pt[2]), edgeLength[2] * 0.5_r, pt);
}

void check( )
{
   using namespace walberla::mesa_pd::collision_detection;

   Vec3 pos        = Vec3(1_r,2_r,3_r);
   Vec3 edgeLength = Vec3(2_r,3_r,1_r);

   auto bx = data::Box(edgeLength);
   Support b0(pos, Rot3(), bx);
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(real_t(+1),real_t(+1),real_t(+1))),  Vec3(2_r,3.5_r,3.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(-1_r,real_t(+1),real_t(+1))),  Vec3(0_r,3.5_r,3.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(real_t(+1),-1_r,real_t(+1))),  Vec3(2_r,0.5_r,3.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(-1_r,-1_r,real_t(+1))),  Vec3(0_r,0.5_r,3.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(real_t(+1),real_t(+1),-1_r)),  Vec3(2_r,3.5_r,2.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(-1_r,real_t(+1),-1_r)),  Vec3(0_r,3.5_r,2.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(real_t(+1),-1_r,-1_r)),  Vec3(2_r,0.5_r,2.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL(b0.support(Vec3(-1_r,-1_r,-1_r)),  Vec3(0_r,0.5_r,2.5_r));

   checkContour(b0.support(Vec3(-1_r,-2_r,-3_r).getNormalized()) - pos, edgeLength );
   checkContour(b0.support(Vec3(-2_r,-3_r,-1_r).getNormalized()) - pos, edgeLength );
   checkContour(b0.support(Vec3(-3_r,-1_r,-2_r).getNormalized()) - pos, edgeLength );

   Rot3 rot = Rot3(Vec3(1,3,2).getNormalized(), 1.56_r);
   Support b1(pos, rot, bx);
   checkContour(rot.getMatrix().getTranspose() * (b1.support(Vec3(-1_r,-2_r,-3_r).getNormalized()) - pos), edgeLength );
   checkContour(rot.getMatrix().getTranspose() * (b1.support(Vec3(-2_r,-3_r,-1_r).getNormalized()) - pos), edgeLength );
   checkContour(rot.getMatrix().getTranspose() * (b1.support(Vec3(-3_r,-1_r,-2_r).getNormalized()) - pos), edgeLength );
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
