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

#include <mesa_pd/sorting/HilbertCompareFunctor.h>
#include <mesa_pd/sorting/LinearizedCompareFunctor.h>
#include <mesa_pd/data/ParticleStorage.h>

#include <core/Abort.h>
#include <core/Environment.h>
#include <core/math/Random.h>
#include <core/grid_generator/SCIterator.h>
#include <core/logging/Logging.h>

namespace walberla {
namespace mesa_pd {

int main()
{
   auto ps = std::make_shared<data::ParticleStorage>(100);

   const math::AABB domain(0_r, 0_r, 0_r,
                           2_r, 2_r, 2_r);
   for (auto pt : grid_generator::SCGrid(domain, Vec3(real_c(0.5)), 1_r))
   {
      auto p                       = ps->create();
      p->getPositionRef()          = pt;
   }

   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[0].getPosition(), Vec3(0.5_r, 0.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[1].getPosition(), Vec3(1.5_r, 0.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[2].getPosition(), Vec3(0.5_r, 1.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[3].getPosition(), Vec3(1.5_r, 1.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[4].getPosition(), Vec3(0.5_r, 0.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[5].getPosition(), Vec3(1.5_r, 0.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[6].getPosition(), Vec3(0.5_r, 1.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[7].getPosition(), Vec3(1.5_r, 1.5_r, 1.5_r));

   sorting::HilbertCompareFunctor hilbert(domain, 2);
   ps->sort(hilbert);

   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[0].getPosition(), Vec3(0.5_r, 0.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[1].getPosition(), Vec3(1.5_r, 0.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[2].getPosition(), Vec3(1.5_r, 1.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[3].getPosition(), Vec3(0.5_r, 1.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[4].getPosition(), Vec3(0.5_r, 1.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[5].getPosition(), Vec3(1.5_r, 1.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[6].getPosition(), Vec3(1.5_r, 0.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[7].getPosition(), Vec3(0.5_r, 0.5_r, 1.5_r));

   sorting::LinearizedCompareFunctor linear(domain, Vector3<uint_t>(2,2,2));
   ps->sort(linear);

   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[0].getPosition(), Vec3(0.5_r, 0.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[1].getPosition(), Vec3(1.5_r, 0.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[2].getPosition(), Vec3(0.5_r, 1.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[3].getPosition(), Vec3(1.5_r, 1.5_r, 0.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[4].getPosition(), Vec3(0.5_r, 0.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[5].getPosition(), Vec3(1.5_r, 0.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[6].getPosition(), Vec3(0.5_r, 1.5_r, 1.5_r));
   WALBERLA_CHECK_FLOAT_EQUAL((*ps)[7].getPosition(), Vec3(1.5_r, 1.5_r, 1.5_r));

   return EXIT_SUCCESS;
}

} // namespace mesa_pd
} // namespace walberla

int main( int argc, char* argv[] )
{
   walberla::Environment env(argc, argv);
   return walberla::mesa_pd::main();
}
