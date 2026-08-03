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
#include <mesa_pd/kernel/SemiImplicitEuler.h>
#include <mesa_pd/kernel/ParticleSelector.h>
#include <mesa_pd/kernel/SpringDashpot.h>

#include <core/Abort.h>
#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <memory>
#include <string>
#include <type_traits>

namespace walberla {
namespace mesa_pd {

int main( int argc, char ** argv )
{
   using namespace walberla::timing;

   Environment env(argc, argv);
   walberla::mpi::MPIManager::instance()->useWorldComm();

   if (std::is_same_v<walberla::real_t, float>)
   {
      WALBERLA_LOG_WARNING("waLBerla build in sp mode: skipping test due to low precision");
      return EXIT_SUCCESS;
   }

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(100);
   auto ss = std::make_shared<data::ShapeStorage>();
   mesa_pd::data::ParticleAccessorWithShape accessor(ps, ss);

   auto p                       = ps->create();
   p->getPositionRef()          = Vec3(0_r, 0_r, 0_r);
   p->getInteractionRadiusRef() = std::numeric_limits<real_t>::infinity();
   p->getShapeIDRef()           = ss->create<data::HalfSpace>( Vec3(0_r, 0_r, 1_r) );
   p->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p->getTypeRef()              = 0;
   using namespace walberla::mesa_pd::data::particle_flags;
   set(p->getFlagsRef(), INFINITE);
   set(p->getFlagsRef(), FIXED);
   set(p->getFlagsRef(), GLOBAL);

   auto sp                       = ps->create();
   sp->getPositionRef()          = Vec3(0_r, 0_r, 0.01_r);
   sp->getInteractionRadiusRef() = 1_r;
   sp->getShapeIDRef()           = ss->create<data::Sphere>( 0.004_r );
   ss->shapes[sp->getShapeID()]->updateMassAndInertia(2707_r);
   WALBERLA_LOG_DEVEL_VAR(ss->shapes[sp->getShapeID()]->getInvMass());
   WALBERLA_LOG_DEVEL_VAR(ss->shapes[sp->getShapeID()]->getInvInertiaBF());
   sp->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   sp->getTypeRef()              = 0;

   auto bx                       = ps->create();
   bx->getPositionRef()          = Vec3(1_r, 0_r, 0.01_r);
   bx->getInteractionRadiusRef() = 1_r;
   bx->getShapeIDRef()           = ss->create<data::Box>( Vec3(real_t(0.008*0.8)) );
   ss->shapes[bx->getShapeID()]->updateMassAndInertia(2707_r);
   WALBERLA_LOG_DEVEL_VAR(ss->shapes[bx->getShapeID()]->getInvMass());
   WALBERLA_LOG_DEVEL_VAR(ss->shapes[bx->getShapeID()]->getInvInertiaBF());
   bx->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   bx->getTypeRef()              = 0;

   auto el                       = ps->create();
   el->getPositionRef()          = Vec3(2_r, 0_r, 0.01_r);
   el->getInteractionRadiusRef() = 1_r;
   el->getShapeIDRef()           = ss->create<data::Ellipsoid>( Vec3(0.004_r) );
   ss->shapes[el->getShapeID()]->updateMassAndInertia(2707_r);
   WALBERLA_LOG_DEVEL_VAR(ss->shapes[el->getShapeID()]->getInvMass());
   WALBERLA_LOG_DEVEL_VAR(ss->shapes[el->getShapeID()]->getInvInertiaBF());
   el->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   el->getTypeRef()              = 0;

   int64_t simulationSteps = 200000;
   real_t dt = 0.00001_r;

   // Init kernels
   kernel::SemiImplicitEuler             implEuler( dt );
   kernel::SpringDashpot                 dem(1);
   auto meff = 1.0_r / ss->shapes[sp->getShapeID()]->getInvMass();
   dem.setParametersFromCOR(0,0,0.9_r, dt * 20_r, meff);
   dem.setDampingT (0, 0, real_t(6.86e1));
   dem.setFriction (0, 0, 1.2_r);
   WALBERLA_LOG_DEVEL_VAR(dem.getStiffness(0,0));
   WALBERLA_LOG_DEVEL_VAR(dem.getDampingN(0,0));
   WALBERLA_LOG_DEVEL_VAR(dem.calcCollisionTime(0,0,meff));
   WALBERLA_LOG_DEVEL_VAR(dem.calcCoefficientOfRestitution(0,0,meff));

   collision_detection::GeneralContactDetection gcd;
   kernel::DoubleCast                 double_cast;

   for (int64_t i=0; i < simulationSteps; ++i)
   {
      ps->forEachParticle(false,
                          kernel::SelectLocal(),
                          accessor,
                          [&](const size_t idx, auto& ac)
      {
         ac.setForce(idx,
                     Vec3(0_r, 0_r, -9.81_r) *
                     1.0_r / ss->shapes[ac.getShapeID(idx)]->getInvMass() );
      }, accessor);

      ps->forEachParticlePairHalf(false,
                                  kernel::SelectAll(),
                                  accessor,
                                  [&](const size_t idx1, const size_t idx2, auto& ac)
      {
         if (double_cast(idx1, idx2, ac, gcd, ac ))
         {
            if (ss->shapes[ac.getShapeID(idx1)]->getShapeType() == data::HalfSpace::SHAPE_TYPE)
            {
               meff = 1.0_r / ss->shapes[ac.getShapeID(idx2)]->getInvMass();
            } else
            {
               meff = 1.0_r / ss->shapes[ac.getShapeID(idx1)]->getInvMass();
            }

            dem.setParametersFromCOR(0,0,0.9_r, dt * 20_r, meff);
            dem(gcd.getIdx1(), gcd.getIdx2(), ac, gcd.getContactPoint(), gcd.getContactNormal(), gcd.getPenetrationDepth());
         }
      },
      accessor );

      ps->forEachParticle(false,
                          kernel::SelectLocal(),
                          accessor,
                          implEuler,
                          accessor);

//      if(i%1 == 0)
//         WALBERLA_LOG_DEVEL_VAR(sp->getPosition()[2]);
   }

   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON(sp->getPosition()[2], 0.004_r,     1e-7_r);
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON(bx->getPosition()[2], real_t(0.004*0.8), 1e-7_r);
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON(el->getPosition()[2], 0.004_r,     1e-7_r);

   return EXIT_SUCCESS;
}

} // namespace mesa_pd
} // namespace walberla

int main( int argc, char* argv[] )
{
   return walberla::mesa_pd::main( argc, argv );
}
