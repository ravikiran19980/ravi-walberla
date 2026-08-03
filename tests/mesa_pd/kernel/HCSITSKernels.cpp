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
//! \author Tobias Leemann <tobias.leemann@fau.de>
//
//======================================================================================================================


/** Test Collision Detection and Insertion of contacts into the contact storage */

#include <mesa_pd/data/DataTypes.h>
#include <mesa_pd/data/ContactStorage.h>
#include <mesa_pd/data/ParticleStorage.h>
#include <mesa_pd/data/ShapeStorage.h>
#include <mesa_pd/kernel/DetectAndStoreContacts.h>
#include <mesa_pd/domain/InfiniteDomain.h>
#include <mesa_pd/data/ContactAccessor.h>
#include <mesa_pd/kernel/ParticleSelector.h>
//HCSITS Kernels

#include "mesa_pd/kernel/InitContactsForHCSITS.h"
#include "mesa_pd/kernel/InitParticlesForHCSITS.h"
#include "mesa_pd/kernel/HCSITSRelaxationStep.h"
#include "mesa_pd/kernel/IntegrateParticlesHCSITS.h"

// Communication kernels
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/BroadcastProperty.h"

#include "mesa_pd/mpi/notifications/VelocityUpdateNotification.h"
#include "mesa_pd/mpi/notifications/VelocityCorrectionNotification.h"

#include <mesa_pd/data/ParticleAccessorWithShape.h>
#include <core/Environment.h>
#include <core/logging/Logging.h>

#include <iostream>

namespace walberla {
namespace mesa_pd {


template<typename PStorage, typename CStorage, typename PAccessor, typename CAccessor>
class TestHCSITSKernel {
public:
   TestHCSITSKernel(PStorage &ps_, CStorage &cs_, PAccessor &pa_, CAccessor &ca_) : ps(ps_), cs(cs_), pa(pa_), ca(ca_),
      erp(1.0_r), model(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticFrictionlessContact), contactThreshold(0), globalAcc(0) {}

   void operator()(real_t dt){
      // Perform Collision detection (call kernel, that stores contacts into cs)
      kernel::DetectAndStoreContacts detectAndStore(cs);
      cs.clear();

      domain::InfiniteDomain domain;
      collision_detection::AnalyticContactDetection acd;
      acd.getContactThreshold() = contactThreshold;
      ps.forEachParticlePairHalf(false, kernel::ExcludeInfiniteInfinite(), pa, detectAndStore, pa, domain, acd);

      // Create Kernels
      kernel::InitContactsForHCSITS initContacts(1);
      initContacts.setFriction(0,0,0.2_r);
      initContacts.setErp(static_cast< real_t >(erp));

      kernel::InitParticlesForHCSITS initParticles;
      initParticles.setGlobalAcceleration(globalAcc);

      kernel::HCSITSRelaxationStep relaxationStep;
      relaxationStep.setRelaxationModel(model);
      relaxationStep.setCor(0.6_r); // Only effective for PGSM

      kernel::IntegrateParticlesHCSITS integration;

      mesa_pd::mpi::ReduceProperty reductionKernel;
      mesa_pd::mpi::BroadcastProperty broadcastKernel;

      // Run the HCSITS loop
      cs.forEachContact(false, kernel::SelectAll(), ca, initContacts, ca, pa);
      ps.forEachParticle(false, kernel::SelectAll(), pa, initParticles, pa, dt);

      VelocityUpdateNotification::Parameters::relaxationParam = 1.0_r;
      reductionKernel.operator()<VelocityCorrectionNotification>(ps);
      broadcastKernel.operator()<VelocityUpdateNotification>(ps);

      VelocityUpdateNotification::Parameters::relaxationParam = 0.8_r;
      for(int i = 0; i < 10; i++){
         cs.forEachContact(false, kernel::SelectAll(), ca, relaxationStep, ca, pa, dt);
         reductionKernel.operator()<VelocityCorrectionNotification>(ps);
         broadcastKernel.operator()<VelocityUpdateNotification>(ps);
      }
      ps.forEachParticle(false, kernel::SelectAll(), pa, integration, pa, dt);
   }

private:
   PStorage  &ps;
   CStorage  &cs;
   PAccessor &pa;
   CAccessor &ca;

public:
   real_t erp;
   kernel::HCSITSRelaxationStep::RelaxationModel model;
   real_t contactThreshold;
   Vec3 globalAcc;
};


void normalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel model)
{
   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(100);
   auto cs = std::make_shared<data::ContactStorage>(100);
   auto ss = std::make_shared<data::ShapeStorage>();
   data::ParticleAccessorWithShape paccessor(ps, ss);
   data::ContactAccessor caccessor(cs);
   auto density = 7.874_r;
   auto radius = 1.1_r;

   //Geometries for sphere and half space.
   auto smallSphere = ss->create<data::Sphere>( radius );
   auto halfSpace = ss->create<data::HalfSpace>(Vec3(0,0,1));

   ss->shapes[halfSpace]->updateMassAndInertia( density );
   ss->shapes[smallSphere]->updateMassAndInertia( density );

   // Create four slightly overlapping spheres in a row (located at x=0,2)
   auto p = ps->create();
   p->getPositionRef()        = Vec3(0_r, 0_r, 0_r);
   p->getShapeIDRef()         = smallSphere;
   p->getOwnerRef()           = walberla::mpi::MPIManager::instance()->rank();
   p->getLinearVelocityRef()  = Vec3(5_r, 5_r, 6_r);
   p->getTypeRef()            = 0;
   //WALBERLA_LOG_INFO(paccessor.ParticleAccessorWithShape::getInvMass(0));


   auto p2 = ps->create();
   p2->getPositionRef()          = Vec3(5_r, 5_r, 5_r);
   p2->getShapeIDRef()           = halfSpace;
   p2->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p2->getTypeRef()               = 0;
   data::particle_flags::set(p2->getFlagsRef(), data::particle_flags::FIXED);
   data::particle_flags::set(p2->getFlagsRef(), data::particle_flags::GLOBAL);

   TestHCSITSKernel<data::ParticleStorage, data::ContactStorage, data::ParticleAccessorWithShape, data::ContactAccessor> testHCSITS(*ps, *cs, paccessor, caccessor);
   testHCSITS.model = model;

   WALBERLA_LOG_INFO(paccessor.getInvMass(0))
         WALBERLA_LOG_INFO(paccessor.getInvMass(1))

         // plane at 5,5,5
         // radius 1.1
         p->setPosition(  Vec3(5,5,6) );
   p->setLinearVelocity( Vec3(0,0,0) );
   testHCSITS( real_c( 1.0_r ) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6.1_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,0.1_r) );
   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());

   p->setPosition(  Vec3(5,5,6) );
   p->setLinearVelocity( Vec3(0,0,0) );
   testHCSITS.erp = 0.5_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6.05_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,0.05_r) );
   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());

   p->setPosition(  Vec3(5,5,6) );
   p->setLinearVelocity( Vec3(0,0,0) );
   testHCSITS.erp = 0.0_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,0) );
   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());

   p->setPosition(  Vec3(5,5,6) );
   p->setLinearVelocity( Vec3(0,0,-1) );
   testHCSITS.erp = 1.0_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6.1_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,0.1_r) );
   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());

   p->setPosition(  Vec3(5,5,6) );
   p->setLinearVelocity( Vec3(0,0,-1) );
   testHCSITS.erp = 0.5_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6.05_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,0.05_r) );
   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());

   p->setPosition(  Vec3(5,5,6) );
   p->setLinearVelocity( Vec3(0,0,-1) );
   testHCSITS.erp = 0.0_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,0) );
   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());

   // No collision
   p->setPosition(  Vec3(5,5,6.2_r) );
   p->setLinearVelocity( Vec3(0,0,-0.2_r) );
   testHCSITS.erp = 1.0_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6.0_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,-0.2_r) );

   testHCSITS.contactThreshold = 1.0_r;
   p->setPosition(  Vec3(5,5,6.2_r) );
   p->setLinearVelocity( Vec3(0,0,-0.2_r) );
   testHCSITS.erp = 1.0_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6.1_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,-0.1_r) );

   p->setPosition(  Vec3(5,5,6.1_r) );
   p->setLinearVelocity( Vec3(0,0,-0.1_r) );
   testHCSITS.erp = 1.0_r ;
   testHCSITS( 1.0_r );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getPosition() , Vec3(5,5,6.1_r) );
   WALBERLA_CHECK_FLOAT_EQUAL( p->getLinearVelocity(), Vec3(0,0,0_r) );
   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());
}


/**Check hard contact constraints on two overlapping, colliding spheres
 * Works only for the solvers that really achieve separation after a single
 * timestep. Use SphereSeperationTest to check for separation after multiple
 * timesteps.
 * @param model The collision model to use.
 * */
void SphereSphereTest(kernel::HCSITSRelaxationStep::RelaxationModel model){

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(100);
   auto cs = std::make_shared<data::ContactStorage>(100);
   auto ss = std::make_shared<data::ShapeStorage>();
   data::ParticleAccessorWithShape paccessor(ps, ss);
   data::ContactAccessor caccessor(cs);
   auto density = 7.874_r;
   auto radius = 1.1_r;

   auto smallSphere = ss->create<data::Sphere>( radius );
   ss->shapes[smallSphere]->updateMassAndInertia( density );

   auto dt = 1_r;

   // Create two slightly overlapping spheres in a row (located at x=0,2)
   auto p = ps->create();
   p->getPositionRef()          = Vec3(0_r, 0_r, 0_r);
   p->getShapeIDRef()           = smallSphere;
   p->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p->getLinearVelocityRef()    = Vec3(1_r, 0_r, 0_r);
   p->getTypeRef()              = 0;
   auto p2 = ps->create();
   p2->getPositionRef()          = Vec3(2_r, 0_r, 0_r);
   p2->getShapeIDRef()           = smallSphere;
   p2->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p2->getLinearVelocityRef() = Vec3(-1_r, 0_r, 0_r);
   p2->getTypeRef()              = 0;
   TestHCSITSKernel<data::ParticleStorage, data::ContactStorage, data::ParticleAccessorWithShape, data::ContactAccessor> testHCSITS(*ps, *cs, paccessor, caccessor);
   testHCSITS.model = model;
   testHCSITS(dt);

   WALBERLA_CHECK_FLOAT_EQUAL(p->getPosition(), Vec3(-0.1_r,0,0));
   WALBERLA_CHECK_FLOAT_EQUAL(p->getLinearVelocity(), Vec3(-0.1_r,0,0));
   WALBERLA_CHECK_FLOAT_EQUAL(p->getAngularVelocity(), Vec3(0,0,0));
   WALBERLA_CHECK_FLOAT_EQUAL(p2->getPosition(), Vec3(2.1_r,0,0));
   WALBERLA_CHECK_FLOAT_EQUAL(p2->getLinearVelocity(), Vec3(0.1_r,0,0))
         WALBERLA_CHECK_FLOAT_EQUAL(p2->getAngularVelocity(), Vec3(0,0,0));

   WALBERLA_LOG_INFO(p->getPosition());
   WALBERLA_LOG_INFO(p->getLinearVelocity());
   WALBERLA_LOG_INFO(p2->getPosition());
   WALBERLA_LOG_INFO(p2->getLinearVelocity());
}

/**
 * Create two overlapping spheres with opposing velocities, that are in contact.
 * Assert that the collision is resolved after a certain (10) number of timesteps.
 * @param model The collision model to use.
 */
void SphereSeperationTest(kernel::HCSITSRelaxationStep::RelaxationModel model){

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(100);
   auto cs = std::make_shared<data::ContactStorage>(100);
   auto ss = std::make_shared<data::ShapeStorage>();
   data::ParticleAccessorWithShape paccessor(ps, ss);
   data::ContactAccessor caccessor(cs);
   auto density = 7.874_r;
   auto radius = 1.1_r;

   auto smallSphere = ss->create<data::Sphere>( radius );

   ss->shapes[smallSphere]->updateMassAndInertia( density );
   auto dt = 0.2_r;

   // Create two slightly overlapping spheres in a row (located at x=0,2)
   auto p = ps->create();
   p->getPositionRef()          = Vec3(0_r, 0_r, 0_r);
   p->getShapeIDRef()           = smallSphere;
   p->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p->getLinearVelocityRef()    = Vec3(1_r, 0_r, 0_r);
   p->getTypeRef()              = 0;
   auto p2 = ps->create();
   p2->getPositionRef()          = Vec3(2.0_r, 0_r, 0_r);
   p2->getShapeIDRef()           = smallSphere;
   p2->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p2->getLinearVelocityRef()    = Vec3(-1_r, 0_r, 0_r);
   p2->getTypeRef()              = 0;
   TestHCSITSKernel<data::ParticleStorage, data::ContactStorage, data::ParticleAccessorWithShape, data::ContactAccessor> testHCSITS(*ps, *cs, paccessor, caccessor);

   int solveCount = 0;
   testHCSITS.model = model;

   // Number of allowed iterations
   int maxIter = 10;
   while(p2->getPosition()[0]-p->getPosition()[0] < 2.2){
      testHCSITS(dt);
      WALBERLA_LOG_INFO(p->getPosition());
      WALBERLA_LOG_INFO(p->getLinearVelocity());
      WALBERLA_LOG_INFO(p2->getPosition());
      WALBERLA_LOG_INFO(p2->getLinearVelocity());
      solveCount ++;
      if(solveCount==maxIter){
         WALBERLA_CHECK(false, "Separation did not occur after " << maxIter << " Iterations performed.");
      }
   }
   WALBERLA_LOG_INFO("Separation achieved after " << solveCount << " iterations.");
}

/**
 * Create a sphere that slides on a plane with only linear velocity. It is
 * put into a spin by the frictional reactions, until it rolls with no slip.
 * Assert that the sphere rolls with all slip resolved and at the physically correct
 * speed after a certain number of timesteps. Use only with solvers that obey the coulomb friction law.
 * @param model The collision model to use.
 */
void SlidingSphereFrictionalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel model){

   //init data structures
   auto ps = std::make_shared<data::ParticleStorage>(100);
   auto cs = std::make_shared<data::ContactStorage>(100);
   auto ss = std::make_shared<data::ShapeStorage>();
   data::ParticleAccessorWithShape paccessor(ps, ss);
   data::ContactAccessor caccessor(cs);
   auto density = 1_r;
   auto radius = 1_r;

   auto smallSphere = ss->create<data::Sphere>( radius );
   auto halfSpace = ss->create<data::HalfSpace>(Vec3(0,0,1));
   ss->shapes[smallSphere]->updateMassAndInertia( density );
   ss->shapes[halfSpace]->updateMassAndInertia( density );
   auto dt = 0.002_r;

   // Create a spheres (located at x=0, height = 1)
   auto p = ps->create();
   p->getPositionRef()          = Vec3(0_r, 0_r, 1_r);
   p->getShapeIDRef()           = smallSphere;
   p->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p->getLinearVelocityRef()    = Vec3(5_r, 0_r, 0_r);
   p->getTypeRef()              = 0;

   auto p2 = ps->create();
   p2->getPositionRef()          = Vec3(0_r, 0_r, 0_r);
   p2->getShapeIDRef()           = halfSpace;
   p2->getOwnerRef()             = walberla::mpi::MPIManager::instance()->rank();
   p2->getTypeRef()               = 0;
   data::particle_flags::set(p2->getFlagsRef(), data::particle_flags::FIXED);
   data::particle_flags::set(p2->getFlagsRef(), data::particle_flags::GLOBAL);

   TestHCSITSKernel<data::ParticleStorage, data::ContactStorage, data::ParticleAccessorWithShape, data::ContactAccessor> testHCSITS(*ps, *cs, paccessor, caccessor);
   testHCSITS.model = model;
   testHCSITS.globalAcc = Vec3(0,0,-10);
   int solveCount = 0;


   // Number of allowed iterations
   int maxIter = 500;
   while(!walberla::floatIsEqual(p->getAngularVelocity()[1],p->getLinearVelocity()[0], 0.002_r)){
      testHCSITS(dt);
      if(solveCount % 50 == 0){
         WALBERLA_LOG_INFO(p->getAngularVelocity());
         WALBERLA_LOG_INFO(p->getLinearVelocity());
      }
      solveCount ++;
      if(solveCount==maxIter){
         WALBERLA_CHECK(false, "End of slip did not occur after " << maxIter << " Iterations performed.");
      }
   }
   WALBERLA_LOG_INFO("Rolling with no slip achieved after " << solveCount << " iterations.");

   // Check if the value obtained values equal the physically correct values
   // (which can be determined by newtons equation to be 25/7).
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON(real_t(25.0/7.0), p->getAngularVelocity()[1], 0.01_r, "Angular velocity is not physically correct");
   WALBERLA_CHECK_FLOAT_EQUAL_EPSILON(real_t(25.0/7.0), p->getLinearVelocity()[0], 0.01_r, "Linear velocity is not physically correct");
}

int main( int argc, char ** argv )
{
   Environment env(argc, argv);
   WALBERLA_UNUSED(env);

   walberla::mpi::MPIManager::instance()->useWorldComm();

   WALBERLA_LOG_INFO_ON_ROOT("*** SETUP - START ***");


   WALBERLA_LOG_INFO_ON_ROOT("InelasticFrictionlessContact");
   normalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticFrictionlessContact);
   SphereSphereTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticFrictionlessContact);

   WALBERLA_LOG_INFO_ON_ROOT("ApproximateInelasticCoulombContactByDecoupling");
   normalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::ApproximateInelasticCoulombContactByDecoupling);
   SphereSphereTest(kernel::HCSITSRelaxationStep::RelaxationModel::ApproximateInelasticCoulombContactByDecoupling);
   SlidingSphereFrictionalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::ApproximateInelasticCoulombContactByDecoupling);

   WALBERLA_LOG_INFO_ON_ROOT("InelasticCoulombContactByDecoupling");
   normalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticCoulombContactByDecoupling);
   SphereSphereTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticCoulombContactByDecoupling);
   SlidingSphereFrictionalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticCoulombContactByDecoupling);

   WALBERLA_LOG_INFO_ON_ROOT("InelasticGeneralizedMaximumDissipationContact");
   normalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticGeneralizedMaximumDissipationContact);
   SphereSphereTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticGeneralizedMaximumDissipationContact);
   SlidingSphereFrictionalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticGeneralizedMaximumDissipationContact);

   WALBERLA_LOG_INFO_ON_ROOT("InelasticCoulombContactByOrthogonalProjections");
   SphereSeperationTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticCoulombContactByOrthogonalProjections);
   SlidingSphereFrictionalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticCoulombContactByOrthogonalProjections);

   WALBERLA_LOG_INFO_ON_ROOT("ApproximateInelasticCoulombContactByOrthogonalProjections");
   SphereSeperationTest(kernel::HCSITSRelaxationStep::RelaxationModel::ApproximateInelasticCoulombContactByOrthogonalProjections);
   SlidingSphereFrictionalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::ApproximateInelasticCoulombContactByOrthogonalProjections);

   WALBERLA_LOG_INFO_ON_ROOT("InelasticProjectedGaussSeidel");
   SphereSeperationTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticProjectedGaussSeidel);
   SlidingSphereFrictionalReactionTest(kernel::HCSITSRelaxationStep::RelaxationModel::InelasticProjectedGaussSeidel);
   return EXIT_SUCCESS;
}

} //namespace mesa_pd
} //namespace walberla

int main( int argc, char ** argv )
{
   return walberla::mesa_pd::main(argc, argv);
}
