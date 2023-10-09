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
//! \file ObliqueWetCollisionPSM.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/debug/Debug.h"
#include "core/debug/TestSubsystem.h"
#include "core/logging/all.h"
#include "core/math/all.h"
#include "core/mpi/Broadcast.h"
#include "core/mpi/Reduce.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/waLBerlaBuildInfo.h"

#include "field/AddToStorage.h"
#include "field/interpolators/NearestNeighborFieldInterpolator.h"
#include "field/vtk/all.h"

#include "geometry/InitBoundaryHandling.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/PSMSweepCollectionGPU.h"
#include "lbm_mesapd_coupling/utility/AddForceOnParticlesKernel.h"
#include "lbm_mesapd_coupling/utility/AddHydrodynamicInteractionKernel.h"
#include "lbm_mesapd_coupling/utility/AverageHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/InitializeHydrodynamicForceTorqueForAveragingKernel.h"
#include "lbm_mesapd_coupling/utility/LubricationCorrectionKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/data/shape/HalfSpace.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDataHandling.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/ExplicitEuler.h"
#include "mesa_pd/kernel/LinearSpringDashpot.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/kernel/VelocityVerlet.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/mpi/notifications/HydrodynamicForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "vtk/all.h"

#include <functional>

#include "PSMPackInfo.h"
#include "PSMSweep.h"
#include "PSM_Density.h"
#include "PSM_InfoHeader.h"
#include "PSM_MacroGetter.h"
#include "PSM_NoSlip.h"

namespace oblique_wet_collision
{

///////////
// USING //
///////////

using namespace walberla;
using walberla::uint_t;
using namespace lbm_mesapd_coupling::psm::gpu;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

using ScalarField_T = GhostLayerField< real_t, 1 >;

const uint_t FieldGhostLayers = 1;
typedef pystencils::PSMPackInfo PackInfo_T;

///////////
// FLAGS //
///////////

const FlagUID Fluid_Flag("Fluid");
const FlagUID NoSlip_Flag("NoSlip");
const FlagUID SimplePressure_Flag("SimplePressure");

//*******************************************************************************************************************
/*!\brief Evaluating the position and velocity of the sphere
 *
 */
//*******************************************************************************************************************
template< typename ParticleAccessor_T >
class SpherePropertyLogger
{
 public:
   SpherePropertyLogger(const shared_ptr< ParticleAccessor_T >& ac, walberla::id_t sphereUid,
                        const std::string& fileNameLogging, const std::string& fileNameForceLogging, bool fileIO,
                        real_t diameter, real_t uIn, real_t impactRatio, uint_t numberOfSubSteps, real_t fGravX,
                        real_t fGravZ)
      : ac_(ac), sphereUid_(sphereUid), fileNameLogging_(fileNameLogging), fileNameForceLogging_(fileNameForceLogging),
        fileIO_(fileIO), diameter_(diameter), uIn_(uIn), impactRatio_(impactRatio), numberOfSubSteps_(numberOfSubSteps),
        fGravX_(fGravX), fGravZ_(fGravZ), gap_(real_t(0)), settlingVelocity_(real_t(0)), tangentialVelocity_(real_t(0))
   {
      if (fileIO_)
      {
         WALBERLA_ROOT_SECTION()
         {
            std::ofstream file;
            file.open(fileNameLogging_.c_str());
            file << "#\t D\t uIn\t impactRatio\t positionX\t positionY\t positionZ\t velX\t velY\t velZ\t angX\t "
                    "angY\t angZ\n";
            file.close();
         }
         WALBERLA_ROOT_SECTION()
         {
            std::ofstream file;
            file.open(fileNameForceLogging_.c_str());
            file << "#\t fGravX\t fGravZ\t fHydX\t fHydZ\t fLubX\t fLubZ\t fColX\t fColZ\t tHydY\t tLubY\t tColY\n";
            file.close();
         }
      }
   }

   void operator()(const uint_t timeStep, const uint_t subStep, Vector3< real_t > fHyd, Vector3< real_t > fLub,
                   Vector3< real_t > fCol, Vector3< real_t > tHyd, Vector3< real_t > tLub, Vector3< real_t > tCol)
   {
      real_t curTimestep = real_c(timeStep) + real_c(subStep) / real_c(numberOfSubSteps_);

      Vector3< real_t > pos(real_t(0));
      Vector3< real_t > transVel(real_t(0));
      Vector3< real_t > angVel(real_t(0));

      size_t idx = ac_->uidToIdx(sphereUid_);
      if (idx != std::numeric_limits< size_t >::max())
      {
         if (!mesa_pd::data::particle_flags::isSet(ac_->getFlags(idx), mesa_pd::data::particle_flags::GHOST))
         {
            pos      = ac_->getPosition(idx);
            transVel = ac_->getLinearVelocity(idx);
            angVel   = ac_->getAngularVelocity(idx);
         }
      }

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace(pos, mpi::SUM);
         mpi::allReduceInplace(transVel, mpi::SUM);
         mpi::allReduceInplace(angVel, mpi::SUM);
      }

      position_           = pos;
      gap_                = pos[2] - real_t(0.5) * diameter_;
      settlingVelocity_   = transVel[2];
      tangentialVelocity_ = transVel[0] - diameter_ * real_t(0.5) * angVel[1];

      if (fileIO_ /* && gap_ < diameter_*/)
      {
         writeToLoggingFile(curTimestep, pos, transVel, angVel);
         writeToForceLoggingFile(curTimestep, fHyd, fLub, fCol, tHyd, tLub, tCol);
      }
   }

   real_t getGapSize() const { return gap_; }

   real_t getSettlingVelocity() const { return settlingVelocity_; }

   real_t getTangentialVelocity() const { return tangentialVelocity_; }

   Vector3< real_t > getPosition() const { return position_; }

 private:
   void writeToLoggingFile(real_t timestep, Vector3< real_t > position, Vector3< real_t > transVel,
                           Vector3< real_t > angVel)
   {
      WALBERLA_ROOT_SECTION()
      {
         std::ofstream file;
         file.open(fileNameLogging_.c_str(), std::ofstream::app);

         file << timestep << "\t" << diameter_ << "\t" << uIn_ << "\t" << impactRatio_ << "\t" << position[0] << "\t"
              << position[1] << "\t" << position[2] << "\t" << transVel[0] << "\t" << transVel[1] << "\t" << transVel[2]
              << "\t" << angVel[0] << "\t" << angVel[1] << "\t" << angVel[2] << "\n";
         file.close();
      }
   }

   void writeToForceLoggingFile(real_t timestep, Vector3< real_t > fHyd, Vector3< real_t > fLub, Vector3< real_t > fCol,
                                Vector3< real_t > tHyd, Vector3< real_t > tLub, Vector3< real_t > tCol)
   {
      WALBERLA_ROOT_SECTION()
      {
         std::ofstream file;
         file.open(fileNameForceLogging_.c_str(), std::ofstream::app);

         file << timestep << "\t" << fGravX_ << "\t" << fGravZ_ << "\t" << fHyd[0] << "\t" << fHyd[2] << "\t" << fLub[0]
              << "\t" << fLub[2] << "\t" << fCol[0] << "\t" << fCol[2] << "\t" << tHyd[1] << "\t" << tLub[1] << "\t"
              << tCol[1] << "\n";
         file.close();
      }
   }

   shared_ptr< ParticleAccessor_T > ac_;
   const walberla::id_t sphereUid_;
   std::string fileNameLogging_, fileNameForceLogging_;
   bool fileIO_;
   real_t diameter_, uIn_, impactRatio_;
   uint_t numberOfSubSteps_;
   real_t fGravX_, fGravZ_;

   real_t gap_, settlingVelocity_, tangentialVelocity_;
   Vector3< real_t > position_;
};

void createPlaneSetup(const shared_ptr< mesa_pd::data::ParticleStorage >& ps,
                      const shared_ptr< mesa_pd::data::ShapeStorage >& ss, const math::AABB& simulationDomain,
                      bool applyOutflowBCAtTop)
{
   // create bounding planes
   mesa_pd::data::Particle&& p0 = *ps->create(true);
   p0.setPosition(simulationDomain.minCorner());
   p0.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p0.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 0, 1)));
   p0.setOwner(mpi::MPIManager::instance()->rank());
   p0.setType(0);
   mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

   if (!applyOutflowBCAtTop)
   {
      // only create top plane when no outflow BC should be set there
      mesa_pd::data::Particle&& p1 = *ps->create(true);
      p1.setPosition(simulationDomain.maxCorner());
      p1.setInteractionRadius(std::numeric_limits< real_t >::infinity());
      p1.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 0, -1)));
      p1.setOwner(mpi::MPIManager::instance()->rank());
      p1.setType(0);
      mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
      mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
   }
}

class ForcesOnSphereLogger
{
 public:
   ForcesOnSphereLogger(const std::string& fileName, bool fileIO, real_t weightForce, uint_t numberOfSubSteps)
      : fileName_(fileName), fileIO_(fileIO), weightForce_(weightForce), numberOfSubSteps_(numberOfSubSteps)
   {
      if (fileIO_)
      {
         WALBERLA_ROOT_SECTION()
         {
            std::ofstream file;
            file.open(fileName_.c_str());
            file << "#t\t BuoyAndGravF\t HydInteractionF\t LubricationF\t CollisionForce\n";
            file.close();
         }
      }
   }

   void operator()(const uint_t timeStep, const uint_t subStep, real_t hydForce, real_t lubForce, real_t collisionForce)
   {
      real_t curTimestep = real_c(timeStep) + real_c(subStep) / real_c(numberOfSubSteps_);

      if (fileIO_)
      {
         WALBERLA_ROOT_SECTION()
         {
            std::ofstream file;
            file.open(fileName_.c_str(), std::ofstream::app);

            file << std::setprecision(10) << curTimestep << "\t" << weightForce_ << "\t" << hydForce << "\t" << lubForce
                 << "\t" << collisionForce << "\n";
            file.close();
         }
      }
   }

 private:
   std::string fileName_;
   bool fileIO_;

   real_t weightForce_;
   uint_t numberOfSubSteps_;
};

template< typename ParticleAccessor_T >
Vector3< real_t > getForce(walberla::id_t uid, ParticleAccessor_T& ac)
{
   auto idx = ac.uidToIdx(uid);
   Vector3< real_t > force(0);
   if (idx != ac.getInvalidIdx()) { force = ac.getForce(idx); }
   WALBERLA_MPI_SECTION() { mpi::allReduceInplace(force, mpi::SUM); }
   return force;
}

template< typename ParticleAccessor_T >
Vector3< real_t > getTorque(walberla::id_t uid, ParticleAccessor_T& ac)
{
   auto idx = ac.uidToIdx(uid);
   Vector3< real_t > torque(0);
   if (idx != ac.getInvalidIdx()) { torque = ac.getTorque(idx); }
   WALBERLA_MPI_SECTION() { mpi::allReduceInplace(torque, mpi::SUM); }
   return torque;
}

//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief PHYSICAL Test case of a sphere settling and colliding with a wall submerged in a viscous fluid.
 *
 * The collision is oblique, i.e. features normal and tangential contributions.
 * There are in principle two ways to carry out this simulation:
 * 1) Use some (artificial) gravitational force such that the x(tangential) and z(normal)- components
 *    have the same ratio as the desired impact ratio of the velocities.
 *    To have similar normal collision velocities, the option 'useFullGravityInNormalDirection' always applies the same
 *    gravitational acceleration in normal direction.
 *    The main problem, however, is that the sphere begins to rotate for increasing impact angles and will thus generate
 *    a reposing force in normal direction, that will alter the intended impact ratio to larger values.
 *    Additionally, a reasonable value for the relaxation time, tau, has to be given to avoid too large settling
 * velocities.
 *
 * 2) Specify a (normal or magnitude-wise) Stokes number and artificially accelerate the sphere to this value.
 *    Thus, the settling velocity is prescribed and no rotational velocity is allowed.
 *    This is stopped before the collision to allow for a 'natural' collision.
 *    To avoid a too large damping by the fluid forces, since no gravitational forces are present to balance them,
 *    an artificial gravity is assumed and taken as the opposite fluid force acting when the artificial
 *    acceleration is stopped.
 *
 * For details see Rettinger, Ruede 2020
 */
//*******************************************************************************************************************

int main(int argc, char** argv)
{
   Environment env(argc, argv);
   gpu::selectDeviceBasedOnMpiRank();

   if (!env.config()) { WALBERLA_ABORT("Usage: " << argv[0] << " path-to-configuration-file \n"); }

   Config::BlockHandle simulationInput = env.config()->getBlock("Setup");

   const int caseNumber                = simulationInput.getParameter< int >("case");
   Config::BlockHandle caseDescription = env.config()->getBlock("Case" + std::to_string(caseNumber));

   const std::string material              = caseDescription.getParameter< std::string >("material");
   Config::BlockHandle materialDescription = env.config()->getBlock("Mat_" + material);
   const real_t densitySphere_SI           = materialDescription.getParameter< real_t >("densitySphere_SI");
   const real_t diameter_SI                = materialDescription.getParameter< real_t >("diameter_SI");
   const real_t restitutionCoeff           = materialDescription.getParameter< real_t >("restitutionCoeff");
   const real_t frictionCoeff              = materialDescription.getParameter< real_t >("frictionCoeff");
   const real_t poissonsRatio              = materialDescription.getParameter< real_t >("poissonsRatio");

   const std::string fluid               = caseDescription.getParameter< std::string >("fluid");
   Config::BlockHandle fluidDescription  = env.config()->getBlock("Fluid_" + fluid);
   const real_t densityFluid_SI          = fluidDescription.getParameter< real_t >("densityFluid_SI");
   const real_t dynamicViscosityFluid_SI = fluidDescription.getParameter< real_t >("dynamicViscosityFluid_SI");

   const std::string simulationVariant    = simulationInput.getParameter< std::string >("variant");
   Config::BlockHandle variantDescription = env.config()->getBlock("Variant_" + simulationVariant);

   const real_t impactRatio       = simulationInput.getParameter< real_t >("impactRatio");
   const bool applyOutflowBCAtTop = simulationInput.getParameter< bool >("applyOutflowBCAtTop");

   // variant dependent parameters
   const Vector3< real_t > domainSizeNonDim = variantDescription.getParameter< Vector3< real_t > >("domainSize");
   const Vector3< uint_t > numberOfBlocksPerDirection =
      variantDescription.getParameter< Vector3< uint_t > >("numberOfBlocksPerDirection");
   WALBERLA_CHECK_EQUAL(
      numberOfBlocksPerDirection[0] * numberOfBlocksPerDirection[1] * numberOfBlocksPerDirection[2],
      uint_t(MPIManager::instance()->numProcesses()),
      "When using GPUs, the number of blocks ("
         << numberOfBlocksPerDirection[0] * numberOfBlocksPerDirection[1] * numberOfBlocksPerDirection[2]
         << ") has to match the number of MPI processes (" << uint_t(MPIManager::instance()->numProcesses()) << ")");
   const real_t initialSphereHeight = variantDescription.getParameter< real_t >("initialSphereHeight");

   // LBM
   const real_t diameter = simulationInput.getParameter< real_t >("diameter");

   const uint_t numRPDSubCycles        = simulationInput.getParameter< uint_t >("numRPDSubCycles");
   const bool useLubricationCorrection = simulationInput.getParameter< bool >("useLubricationCorrection");
   const real_t lubricationCutOffDistanceNormal =
      simulationInput.getParameter< real_t >("lubricationCutOffDistanceNormal");
   const real_t lubricationCutOffDistanceTangentialTranslational =
      simulationInput.getParameter< real_t >("lubricationCutOffDistanceTangentialTranslational");
   const real_t lubricationCutOffDistanceTangentialRotational =
      simulationInput.getParameter< real_t >("lubricationCutOffDistanceTangentialRotational");
   const real_t lubricationMinimalGapSizeNonDim =
      simulationInput.getParameter< real_t >("lubricationMinimalGapSizeNonDim");

   // Collision Response
   const bool useVelocityVerlet = simulationInput.getParameter< bool >("useVelocityVerlet");
   const real_t collisionTime   = simulationInput.getParameter< real_t >("collisionTime");

   const bool averageForceTorqueOverTwoTimeSteps =
      simulationInput.getParameter< bool >("averageForceTorqueOverTwoTimeSteps");

   const bool fileIO            = simulationInput.getParameter< bool >("fileIO");
   const uint_t vtkIOFreq       = simulationInput.getParameter< uint_t >("vtkIOFreq");
   const std::string baseFolder = simulationInput.getParameter< std::string >("baseFolder");

   WALBERLA_ROOT_SECTION()
   {
      if (fileIO)
      {
         // create base directory if it does not yet exist
         filesystem::path tpath(baseFolder);
         if (!filesystem::exists(tpath)) filesystem::create_directory(tpath);
      }
   }

   //////////////////////////////////////
   // SIMULATION PROPERTIES in SI units//
   //////////////////////////////////////

   const real_t impactAngle                = std::atan(impactRatio);
   const real_t densityRatio               = densitySphere_SI / densityFluid_SI;
   const real_t kinematicViscosityFluid_SI = dynamicViscosityFluid_SI / densityFluid_SI;

   WALBERLA_LOG_INFO_ON_ROOT("SETUP OF CASE " << caseNumber);
   WALBERLA_LOG_INFO_ON_ROOT("Setup (in SI units):");
   WALBERLA_LOG_INFO_ON_ROOT(" - impact ratio = " << impactRatio);
   WALBERLA_LOG_INFO_ON_ROOT(" - impact angle = " << impactAngle << " (" << impactAngle * real_t(180) / math::pi
                                                  << " degrees) ");
   WALBERLA_LOG_INFO_ON_ROOT(" - sphere: diameter = " << diameter_SI << ", densityRatio = " << densityRatio);
   WALBERLA_LOG_INFO_ON_ROOT(" - fluid: density = " << densityFluid_SI << ", dyn. visc = " << dynamicViscosityFluid_SI
                                                    << ", kin. visc = " << kinematicViscosityFluid_SI);
   WALBERLA_LOG_INFO_ON_ROOT(" - domain size = <" << real_c(domainSizeNonDim[0]) * diameter_SI << ","
                                                  << real_c(domainSizeNonDim[1]) * diameter_SI << ","
                                                  << real_c(domainSizeNonDim[2]) * diameter_SI << ">");

   //////////////////////////
   // NUMERICAL PARAMETERS //
   //////////////////////////

   WALBERLA_LOG_INFO_ON_ROOT("Setup (in simulation, i.e. lattice, units):");

   const real_t dx_SI        = diameter_SI / diameter;
   const real_t sphereVolume = math::pi / real_t(6) * diameter * diameter * diameter;

   real_t dt_SI;
   real_t viscosity;
   real_t omega;
   real_t uIn                                   = real_t(1);
   real_t accelerationFactor                    = real_t(0);
   bool applyArtificialGravityAfterAccelerating = false;
   bool useFullGravityInNormalDirection         = false;
   bool artificiallyAccelerateSphere            = false;
   Vector3< real_t > gravitationalAccelerationVec(real_t(0));

   if (simulationVariant == "Acceleration")
   {
      WALBERLA_LOG_INFO_ON_ROOT("USING MODE OF ARTIFICIALLY ACCELERATED SPHERE");

      artificiallyAccelerateSphere = true;

      real_t StTarget                         = variantDescription.getParameter< real_t >("StTarget");
      const bool useStTargetInNormalDirection = variantDescription.getParameter< bool >("useStTargetInNormalDirection");
      applyArtificialGravityAfterAccelerating =
         variantDescription.getParameter< bool >("applyArtificialGravityAfterAccelerating");
      bool applyUInNormalDirection = variantDescription.getParameter< bool >("applyUInNormalDirection");
      uIn                          = variantDescription.getParameter< real_t >("uIn");
      accelerationFactor           = variantDescription.getParameter< real_t >("accelerationFactor");

      if (applyUInNormalDirection)
      {
         // the value for uIn is defined as the normal impact velocity, i.e. uNIn
         uIn = uIn / std::cos(impactAngle);
      }

      real_t StTargetN;
      if (useStTargetInNormalDirection)
      {
         StTargetN = StTarget;
         StTarget  = StTarget / std::cos(impactAngle);
      }
      else { StTargetN = StTarget * std::cos(impactAngle); }
      WALBERLA_LOG_INFO_ON_ROOT(" - target St in settling direction = " << StTarget);
      WALBERLA_LOG_INFO_ON_ROOT(" - target St in normal direction = " << StTargetN);

      const real_t uIn_SI = StTarget * real_t(9) / densityRatio * kinematicViscosityFluid_SI / diameter_SI;
      dt_SI               = uIn / uIn_SI * dx_SI;
      viscosity           = kinematicViscosityFluid_SI * dt_SI / (dx_SI * dx_SI);

      // note: no gravity when accelerating artificially
      omega = lbm::collision_model::omegaFromViscosity(viscosity);

      const real_t uNIn = uIn * std::cos(impactAngle);
      const real_t Re_p = diameter * uIn / viscosity;

      WALBERLA_LOG_INFO_ON_ROOT(" - target Reynolds number = " << Re_p);
      WALBERLA_LOG_INFO_ON_ROOT(" - expected normal impact velocity = " << uNIn);
      WALBERLA_LOG_INFO_ON_ROOT(" - expected tangential impact velocity = " << uIn * std::sin(impactAngle));
   }
   else
   {
      WALBERLA_LOG_INFO_ON_ROOT("USING MODE OF SPHERE SETTLING UNDER GRAVITY");

      const real_t gravitationalAcceleration_SI =
         variantDescription.getParameter< real_t >("gravitationalAcceleration_SI");
      useFullGravityInNormalDirection = variantDescription.getParameter< bool >("useFullGravityInNormalDirection");
      const real_t tau                = variantDescription.getParameter< real_t >("tau");

      const real_t ug_SI = std::sqrt((densityRatio - real_t(1)) * gravitationalAcceleration_SI * diameter_SI);
      const real_t Ga    = ug_SI * diameter_SI / kinematicViscosityFluid_SI;

      viscosity                        = lbm::collision_model::viscosityFromOmega(real_t(1) / tau);
      dt_SI                            = viscosity * dx_SI * dx_SI / kinematicViscosityFluid_SI;
      real_t gravitationalAcceleration = gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;
      omega                            = real_t(1) / tau;

      WALBERLA_LOG_INFO_ON_ROOT(" - ug = " << ug_SI);
      WALBERLA_LOG_INFO_ON_ROOT(" - Galileo number = " << Ga);

      gravitationalAccelerationVec =
         useFullGravityInNormalDirection ?
            Vector3< real_t >(impactRatio, real_t(0), -real_t(1)) * gravitationalAcceleration :
            Vector3< real_t >(std::sin(impactAngle), real_t(0), -std::cos(impactAngle)) * gravitationalAcceleration;
      WALBERLA_LOG_INFO_ON_ROOT(" - g " << gravitationalAcceleration);
   }

   const real_t densityFluid  = real_t(1);
   const real_t densitySphere = densityRatio;

   const real_t dx = real_t(1);

   const real_t responseTime = densityRatio * diameter * diameter / (real_t(18) * viscosity);

   const real_t particleMass = densitySphere * sphereVolume;
   const real_t Mij =
      particleMass; // * particleMass / ( real_t(2) * particleMass ); // Mij = M for sphere-wall collision
   const real_t kappa = real_t(2) * (real_t(1) - poissonsRatio) / (real_t(2) - poissonsRatio);

   Vector3< uint_t > domainSize(uint_c(domainSizeNonDim[0] * diameter), uint_c(domainSizeNonDim[1] * diameter),
                                uint_c(domainSizeNonDim[2] * diameter));

   real_t initialSpherePosition = initialSphereHeight * diameter;

   WALBERLA_LOG_INFO_ON_ROOT(" - dt_SI = " << dt_SI << " s, dx_SI = " << dx_SI << " m");
   WALBERLA_LOG_INFO_ON_ROOT(" - domain size = " << domainSize);
   if (applyOutflowBCAtTop) WALBERLA_LOG_INFO_ON_ROOT(" - outflow BC at top");
   WALBERLA_LOG_INFO_ON_ROOT(" - sphere: diameter = " << diameter << ", density = " << densitySphere);
   WALBERLA_LOG_INFO_ON_ROOT(" - initial sphere position = " << initialSpherePosition);
   WALBERLA_LOG_INFO_ON_ROOT(" - fluid: density = " << densityFluid << ", relaxation time (tau) = " << real_t(1) / omega
                                                    << ", omega = " << omega << ", kin. visc = " << viscosity);
   WALBERLA_LOG_INFO_ON_ROOT(" - gravitational acceleration = " << gravitationalAccelerationVec);
   WALBERLA_LOG_INFO_ON_ROOT(" - Stokes response time = " << responseTime);
   if (artificiallyAccelerateSphere)
   {
      WALBERLA_LOG_INFO_ON_ROOT(" - artificially accelerating sphere with factor " << accelerationFactor)
      if (applyArtificialGravityAfterAccelerating)
      {
         WALBERLA_LOG_INFO_ON_ROOT(" - applying artificial gravity after accelerating");
      }
   }
   WALBERLA_LOG_INFO_ON_ROOT(" - integrator = " << (useVelocityVerlet ? "Velocity Verlet" : "Explicit Euler"));
   if (vtkIOFreq > 0)
   {
      WALBERLA_LOG_INFO_ON_ROOT(" - writing vtk files to folder \"" << baseFolder << "\" with frequency " << vtkIOFreq);
   }

   WALBERLA_LOG_INFO_ON_ROOT("Collision Response properties:");
   WALBERLA_LOG_INFO_ON_ROOT(" - collision time = " << collisionTime);
   WALBERLA_LOG_INFO_ON_ROOT(" - coeff of restitution = " << restitutionCoeff);
   WALBERLA_LOG_INFO_ON_ROOT(" - coeff of friction = " << frictionCoeff);

   WALBERLA_LOG_INFO_ON_ROOT("Coupling properties:");
   WALBERLA_LOG_INFO_ON_ROOT(" - number of RPD sub cycles = " << numRPDSubCycles);
   WALBERLA_LOG_INFO_ON_ROOT(" - lubrication correction = " << (useLubricationCorrection ? "yes" : "no"));
   if (useLubricationCorrection)
   {
      WALBERLA_LOG_INFO_ON_ROOT("  - lubrication correction cut off normal = " << lubricationCutOffDistanceNormal);
      WALBERLA_LOG_INFO_ON_ROOT("  - lubrication correction cut off tangential translational = "
                                << lubricationCutOffDistanceTangentialTranslational);
      WALBERLA_LOG_INFO_ON_ROOT("  - lubrication correction cut off tangential rotational = "
                                << lubricationCutOffDistanceTangentialRotational);
      WALBERLA_LOG_INFO_ON_ROOT("  - lubrication correction minimal gap size non dim = "
                                << lubricationMinimalGapSizeNonDim
                                << " ( = " << lubricationMinimalGapSizeNonDim * diameter * real_t(0.5) << " cells )");
   }
   WALBERLA_LOG_INFO_ON_ROOT(
      " - average forces over two LBM time steps = " << (averageForceTorqueOverTwoTimeSteps ? "yes" : "no"));

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   Vector3< uint_t > cellsPerBlockPerDirection(domainSize[0] / numberOfBlocksPerDirection[0],
                                               domainSize[1] / numberOfBlocksPerDirection[1],
                                               domainSize[2] / numberOfBlocksPerDirection[2]);
   for (uint_t i = 0; i < 3; ++i)
   {
      WALBERLA_CHECK_EQUAL(cellsPerBlockPerDirection[i] * numberOfBlocksPerDirection[i], domainSize[i],
                           "Unmatching domain decomposition in direction " << i << "!");
   }

   auto blocks = blockforest::createUniformBlockGrid(numberOfBlocksPerDirection[0], numberOfBlocksPerDirection[1],
                                                     numberOfBlocksPerDirection[2], cellsPerBlockPerDirection[0],
                                                     cellsPerBlockPerDirection[1], cellsPerBlockPerDirection[2], dx, 0,
                                                     false, false, true, true, false, // periodicity
                                                     false);

   WALBERLA_LOG_INFO_ON_ROOT("Domain decomposition:");
   WALBERLA_LOG_INFO_ON_ROOT(" - blocks per direction = " << numberOfBlocksPerDirection);
   WALBERLA_LOG_INFO_ON_ROOT(" - cells per block = " << cellsPerBlockPerDirection);

   // write domain decomposition to file
   if (vtkIOFreq > 0) { vtk::writeDomainDecomposition(blocks, "initial_domain_decomposition", baseFolder); }

   //////////////////
   // RPD COUPLING //
   //////////////////

   auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

   // init data structures
   auto ps                  = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto ss                  = walberla::make_shared< mesa_pd::data::ShapeStorage >();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);

   // bounding planes
   createPlaneSetup(ps, ss, blocks->getDomain(), applyOutflowBCAtTop);

   // create sphere and store Uid
   Vector3< real_t > initialPosition(real_t(0.5) * real_c(domainSize[0]), real_t(0.5) * real_c(domainSize[1]),
                                     initialSpherePosition);
   auto sphereShape = ss->create< mesa_pd::data::Sphere >(diameter * real_t(0.5));
   ss->shapes[sphereShape]->updateMassAndInertia(densitySphere);

   walberla::id_t sphereUid = 0;
   // create sphere
   if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), initialPosition))
   {
      mesa_pd::data::Particle&& p = *ps->create();
      p.setPosition(initialPosition);
      p.setInteractionRadius(diameter * real_t(0.5));
      p.setOwner(mpi::MPIManager::instance()->rank());
      p.setShapeID(sphereShape);
      p.setType(0);
      sphereUid = p.getUid();
   }
   mpi::allReduceInplace(sphereUid, mpi::SUM);

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // add PDF field
   BlockDataID pdfFieldID =
      field::addToStorage< PdfField_T >(blocks, "pdf field (fzyx)", real_c(std::nan("")), field::fzyx);
   BlockDataID pdfFieldGPUID = gpu::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldID, "pdf field GPU");

   // add flag field
   BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

   // add boundary handling
   BlockDataID densityFieldID = field::addToStorage< DensityField_T >(blocks, "Density", real_t(0), field::fzyx);
   BlockDataID velFieldID     = field::addToStorage< VelocityField_T >(blocks, "Velocity", real_t(0), field::fzyx);

   // assemble boundary block string
   std::string boundariesBlockString = " Boundaries { Border { direction B;    walldistance -1;  flag NoSlip; }";

   if (applyOutflowBCAtTop)
   {
      boundariesBlockString += "Border { direction T;    walldistance -1;  flag SimplePressure; }";
   }
   else { boundariesBlockString += "Border { direction T;    walldistance -1;  flag NoSlip; }"; }

   boundariesBlockString += "}";

   WALBERLA_ROOT_SECTION()
   {
      std::ofstream boundariesFile("boundaries.prm");
      boundariesFile << boundariesBlockString;
      boundariesFile.close();
   }
   WALBERLA_MPI_BARRIER()

   auto boundariesCfgFile = Config();
   boundariesCfgFile.readParameterFile("boundaries.prm");
   auto boundariesConfig = boundariesCfgFile.getBlock("Boundaries");

   // map boundaries into the LBM simulation
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldID, boundariesConfig);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldID, Fluid_Flag);

   // set up RPD functionality
   std::function< void(void) > syncCall = [&ps, &rpdDomain]() {
      mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
      syncNextNeighborFunc(*ps, *rpdDomain);
   };

   syncCall();

   real_t timeStepSizeRPD = real_t(1) / real_t(numRPDSubCycles);
   mesa_pd::kernel::ExplicitEuler explEulerIntegrator(timeStepSizeRPD);
   mesa_pd::kernel::VelocityVerletPreForceUpdate vvIntegratorPreForce(timeStepSizeRPD);
   mesa_pd::kernel::VelocityVerletPostForceUpdate vvIntegratorPostForce(timeStepSizeRPD);

   // linear model
   mesa_pd::kernel::LinearSpringDashpot linearCollisionResponse(1);
   linearCollisionResponse.setStiffnessAndDamping(0, 0, restitutionCoeff, collisionTime, kappa, Mij);
   // linearCollisionResponse.setFrictionCoefficientStatic(0,0,frictionCoeff); // not used in this test case
   linearCollisionResponse.setFrictionCoefficientDynamic(0, 0, frictionCoeff);

   mesa_pd::mpi::ReduceProperty reduceProperty;
   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;

   // set up coupling functionality
   Vector3< real_t > gravitationalForce = (densitySphere - densityFluid) * sphereVolume * gravitationalAccelerationVec;
   lbm_mesapd_coupling::AddForceOnParticlesKernel addGravitationalForce(gravitationalForce);
   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
   lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;
   lbm_mesapd_coupling::LubricationCorrectionKernel lubricationCorrectionKernel(
      viscosity, [lubricationMinimalGapSizeNonDim](real_t r) { return lubricationMinimalGapSizeNonDim * r; },
      lubricationCutOffDistanceNormal, lubricationCutOffDistanceTangentialTranslational,
      lubricationCutOffDistanceTangentialRotational);

   ///////////////
   // TIME LOOP //
   ///////////////

   // add particle and volume fraction data structures
   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA(
      blocks, lbm::collision_model::omegaFromViscosity(viscosity));
   // map particles and calculate solid volume fraction initially
   PSMSweepCollectionGPU psmSweepCollection(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                            particleAndVolumeFractionSoA, 1);
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollection.particleMappingSweep(&(*blockIt));
   }

   pystencils::InitializeDomainForPSM pdfSetter(
      particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
      particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0), real_t(0), real_t(0),
      real_t(1.0), real_t(0), real_t(0), real_t(0));

   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      // pdfSetter requires particle velocities at cell centers
      psmSweepCollection.setParticleVelocitiesSweep(&(*blockIt));
      pdfSetter(&(*blockIt));
   }

   lbm::PSM_NoSlip noSlip(blocks, pdfFieldGPUID);
   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldID, NoSlip_Flag, Fluid_Flag);
   lbm::PSM_Density density(blocks, pdfFieldGPUID, real_t(1));
   density.fillFromFlagField< FlagField_T >(blocks, flagFieldID, SimplePressure_Flag, Fluid_Flag);

   // setup of the LBM communication for synchronizing the pdf field between neighboring blocks
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, 1, false);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldGPUID));
   auto communication = std::function< void() >([&]() { com.communicate(nullptr); });

   // create the timeloop
   const uint_t timesteps = uint_c(1000000000); // just some large value

   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   pystencils::PSMSweep PSMSweep(particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
                                 particleAndVolumeFractionSoA.particleForcesFieldID,
                                 particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0.0),
                                 real_t(0.0), real_t(0.0), omega);

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   pystencils::PSM_MacroGetter getterSweep(densityFieldID, pdfFieldID, velFieldID, real_t(0.0), real_t(0.0),
                                           real_t(0.0));

   // vtk output
   if (vtkIOFreq != uint_t(0))
   {
      // sphere
      auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleAngularVelocity >("angular velocity");
      particleVtkOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt) {
         return pIt->getShapeID() == sphereShape;
      }); // limit output to sphere
      auto particleVtkWriter =
         vtk::createVTKOutput_PointData(particleVtkOutput, "Particles", vtkIOFreq, baseFolder, "simulation_step");
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(particleVtkWriter), "VTK (sphere data)");

      // pdf field, as slice
      auto pdfFieldVTK = vtk::createVTKOutput_BlockData(blocks, "fluid_field", vtkIOFreq, 0, false, baseFolder);

      pdfFieldVTK->addBeforeFunction(communication);
      pdfFieldVTK->addBeforeFunction([&]() {
         gpu::fieldCpy< PdfField_T, gpu::GPUField< real_t > >(blocks, pdfFieldID, pdfFieldGPUID);
         for (auto& block : *blocks)
            getterSweep(&block);
      });

      AABB sliceAABB(real_t(0), real_c(domainSize[1]) * real_t(0.5) - real_t(1), real_t(0), real_c(domainSize[0]),
                     real_c(domainSize[1]) * real_t(0.5) + real_t(1), real_c(domainSize[2]));
      vtk::AABBCellFilter aabbSliceFilter(sliceAABB);

      field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldID);
      fluidFilter.addFlag(Fluid_Flag);

      vtk::ChainedFilter combinedSliceFilter;
      combinedSliceFilter.addFilter(fluidFilter);
      combinedSliceFilter.addFilter(aabbSliceFilter);

      pdfFieldVTK->addCellInclusionFilter(combinedSliceFilter);

      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< VelocityField_T > >(velFieldID, "Velocity"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< DensityField_T > >(densityFieldID, "Density"));

      auto flagFieldVTK =
         vtk::createVTKOutput_BlockData(blocks, "flag_field", vtkIOFreq, FieldGhostLayers, false, baseFolder);
      flagFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< FlagField_T > >(flagFieldID, "FlagField"));
      vtk::writeFiles(flagFieldVTK)();

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(pdfFieldVTK), "VTK (fluid field data)");
   }

   // add LBM communication function and boundary handling sweep (does the hydro force calculations and the no-slip
   // treatment)
   timeloop.add() << BeforeFunction(communication, "LBM Communication")
                  << Sweep(deviceSyncWrapper(density.getSweep()), "Boundary Handling (Density)");
   timeloop.add() << Sweep(deviceSyncWrapper(noSlip.getSweep()), "Boundary Handling (NoSlip)");

   // stream + collide LBM step
   addPSMSweepsToTimeloop(timeloop, psmSweepCollection, PSMSweep);

   // evaluation functionality
   std::string loggingFileName(baseFolder + "/LoggingObliqueWetCollision_");
   loggingFileName += std::to_string(caseNumber);
   loggingFileName += "_ratio_" + std::to_string(impactRatio) + "_";
   std::string executableName = argv[0];
   size_t lastSlash           = executableName.find_last_of("/\\");
   if (lastSlash != std::string::npos) { loggingFileName += executableName.substr(lastSlash + 1) + ".txt"; }
   else { loggingFileName += executableName + ".txt"; }
   std::string forceLoggingFileName(baseFolder + "/ForceLoggingObliqueWetCollision_");
   forceLoggingFileName += std::to_string(caseNumber);
   forceLoggingFileName += "_ratio_" + std::to_string(impactRatio) + "_";
   if (lastSlash != std::string::npos) { forceLoggingFileName += executableName.substr(lastSlash + 1) + ".txt"; }
   else { forceLoggingFileName += executableName + ".txt"; }
   if (fileIO)
   {
      WALBERLA_LOG_INFO_ON_ROOT(" - writing logging output to file \"" << loggingFileName << "\"");
      WALBERLA_LOG_INFO_ON_ROOT(" - writing force logging output to file \"" << forceLoggingFileName << "\"");
   }
   SpherePropertyLogger< ParticleAccessor_T > logger(accessor, sphereUid, loggingFileName, forceLoggingFileName, fileIO,
                                                     diameter, uIn, impactRatio, numRPDSubCycles, gravitationalForce[0],
                                                     gravitationalForce[2]);

   ////////////////////////
   // EXECUTE SIMULATION //
   ////////////////////////

   WcTimingPool timeloopTiming;

   // evaluation quantities
   uint_t numBounces = uint_t(0);
   uint_t tImpact    = uint_t(0);

   real_t curVel(real_t(0));
   real_t oldVel(real_t(0));
   real_t maxSettlingVel(real_t(0));

   real_t minGapSize(real_t(0));

   real_t actualSt(real_t(0));
   real_t actualRe(real_t(0));

   WALBERLA_LOG_INFO_ON_ROOT("Running for maximum of " << timesteps << " timesteps!");

   const bool useOpenMP = false;

   uint_t averagingSampleSize = uint_c(real_t(1) / uIn);
   std::vector< Vector3< real_t > > forceValues(averagingSampleSize, Vector3< real_t >(real_t(0)));

   // generally: z: normal direction, x: tangential direction

   uint_t endTimestep = timesteps; // will be adapted after bounce
   // time loop
   for (uint_t i = 0; i < endTimestep; ++i)
   {
      // perform a single simulation step -> this contains LBM and setting of the hydrodynamic interactions
      timeloop.singleStep(timeloopTiming);

      reduceProperty.operator()< mesa_pd::HydrodynamicForceTorqueNotification >(*ps);

      if (averageForceTorqueOverTwoTimeSteps)
      {
         if (i == 0)
         {
            lbm_mesapd_coupling::InitializeHydrodynamicForceTorqueForAveragingKernel
               initializeHydrodynamicForceTorqueForAveragingKernel;
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor,
                                initializeHydrodynamicForceTorqueForAveragingKernel, *accessor);
         }
         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, averageHydrodynamicForceTorque,
                             *accessor);
      }

      Vector3< real_t > hydForce(real_t(0));
      Vector3< real_t > lubForce(real_t(0));
      Vector3< real_t > collisionForce(real_t(0));

      Vector3< real_t > hydTorque(real_t(0));
      Vector3< real_t > lubTorque(real_t(0));
      Vector3< real_t > collisionTorque(real_t(0));

      for (auto subCycle = uint_t(0); subCycle < numRPDSubCycles; ++subCycle)
      {
         timeloopTiming["RPD"].start();

         if (useVelocityVerlet)
         {
            Vector3< real_t > oldForce;
            Vector3< real_t > oldTorque;

            if (artificiallyAccelerateSphere)
            {
               // since the pre-force step of VV updates the position based on velocity and old force, we set oldForce
               // to zero here for this step while artificially accelerating to not perturb the step after which
               // artificial acceleration is switched off (which requires valid oldForce values then) we store the old
               // force and then re-apply it
               auto idx = accessor->uidToIdx(sphereUid);
               if (idx != accessor->getInvalidIdx())
               {
                  oldForce = accessor->getOldForce(idx);
                  accessor->setOldForce(idx, Vector3< real_t >(real_t(0)));

                  oldTorque = accessor->getOldTorque(idx);
                  accessor->setOldTorque(idx, Vector3< real_t >(real_t(0)));
               }
            }

            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPreForce, *accessor);
            syncCall();

            if (artificiallyAccelerateSphere)
            {
               // re-apply old force
               auto idx = accessor->uidToIdx(sphereUid);
               if (idx != accessor->getInvalidIdx())
               {
                  accessor->setOldForce(idx, oldForce);
                  accessor->setOldTorque(idx, oldTorque);
               }
            }
         }

         // lubrication correction
         ps->forEachParticlePairHalf(
            useOpenMP, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
            [&lubricationCorrectionKernel, &rpdDomain](const size_t idx1, const size_t idx2, auto& ac) {
               mesa_pd::collision_detection::AnalyticContactDetection acd;
               acd.getContactThreshold() = lubricationCorrectionKernel.getNormalCutOffDistance();
               mesa_pd::kernel::DoubleCast double_cast;
               mesa_pd::mpi::ContactFilter contact_filter;
               if (double_cast(idx1, idx2, ac, acd, ac))
               {
                  if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                  {
                     double_cast(acd.getIdx1(), acd.getIdx2(), ac, lubricationCorrectionKernel, ac,
                                 acd.getContactNormal(), acd.getPenetrationDepth());
                  }
               }
            },
            *accessor);

         lubForce  = getForce(sphereUid, *accessor);
         lubTorque = getTorque(sphereUid, *accessor);

         // one could add linked cells here

         // collision response
         ps->forEachParticlePairHalf(
            useOpenMP, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
            [&linearCollisionResponse, &rpdDomain, timeStepSizeRPD](const size_t idx1, const size_t idx2, auto& ac) {
               mesa_pd::collision_detection::AnalyticContactDetection acd;
               mesa_pd::kernel::DoubleCast double_cast;
               mesa_pd::mpi::ContactFilter contact_filter;
               if (double_cast(idx1, idx2, ac, acd, ac))
               {
                  if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                  {
                     linearCollisionResponse(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),
                                             acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSizeRPD);
                  }
               }
            },
            *accessor);

         collisionForce  = getForce(sphereUid, *accessor) - lubForce;
         collisionTorque = getTorque(sphereUid, *accessor) - lubTorque;

         reduceAndSwapContactHistory(*ps);

         // add hydrodynamic force
         lbm_mesapd_coupling::AddHydrodynamicInteractionKernel addHydrodynamicInteraction;
         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, addHydrodynamicInteraction,
                             *accessor);

         hydForce = getForce(sphereUid, *accessor) - lubForce - collisionForce;
         WALBERLA_ASSERT(!std::isnan(hydForce[0]) && !std::isnan(hydForce[1]) && !std::isnan(hydForce[2]),
                         "Found nan value in hydrodynamic force = " << hydForce);
         hydTorque = getTorque(sphereUid, *accessor) - lubTorque - collisionTorque;

         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, addGravitationalForce, *accessor);

         reduceProperty.operator()< mesa_pd::ForceTorqueNotification >(*ps);

         // integration
         if (useVelocityVerlet)
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPostForce, *accessor);
         else
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, explEulerIntegrator, *accessor);

         syncCall();

         if (artificiallyAccelerateSphere)
         {
            lbm_mesapd_coupling::RegularParticlesSelector sphereSelector;
            real_t newSphereVel = uIn * (std::exp(-accelerationFactor * real_t(i) / responseTime) - real_t(1));
            ps->forEachParticle(
               useOpenMP, sphereSelector, *accessor,
               [newSphereVel, impactAngle](const size_t idx, ParticleAccessor_T& ac) {
                  ac.setLinearVelocity(
                     idx, Vector3< real_t >(-std::sin(impactAngle), real_t(0), std::cos(impactAngle)) * newSphereVel);
                  ac.setAngularVelocity(idx, Vector3< real_t >(real_t(0)));
               },
               *accessor);
         }

         timeloopTiming["RPD"].end();

         // logging
         timeloopTiming["Logging"].start();
         logger(i, subCycle, hydForce, lubForce, collisionForce, hydTorque, lubTorque, collisionTorque);
         timeloopTiming["Logging"].end();
      }

      // store hyd force to average
      forceValues[i % averagingSampleSize] = hydForce;

      ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor);

      // check for termination
      oldVel = curVel;
      curVel = logger.getSettlingVelocity();

      maxSettlingVel = std::min(maxSettlingVel, curVel);
      minGapSize     = std::min(minGapSize, logger.getGapSize());

      // detect the bounce
      if (oldVel < real_t(0) && curVel > real_t(0) && logger.getGapSize() < real_t(1))
      {
         ++numBounces;

         actualSt = densityRatio * std::abs(maxSettlingVel) * diameter / (real_t(9) * viscosity);
         actualRe = std::abs(maxSettlingVel) * diameter / viscosity;

         WALBERLA_LOG_INFO_ON_ROOT("Detected bounce with max settling velocity " << maxSettlingVel
                                                                                 << " -> St = " << actualSt);

         // end simulation after one non-dim timestep
         uint_t remainingTimeSteps = uint_t(real_t(1) * diameter / std::abs(maxSettlingVel));
         endTimestep               = i + remainingTimeSteps;
         WALBERLA_LOG_INFO_ON_ROOT("Will terminate simulation after "
                                   << remainingTimeSteps << " time steps, i.e. at time step " << endTimestep);
      }

      // impact times are measured when the contact between sphere and wall is broken up again
      if (tImpact == uint_t(0) && numBounces == uint_t(1) && logger.getGapSize() > real_t(0))
      {
         tImpact = i;
         WALBERLA_LOG_INFO_ON_ROOT("Detected impact time at time step " << tImpact);
      }

      // check if sphere is close to bottom plane
      if (logger.getGapSize() < real_t(1) * diameter && artificiallyAccelerateSphere)
      {
         WALBERLA_LOG_INFO_ON_ROOT("Switching off acceleration!");
         artificiallyAccelerateSphere = false;

         if (applyArtificialGravityAfterAccelerating)
         {
            // to avoid a too large deceleration due to the missing gravitational force, we apply the averaged
            // hydrodynamic force (that would have been balanced by the gravitational force) as a kind of artificial
            // gravity
            Vector3< real_t > artificialGravitationalForce =
               -std::accumulate(forceValues.begin(), forceValues.end(), Vector3< real_t >(real_t(0))) /
               real_t(averagingSampleSize);
            WALBERLA_LOG_INFO_ON_ROOT("Applying artificial gravitational and buoyancy force of "
                                      << artificialGravitationalForce);
            real_t actualGravitationalAcceleration =
               -artificialGravitationalForce[2] / ((densitySphere - densityFluid) * sphereVolume);
            WALBERLA_LOG_INFO_ON_ROOT("This would correspond to a gravitational acceleration of g = "
                                      << actualGravitationalAcceleration
                                      << ", g_SI = " << actualGravitationalAcceleration * dx_SI / (dt_SI * dt_SI));
            addGravitationalForce = lbm_mesapd_coupling::AddForceOnParticlesKernel(artificialGravitationalForce);
         }
      }
   }

   WALBERLA_LOG_INFO_ON_ROOT("Terminating simulation");
   WALBERLA_LOG_INFO_ON_ROOT("Maximum settling velocities: " << maxSettlingVel);

   std::string summaryFile(baseFolder + "/Summary.txt");
   WALBERLA_LOG_INFO_ON_ROOT("Writing summary file " << summaryFile);
   WALBERLA_ROOT_SECTION()
   {
      std::ofstream file;
      file.open(summaryFile.c_str());

      file << "waLBerla Revision = " << std::string(WALBERLA_GIT_SHA1).substr(0, 8) << "\n";
      file << "\nInput parameters:\n";
      file << "case: " << caseNumber << "\n";
      file << "fluid: " << fluid << "\n";
      file << "material: " << material << "\n";
      file << "variant: " << simulationVariant << "\n";
      file << "LBM parameters:\n";
      file << "Collision parameters:\n";
      file << " - subCycles = " << numRPDSubCycles << "\n";
      file << " - collision time (Tc) = " << collisionTime << "\n";
      file << "use lubrication correction = " << useLubricationCorrection << "\n";
      file << " - minimum gap size non dim = " << lubricationMinimalGapSizeNonDim << "\n";
      file << " - lubrication correction cut off normal = " << lubricationCutOffDistanceNormal << "\n";
      file << " - lubrication correction cut off tangential translational = "
           << lubricationCutOffDistanceTangentialTranslational << "\n";
      file << " - lubrication correction cut off tangential rotational = "
           << lubricationCutOffDistanceTangentialRotational << "\n";
      file << "apply outflow BC at top = " << applyOutflowBCAtTop << "\n";

      file << "\nOutput quantities:\n";
      file << "actual St = " << actualSt << "\n";
      file << "actual Re = " << actualRe << "\n";
      file << "impact times = " << tImpact << "\n";
      file << "settling velocity = " << maxSettlingVel << "\n";
      file << "maximal overlap = " << std::abs(minGapSize) << " (" << std::abs(minGapSize) / diameter << "%)\n";

      file.close();
   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}

} // namespace oblique_wet_collision

int main(int argc, char** argv) { oblique_wet_collision::main(argc, argv); }
