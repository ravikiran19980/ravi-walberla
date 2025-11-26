//
// Created by dy94rovu on 6/24/24.
//
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
//! \file thermalPSM.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Ravi Ayyala Somayajula <ravi.k.ayyala@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/grid_generator/SCIterator.h"
#include "core/logging/all.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/math/all.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "geometry/InitBoundaryHandling.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "lbm/PerformanceLogger.h"
#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypesCodegen.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/codegen/PSMSweepCollection.h"
#include "lbm_mesapd_coupling/utility/AddForceOnParticlesKernel.h"
#include "lbm_mesapd_coupling/utility/AddHydrodynamicInteractionKernel.h"
#include "lbm_mesapd_coupling/utility/AverageHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/InitializeHydrodynamicForceTorqueForAveragingKernel.h"
#include "lbm_mesapd_coupling/utility/LubricationCorrectionKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/LinkedCells.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/data/shape/HalfSpace.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDataHandling.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/AssocToBlock.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/InsertParticleIntoLinkedCells.h"
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

#include "sqlite/SQLite.h"

#include "vtk/all.h"

#include "../../utilities/InitializerFunctions.h"
#include "EnergyMacroGetter.h"
#include "FluidMacroGetter.h"
#include "GeneralInfoHeader.h"
#include "PSMFluidSweep.h"
#include "PackInfoFluid.h"
#include "PackInfoEnergy.h"
#include "math.h"
#include <fstream>
#include <iomanip>
#include "../../utilities/settemperaturesweep.h"
#include "randomPoints.cpp"

namespace MaterialTransport
{
///////////
// USING //
///////////

using namespace walberla;
using namespace lbm_mesapd_coupling::psm::gpu;
typedef pystencils::PackInfoFluid PackInfoFluid_T;
typedef pystencils::PackInfoEnergy PackInfoEnergy_T;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   using particleTemperaturesFieldGPU_T        = walberla::gpu::GPUField< real_t >;
#else
   using particleTemperaturesField_T  = GhostLayerField< real_t, MaxParticlesPerCell*1 >;
#endif

///////////
// FLAGS //
///////////

// Fluid Flags
const FlagUID Fluid_Flag("Fluid");
const FlagUID Density_Fluid_Flag("Density_Fluid");
const FlagUID NoSlip_Fluid_Flag("NoSlip_Fluid");
const FlagUID Inflow_Fluid_Flag("Inflow_Fluid");
const FlagUID FreeSlip_Fluid_Flag("Free_Slip_Fluid");


// Energy Flags
const FlagUID Energy_Flag("Energy");
const FlagUID Density_Energy_Flag_dynamic("Density_Energy_dynamic");
const FlagUID Density_Energy_Flag_static_cold("Density_Energy_static_cold");
const FlagUID Density_Energy_Flag_static_hot("Density_Energy_static_hot");
const FlagUID NoSlip_Energy_Flag("NoSlip_Energy");
const FlagUID Inflow_Energy_Flag("Inflow_Energy");
const FlagUID Neumann_Energy_Flag("Neumann_Energy");

void createPlane(const shared_ptr< mesa_pd::data::ParticleStorage >& ps,
                 const shared_ptr< mesa_pd::data::ShapeStorage >& ss, Vector3< real_t > position,
                 Vector3< real_t > normal)
{
   mesa_pd::data::Particle&& p0 = *ps->create(true);
   p0.setPosition(position);
   p0.setInteractionRadius(std::numeric_limits< real_t >::infinity());
   p0.setShapeID(ss->create< mesa_pd::data::HalfSpace >(normal));
   p0.setOwner(mpi::MPIManager::instance()->rank());
   p0.setType(0);
   mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
   mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
}

void createPlaneSetup(const shared_ptr< mesa_pd::data::ParticleStorage >& ps,
                      const shared_ptr< mesa_pd::data::ShapeStorage >& ss, const math::AABB& simulationDomain,
                      bool periodicInX, bool periodicInY, bool periodicInZ,real_t offsetAtInflow, real_t offsetAtOutflow)
{

   if(!periodicInZ)
   {
      createPlane(ps, ss, simulationDomain.minCorner() + Vector3< real_t >(0, 0, offsetAtInflow),
                  Vector3< real_t >(0, 0, 1));
      createPlane(ps, ss, simulationDomain.maxCorner() + Vector3< real_t >(0, 0, offsetAtOutflow),
                  Vector3< real_t >(0, 0, -1));
   }

   if (!periodicInX)
   {
      createPlane(ps, ss, simulationDomain.minCorner(), Vector3< real_t >(1, 0, 0));
      createPlane(ps, ss, simulationDomain.maxCorner(), Vector3< real_t >(-1, 0, 0));
   }

   if (!periodicInY)
   {
      createPlane(ps, ss, simulationDomain.minCorner(), Vector3< real_t >(0, 1, 0));
      createPlane(ps, ss, simulationDomain.maxCorner(), Vector3< real_t >(0, -1, 0));
   }
}

struct ParticleInfo
{
   real_t averageVelocity = 0_r;
   real_t maximumVelocity = 0_r;
   uint_t numParticles    = 0;
   real_t maximumHeight   = 0_r;
   real_t particleVolume  = 0_r;
   real_t heightOfMass    = 0_r;

   void allReduce()
   {
      walberla::mpi::allReduceInplace(numParticles, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(averageVelocity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(maximumVelocity, walberla::mpi::MAX);
      walberla::mpi::allReduceInplace(maximumHeight, walberla::mpi::MAX);
      walberla::mpi::allReduceInplace(particleVolume, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(heightOfMass, walberla::mpi::SUM);

      averageVelocity /= real_c(numParticles);
      heightOfMass /= particleVolume;
   }
};

std::ostream& operator<<(std::ostream& os, ParticleInfo const& m)
{
   return os << "Particle Info: uAvg = " << m.averageVelocity << ", uMax = " << m.maximumVelocity
             << ", numParticles = " << m.numParticles << ", zMax = " << m.maximumHeight << ", Vp = " << m.particleVolume
             << ", zMass = " << m.heightOfMass;
}

template< typename Accessor_T >
ParticleInfo evaluateParticleInfo(const Accessor_T& ac)
{
   static_assert(std::is_base_of_v< mesa_pd::data::IAccessor, Accessor_T >, "Provide a valid accessor");

   ParticleInfo info;
   for (uint_t i = 0; i < ac.size(); ++i)
   {
      if (isSet(ac.getFlags(i), mesa_pd::data::particle_flags::GHOST)) continue;
      if (isSet(ac.getFlags(i), mesa_pd::data::particle_flags::GLOBAL)) continue;

      ++info.numParticles;
      real_t velMagnitude   = std::abs(ac.getLinearVelocity(i)[2]);
      real_t particleVolume = ac.getShape(i)->getVolume();
      real_t height         = ac.getPosition(i)[2];
      info.averageVelocity += velMagnitude;
      info.maximumVelocity = std::max(info.maximumVelocity, velMagnitude);
      info.maximumHeight   = std::max(info.maximumHeight, height);
      info.particleVolume += particleVolume;
      info.heightOfMass += particleVolume * height;
   }

   info.allReduce();

   return info;
}

struct FluidInfo
{
   uint_t numFluidCells   = 0;
   real_t averageVelocity = 0_r;
   real_t maximumVelocity = 0_r;
   real_t averageDensity  = 0_r;
   real_t maximumDensity  = 0_r;

   void allReduce()
   {
      walberla::mpi::allReduceInplace(numFluidCells, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(averageVelocity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(maximumVelocity, walberla::mpi::MAX);
      ;
      walberla::mpi::allReduceInplace(averageDensity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(maximumDensity, walberla::mpi::MAX);

      averageVelocity /= real_c(numFluidCells);
      averageDensity /= real_c(numFluidCells);
   }
};

std::ostream& operator<<(std::ostream& os, FluidInfo const& m)
{
   return os << "Fluid Info: numFluidCells = " << m.numFluidCells << ", uAvg = " << m.averageVelocity
             << ", uMax = " << m.maximumVelocity << ", densityAvg = " << m.averageDensity
             << ", densityMax = " << m.maximumDensity;
}

FluidInfo evaluateFluidInfo(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& densityFieldID,
                            const BlockDataID& velocityFieldID)
{
   FluidInfo info;

   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      auto densityField  = blockIt->getData< DensityField_fluid_T >(densityFieldID);
      auto velocityField = blockIt->getData< VelocityField_fluid_T >(velocityFieldID);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         densityField, ++info.numFluidCells; Vector3< real_t > velocity(
            velocityField->get(x, y, z, 0), velocityField->get(x, y, z, 1), velocityField->get(x, y, z, 2));
         real_t density = densityField->get(x, y, z); real_t velMagnitude = velocity.length();
         info.averageVelocity += velMagnitude; info.maximumVelocity = std::max(info.maximumVelocity, velMagnitude);
         info.averageDensity += density; info.maximumDensity        = std::max(info.maximumDensity, density);)
   }
   info.allReduce();
   return info;
}

void writeVelocityToFile(const ParticleInfo &info, uint_t time, const std::string &filename = "velocity_vs_time_gpu_trt.txt")
{
   // open file in append mode so new results get added each timestep
   std::ofstream file(filename, std::ios::app);

   if (!file.is_open())
   {
      throw std::runtime_error("Could not open file " + filename);
   }

   // write: time  averageVelocity  maximumVelocity
   if(time == 0){
      file << "time averagevel position\n";
   }

   file << std::fixed << std::setprecision(6)
        << time << "  "
        << info.averageVelocity << " "
        << info.heightOfMass << "\n";
}
//////////
// MAIN //
//////////

int main(int argc, char** argv)
{
   Environment env(argc, argv);
   auto cfgFile = env.config();
   if (!cfgFile) { WALBERLA_ABORT("Usage: " << argv[0] << " path-to-configuration-file \n"); }

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   gpu::selectDeviceBasedOnMpiRank();
#endif

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));
   WALBERLA_LOG_INFO_ON_ROOT("compiler flags: " << std::string(WALBERLA_COMPILER_FLAGS));
   WALBERLA_LOG_INFO_ON_ROOT("build machine: " << std::string(WALBERLA_BUILD_MACHINE));
   WALBERLA_LOG_INFO_ON_ROOT(*cfgFile);

   // read all parameters from the config file

   Config::BlockHandle physicalSetup         = cfgFile->getBlock("PhysicalSetup");
   const real_t xSize_SI                   = physicalSetup.getParameter< real_t >("xSize");
   const real_t ySize_SI                   = physicalSetup.getParameter< real_t >("ySize");
   const real_t zSize_SI                   = physicalSetup.getParameter< real_t >("zSize");
   const bool periodicInX                  = physicalSetup.getParameter< bool >("periodicInX");
   const bool periodicInY                  = physicalSetup.getParameter< bool >("periodicInY");
   const bool periodicInZ                  = physicalSetup.getParameter< bool >("periodicInZ");
   const real_t runtime_SI                    = physicalSetup.getParameter< real_t >("runtime");
   const real_t densityFluid_SI               = physicalSetup.getParameter< real_t >("densityFluid");
   const real_t particleDiameter_SI           = physicalSetup.getParameter< real_t >("particleDiameter");
   const real_t densityParticle_SI            = physicalSetup.getParameter< real_t >("densityParticle");
   const real_t particleRe                 = physicalSetup.getParameter< real_t >("particleRe");
   const real_t dynamicFrictionCoefficient = physicalSetup.getParameter< real_t >("dynamicFrictionCoefficient");
   const real_t coefficientOfRestitution   = physicalSetup.getParameter< real_t >("coefficientOfRestitution");
   const real_t collisionTimeFactor        = physicalSetup.getParameter< real_t >("collisionTimeFactor");

   Config::BlockHandle numericalSetup = cfgFile->getBlock("NumericalSetup");
   const real_t dx_SI                 = numericalSetup.getParameter< real_t >("dx");
   const real_t dt_SI          = numericalSetup.getParameter< real_t >("dt");
   const real_t Uc          = numericalSetup.getParameter< real_t >("Uc");
   const uint_t numXBlocks            = numericalSetup.getParameter< uint_t >("numXBlocks");
   const uint_t numYBlocks            = numericalSetup.getParameter< uint_t >("numYBlocks");
   const uint_t numZBlocks            = numericalSetup.getParameter< uint_t >("numZBlocks");
   const bool use2DRefVel            = numericalSetup.getParameter< bool >("use2DRefVel");
   WALBERLA_CHECK_EQUAL(numXBlocks * numYBlocks * numZBlocks, uint_t(MPIManager::instance()->numProcesses()),
                        "When using GPUs, the number of blocks ("
                           << numXBlocks * numYBlocks * numZBlocks << ") has to match the number of MPI processes ("
                           << uint_t(MPIManager::instance()->numProcesses()) << ")");
   if ((periodicInX && numXBlocks == 1) || (periodicInY && numYBlocks == 1) || (periodicInZ && numZBlocks == 1))
   {
      WALBERLA_ABORT("The number of blocks must be greater than 1 in periodic dimensions.")
   }

   const bool useLubricationForces        = numericalSetup.getParameter< bool >("useLubricationForces");
   const uint_t numberOfParticleSubCycles = numericalSetup.getParameter< uint_t >("numberOfParticleSubCycles");
   const bool useIntegrators        = numericalSetup.getParameter< bool >("useIntegrators");
   const Vector3< uint_t > particleSubBlockSize =
      numericalSetup.getParameter< Vector3< uint_t > >("particleSubBlockSize");
   const real_t linkedCellWidthRation = numericalSetup.getParameter< real_t >("linkedCellWidthRation");
   const bool particleBarriers        = numericalSetup.getParameter< bool >("particleBarriers");

   const Vector3< real_t > SingleparticleLocation =
      numericalSetup.getParameter< Vector3< real_t > >("SingleparticleLocation");
   const Vector3< real_t > generationDomainFraction =
      numericalSetup.getParameter< Vector3< real_t > >("generationDomainFraction");
   const Vector3< real_t > particleGenerationSpacing =
      numericalSetup.getParameter<Vector3< real_t >>("particleGenerationSpacing");

   const bool useParticles = numericalSetup.getParameter< bool >("useParticles");

   const bool writeSlice =
      numericalSetup.getParameter< bool >("writeSlice");
   const bool sendDirectlyFromGPU =
      numericalSetup.getParameter< bool >("sendDirectlyFromGPU");

   const real_t resThreshold =
      numericalSetup.getParameter< real_t >("resThreshold");
   const real_t volfraction = numericalSetup.getParameter< real_t >("volfraction");
   const bool randomParticles = numericalSetup.getParameter< bool >("randomParticles");

   Config::BlockHandle TemperatureSetup         = cfgFile->getBlock("TemperatureSetup");
   const real_t Thot_SI           = TemperatureSetup.getParameter< real_t >("Thot");
   const real_t Tcold_SI          = TemperatureSetup.getParameter< real_t >("Tcold");
   const real_t Tref_SI           = TemperatureSetup.getParameter< real_t >("Tref");
   const real_t Tparticle_SI           = TemperatureSetup.getParameter< real_t >("Tparticle");
   const real_t Pr                = TemperatureSetup.getParameter< real_t >("PrandtlNumber");
   const real_t Cp_f_SI                    = TemperatureSetup.getParameter<real_t>("Cpf");
   const real_t Cp_s_SI                    = TemperatureSetup.getParameter<real_t>("Cps");
   const real_t Kr                    = TemperatureSetup.getParameter<real_t>("Kr");
   const real_t Gr                    = TemperatureSetup.getParameter<real_t>("Gr");
   const real_t Qso                    = TemperatureSetup.getParameter<real_t>("Qso");

   Config::BlockHandle outputSetup      = cfgFile->getBlock("Output");
   const real_t infoSpacing_SI          = outputSetup.getParameter< real_t >("infoSpacing");
   const real_t vtkSpacingParticles_SI  = outputSetup.getParameter< real_t >("vtkSpacingParticles");
   const real_t vtkSpacingFluid_SI      = outputSetup.getParameter< real_t >("vtkSpacingFluid");
   const std::string vtkFolder          = outputSetup.getParameter< std::string >("vtkFolder");
   const uint_t performanceLogFrequency = outputSetup.getParameter< uint_t >("performanceLogFrequency");





   // convert SI units to simulation (LBM) units and check setup

   Vector3< uint_t > domainSize(uint_c(std::ceil(xSize_SI / dx_SI)), uint_c(std::ceil(ySize_SI / dx_SI)),
                                uint_c(std::ceil(zSize_SI / dx_SI)));
   WALBERLA_LOG_INFO_ON_ROOT("domain size is " << domainSize);
   WALBERLA_CHECK_FLOAT_EQUAL(real_t(domainSize[0]) * dx_SI, xSize_SI, "domain size in x is not divisible by given dx");
   WALBERLA_CHECK_FLOAT_EQUAL(real_t(domainSize[1]) * dx_SI, ySize_SI, "domain size in y is not divisible by given dx");
   WALBERLA_CHECK_FLOAT_EQUAL(real_t(domainSize[2]) * dx_SI, zSize_SI, "domain size in z is not divisible by given dx");

   Vector3< uint_t > cellsPerBlockPerDirection(domainSize[0] / numXBlocks, domainSize[1] / numYBlocks,
                                               domainSize[2] / numZBlocks);

   WALBERLA_CHECK_EQUAL(domainSize[0], cellsPerBlockPerDirection[0] * numXBlocks,
                        "number of cells in x of " << domainSize[0]
                                                   << " is not divisible by given number of blocks in x direction");
   WALBERLA_CHECK_EQUAL(domainSize[1], cellsPerBlockPerDirection[1] * numYBlocks,
                        "number of cells in y of " << domainSize[1]
                                                   << " is not divisible by given number of blocks in y direction");
   WALBERLA_CHECK_EQUAL(domainSize[2], cellsPerBlockPerDirection[2] * numZBlocks,
                        "number of cells in z of " << domainSize[2]
                                                   << " is not divisible by given number of blocks in z direction");

   WALBERLA_CHECK_GREATER_EQUAL(
      particleDiameter_SI / dx_SI, 5_r,
      "Your numerical resolution is below 5 cells per diameter and thus too small for such simulations!");

   real_t densityRatio           = densityParticle_SI / densityFluid_SI;

   // in simulation units: dt = 1, dx = 1, densityFluid = 1

   const real_t particleDiameter = particleDiameter_SI / dx_SI;
   const real_t particleVolume   = math::pi / 6_r * particleDiameter * particleDiameter * particleDiameter;

   const real_t densityFluid = real_t(1);
   real_t densityParticle    = densityRatio;
   const real_t dx           = real_t(1);

   const uint_t numTimeSteps        = uint_c(std::ceil(runtime_SI / dt_SI));
   const uint_t infoSpacing         = uint_c(std::ceil(infoSpacing_SI / dt_SI));
   const uint_t vtkSpacingParticles = uint_c(std::ceil(vtkSpacingParticles_SI / dt_SI));
   const uint_t vtkSpacingFluid     = uint_c(std::ceil(vtkSpacingFluid_SI / dt_SI));


   const real_t poissonsRatio         = real_t(0.22);
   const real_t kappa                 = real_t(2) * (real_t(1) - poissonsRatio) / (real_t(2) - poissonsRatio);
   const real_t particleCollisionTime = collisionTimeFactor * particleDiameter;

   Vector3< uint_t > domainSizeLB;
   Vector3< real_t > Uinitialize(0, 0,0);

   const real_t domainVolume = real_c(domainSize[0] * domainSize[1] * domainSize[2]);
   const uint_t numParticles = uint_c((volfraction*domainVolume)/(particleVolume));
   WALBERLA_LOG_INFO_ON_ROOT(numParticles << " particles will be created");

   const real_t T_conversion = real_t(1);
   // conversion for the various temperature quantities:
   const real_t rho_0 = densityFluid;
   const real_t Thot = Thot_SI;
   const real_t Tcold = Tcold_SI;
   real_t Cp_f =  Cp_f_SI;
   real_t Cp_s = Cp_s_SI;
   const real_t Tref = Tref_SI;
   const real_t particleTemperature = Tparticle_SI;
   const real_t delta_T = Thot - Tcold;
   const real_t Uchar = Uc;
   const real_t kinematicViscosityLB  = (Uchar*particleDiameter)/(particleRe);
   const real_t omega_f = lbm::collision_model::omegaFromViscosity(kinematicViscosityLB);
   real_t gravitationalAcceleration = (3 * Uchar * Uchar * densityFluid) / (4 * particleDiameter * (densityParticle - densityFluid));

   if (use2DRefVel)
   {
      WALBERLA_LOG_INFO_ON_ROOT("pi value is  " << math::pi);
      gravitationalAcceleration =
         (2 * Uc * Uc * densityFluid) / ( math::pi * particleDiameter * (densityParticle - densityFluid));
   }

   const real_t rho_Cp_ref =
      2 * densityFluid * Cp_f * densityParticle * Cp_s / (densityFluid * Cp_f + densityParticle * Cp_s);

   const real_t rhoCpRef = rho_Cp_ref;
   WALBERLA_LOG_INFO_ON_ROOT("rho cp reference is  " << rhoCpRef);
   const real_t thermalDiffusivityFluid_LB = kinematicViscosityLB / Pr;



   const real_t alphaLB = (Gr * kinematicViscosityLB * kinematicViscosityLB) /
                          (delta_T * particleDiameter * particleDiameter *
                           particleDiameter * gravitationalAcceleration);

   const real_t omegaT_f = lbm::collision_model::omegaFromViscosity(thermalDiffusivityFluid_LB);
   const real_t Qs = (Qso)*densityFluid*Cp_f*Uc*delta_T/particleDiameter;
   const real_t kf = rhoCpRef*thermalDiffusivityFluid_LB;
   const real_t ks = Kr*kf;
   const real_t thermalDiffusivityParticle_LB = ks/rhoCpRef;
   const real_t omegaT_s = lbm::collision_model::omegaFromViscosity(thermalDiffusivityParticle_LB);
   WALBERLA_LOG_INFO_ON_ROOT("Known Quantities are    ");
   WALBERLA_LOG_INFO_ON_ROOT("density particle LB is " << densityParticle);
   WALBERLA_LOG_INFO_ON_ROOT("density fluid LB is " << densityFluid);
   WALBERLA_LOG_INFO_ON_ROOT("Cp particle is " << Cp_s << " Cp fluid is  " << Cp_f);
   WALBERLA_LOG_INFO_ON_ROOT("Particle Reynolds Number Re_ref = " << particleRe);
   WALBERLA_LOG_INFO_ON_ROOT("Grashof Number Gr = " << Gr);
   WALBERLA_LOG_INFO_ON_ROOT("Particle Diameter is = " << particleDiameter);

   WALBERLA_LOG_INFO_ON_ROOT("------------------------------");
   WALBERLA_LOG_INFO_ON_ROOT("Extracted Quantities are;   ");
   WALBERLA_LOG_INFO_ON_ROOT("Reference temperature delta_T = " << delta_T);
   WALBERLA_LOG_INFO_ON_ROOT("Kinematic Viscosity = " << kinematicViscosityLB);
   WALBERLA_LOG_INFO_ON_ROOT("gravitational acceleration is " << gravitationalAcceleration);
   WALBERLA_LOG_INFO_ON_ROOT("Characteristic velocity is " << Uchar);
   WALBERLA_LOG_INFO_ON_ROOT("Energy Relaxation rate  fluid is " << omegaT_f);
   WALBERLA_LOG_INFO_ON_ROOT("Energy Relaxation rate  particle is " << omegaT_s);
   WALBERLA_LOG_INFO_ON_ROOT("Hydrodynamic Relaxation rate  fluid is " << omega_f);
   WALBERLA_LOG_INFO_ON_ROOT("coeff of expansion alphaLB = " << alphaLB);

   WALBERLA_LOG_INFO_ON_ROOT("Sanity checks------------------------------");
   WALBERLA_LOG_INFO_ON_ROOT("Grashof number from parameter file is = "  << Gr);
   WALBERLA_LOG_INFO_ON_ROOT("Grashof number =    " << (alphaLB*gravitationalAcceleration*delta_T*particleDiameter*particleDiameter*particleDiameter)/(kinematicViscosityLB*kinematicViscosityLB));
   WALBERLA_LOG_INFO_ON_ROOT("Prandtl number = " <<  (kinematicViscosityLB/thermalDiffusivityFluid_LB) );
   WALBERLA_LOG_INFO_ON_ROOT("Reynolds number = "  << (Uchar*particleDiameter/kinematicViscosityLB) );
   WALBERLA_LOG_INFO_ON_ROOT("conductivity fluid is "  << kf << " conductivity particle is  " << ks );


   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   shared_ptr< StructuredBlockForest > blocks = blockforest::createUniformBlockGrid(
      numXBlocks, numYBlocks, numZBlocks, cellsPerBlockPerDirection[0], cellsPerBlockPerDirection[1], cellsPerBlockPerDirection[2], real_t(1), uint_t(0),
      false, false, periodicInX, periodicInY, periodicInZ, // periodicity
      false);

   auto simulationDomain = blocks->getDomain();
   /////////////
   // MesaPD  //
   /////////////

   auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

   // Init data structures
   auto ps                  = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto ss                  = walberla::make_shared< mesa_pd::data::ShapeStorage >();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);
   auto sphereShape         = ss->create< mesa_pd::data::Sphere >(particleDiameter * real_t(0.5));
   ss->shapes[sphereShape]->updateMassAndInertia(densityParticle);

   // prevent particles from interfering with inflow and outflow by putting the bounding planes slightly in front
   const real_t planeOffsetFromInflow  = dx;
   const real_t planeOffsetFromOutflow = dx;
   createPlaneSetup(ps, ss, simulationDomain, periodicInX, periodicInY,periodicInZ, planeOffsetFromInflow, planeOffsetFromOutflow);
   // Create spheres

   // Ensure that generation domain is computed correctly
   WALBERLA_CHECK_FLOAT_EQUAL(simulationDomain.xMin(), real_t(0));
   WALBERLA_CHECK_FLOAT_EQUAL(simulationDomain.yMin(), real_t(0));
   WALBERLA_CHECK_FLOAT_EQUAL(simulationDomain.zMin(), real_t(0));

   //const real_t spacing  = uint_c(std::ceil(particleGenerationSpacing / dx_SI));
   const Vector3<real_t> spacingVector(
      uint_c(std::ceil(particleGenerationSpacing[0] / dx_SI)),
      uint_c(std::ceil(particleGenerationSpacing[1] / dx_SI)),
      uint_c(std::ceil(particleGenerationSpacing[2] / dx_SI))
   );

   auto generationDomain = math::AABB::createFromMinMaxCorner(
      math::Vector3< real_t >(simulationDomain.xMax() * (real_t(1) - generationDomainFraction[0]) / real_t(2),
                              simulationDomain.yMax() * (real_t(1) - generationDomainFraction[1]) / real_t(2),
                              simulationDomain.zMax() * (real_t(1) - generationDomainFraction[2]) / real_t(2)),
      math::Vector3< real_t >(simulationDomain.xMax() * (real_t(1) + generationDomainFraction[0]) / real_t(2),
                              simulationDomain.yMax() * (real_t(1) + generationDomainFraction[1]) / real_t(2),
                              simulationDomain.zMax() * (real_t(1) + generationDomainFraction[2]) / real_t(2)));
   if (useParticles && randomParticles == false)
      {
         // Ensure that generation domain is computed correctly
         WALBERLA_LOG_INFO_ON_ROOT("generating particles");
         WALBERLA_CHECK_FLOAT_EQUAL(simulationDomain.xMin(), real_t(0));
         WALBERLA_CHECK_FLOAT_EQUAL(simulationDomain.yMin(), real_t(0));
         WALBERLA_CHECK_FLOAT_EQUAL(simulationDomain.zMin(), real_t(0));
         uint_t nump = 0;
         for (auto pt :
              grid_generator::SCGrid(generationDomain, generationDomain.center(),
                                     spacingVector))
         {
            if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pt))
            {
               mesa_pd::data::Particle&& p = *ps->create();
               p.setPosition(pt);
               p.setInteractionRadius(particleDiameter * real_t(0.5));
               p.setOwner(mpi::MPIManager::instance()->rank());
               p.setShapeID(sphereShape);
               p.setType(1);
               p.setTemperature(particleTemperature);
               p.setLinearVelocity(0.1_r * Vector3< real_t >(math::realRandom(
                                              -Uc, Uc)));
            }
            nump +=1;
            if(nump == numParticles){
               break;
               WALBERLA_LOG_INFO_ON_ROOT("generating particles done");
            }
         }
      }
      if(useParticles && randomParticles ==true){
         const int rank = mpi::MPIManager::instance()->rank();

         std::vector< math::Vector3<real_t> > positions;
         if (rank == 0) {
            const unsigned base_seed = 123456u;           // no rank in the seed!
            std::seed_seq seq{ base_seed };
            std::mt19937 gen(seq);

            // min center-to-center distance = particleDiameter (or a bit more)
            const real_t minCenterDistance = particleDiameter;
            real_t boundarymargin = minCenterDistance/2;
            positions = generatePositionsSimple(generationDomain, numParticles, minCenterDistance,boundarymargin, gen);

            if (positions.size() != numParticles) {
               WALBERLA_ABORT("Requested " << numParticles
                                           << " but only placed " << positions.size()
                                           << " with min spacing " << minCenterDistance
                                           << ". Enlarge domain or reduce spacing.");
            }
         }
         walberla::mpi::broadcastObject(positions);
         uint_t particlecount = 0;
         for (const auto& pos : positions) {
            if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pos)) {
               mesa_pd::data::Particle&& p = *ps->create();
               p.setPosition(pos);
               p.setInteractionRadius(particleDiameter * real_t(0.5));
               p.setOwner(mpi::MPIManager::instance()->rank());
               p.setShapeID(sphereShape);
               p.setType(1);
               p.setTemperature(particleTemperature);
            }
            particlecount += 1;
            if (particlecount == numParticles) break;
         }
      }


   ////////////////////////
   // ADD DATA TO BLOCKS //
   ///////////////////////
   // Setting initial PDFs to nan helps to detect bugs in the initialization/BC handling
   // Depending on WALBERLA_BUILD_WITH_GPU_SUPPORT, pdfFieldCPUGPUID is either a CPU or a CPU field
   BlockDataID velFieldFluidID;
   BlockDataID densityConcentrationFieldID;
   BlockDataID energyFieldID;

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   // Fluid PDFs on GPU
   BlockDataID pdfFieldFluidID =
      field::addToStorage< PdfField_fluid_T >(blocks, "pdf fluid field (fzyx)", real_c(std::nan("")), field::fzyx);
   BlockDataID pdfFieldFluidCPUGPUID =
      gpu::addGPUFieldToStorage< PdfField_fluid_T >(blocks, pdfFieldFluidID, "pdf fluid field GPU");

   // Fluid velocity field on GPU
   velFieldFluidID =
      field::addToStorage< VelocityField_fluid_T >(blocks, "velocity fluid field", real_t(0), field::fzyx);
   BlockDataID velFieldFluidCPUGPUID =
      gpu::addGPUFieldToStorage< VelocityField_fluid_T >(blocks, velFieldFluidID, "velocity fluid field GPU");

   // Concentration Density on GPU
   densityConcentrationFieldID = field::addToStorage< DensityField_concentration_T >(
      blocks, "density concentration field", real_t(0), field::fzyx);
   BlockDataID densityConcentrationFieldCPUGPUID = gpu::addGPUFieldToStorage< DensityField_concentration_T >(
      blocks, densityConcentrationFieldID, "density concentration field GPU");

   // Energy PDFs on GPU
   BlockDataID pdfFieldEnergyID = field::addToStorage< PdfField_energy_T >(
      blocks, "pdf energy field (fzyx)", real_c(std::nan("")), field::fzyx);
   BlockDataID pdfFieldEnergyCPUGPUID =
      gpu::addGPUFieldToStorage< PdfField_energy_T >(blocks, pdfFieldEnergyID, "pdf energy field GPU");

   // Energy density field on GPU
   energyFieldID =
      field::addToStorage< DensityField_energy_T >(blocks, "energy field", real_t(0), field::fzyx);
   BlockDataID energyFieldCPUGPUID =
      gpu::addGPUFieldToStorage< DensityField_energy_T >(blocks, energyFieldID, "energy field GPU");

   // fraction field on GPU
   BlockDataID BFieldID   = field::addToStorage< BField_T >(blocks, "B field CPU", real_t(0), field::fzyx, uint_t(1), true);
   BlockDataID BsFieldID  = field::addToStorage< BsField_T >(blocks, "Bs field CPU", real_t(0), field::fzyx, uint_t(1), true);

   BlockDataID particleTemperaturesFieldID = field::addToStorage< particleTemperaturesField_T >(blocks, "particle temperatures field CPU", real_t(0),
                                                                                                     field::fzyx, uint_t(1), true);
   BlockDataID particleTemperatureFieldCPUGPUID = gpu::addGPUFieldToStorage< particleTemperaturesFieldGPU_T >(
      blocks, "particle forces field GPU", MaxParticlesPerCell, field::fzyx, uint_t(1), true);
#else

   // Fluid PDFs on CPU
   BlockDataID pdfFieldFluidCPUGPUID =
      field::addToStorage< PdfField_fluid_T >(blocks, "pdf fluid field CPU", real_c(std::nan("")), field::fzyx);

   BlockDataID velFieldFluidCPUGPUID =
      field::addToStorage< VelocityField_fluid_T >(blocks, "velocity fluid field CPU", real_t(0), field::fzyx);
   velFieldFluidID =
      field::addToStorage< VelocityField_fluid_T >(blocks, "velocity fluid field", real_t(0), field::fzyx);


   BlockDataID densityConcentrationFieldCPUGPUID = field::addToStorage< DensityField_concentration_T >(
      blocks, "density concentration field", real_t(0), field::fzyx);

   // Energy PDFs on CPU
   BlockDataID pdfFieldEnergyCPUGPUID = field::addToStorage< PdfField_energy_T >(
      blocks, "pdf energy field CPU", real_c(std::nan("")), field::fzyx);

   BlockDataID energyFieldCPUGPUID = field::addToStorage< DensityField_energy_T >(
      blocks, "energy field", real_t(0), field::fzyx);

   BlockDataID particleTemperaturesFieldCPUGPUID = field::addToStorage< particleTemperaturesField_T >(blocks, "particle temperatures field CPU", real_t(0),
                                                                                                field::fzyx, uint_t(1), true);

#endif
   BlockDataID densityFluidFieldID =
      field::addToStorage< DensityField_fluid_T >(blocks, "density fluid field", real_t(0), field::fzyx);
   BlockDataID flagFieldFluidID = field::addFlagFieldToStorage< FlagField_T >(blocks, "fluid flag field");
   BlockDataID flagFieldEnergyID =
      field::addFlagFieldToStorage< FlagField_T >(blocks, "energy flag field");
   BlockDataID olddensityConcentrationFieldCPUGPUID = field::addToStorage< DensityField_concentration_T >(
      blocks, "density concentration field old", real_t(0), field::fzyx);

   // Synchronize particles between the blocks for the correct mapping of ghost particles
   // set up RPD functionality
   std::function< void(void) > syncCall = [&ps, &rpdDomain]() {
      // keep overlap for lubrication
      const real_t overlap = real_t(1.5);
      mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
      syncNextNeighborFunc(*ps, *rpdDomain, overlap);
   };

   syncCall();

   real_t timeStepSizeRPD = real_t(1) / real_t(numberOfParticleSubCycles);
   mesa_pd::kernel::VelocityVerletPreForceUpdate vvIntegratorPreForce(timeStepSizeRPD);
   mesa_pd::kernel::VelocityVerletPostForceUpdate vvIntegratorPostForce(timeStepSizeRPD);
   mesa_pd::kernel::LinearSpringDashpot collisionResponse(2);
   collisionResponse.setFrictionCoefficientDynamic(0, 1, dynamicFrictionCoefficient);
   collisionResponse.setFrictionCoefficientDynamic(1, 1, dynamicFrictionCoefficient);
   real_t massSphere       = densityParticle * particleVolume;
   real_t meffSpherePlane  = massSphere;
   real_t meffSphereSphere = massSphere * massSphere / (real_t(2) * massSphere);
   collisionResponse.setStiffnessAndDamping(0, 1, coefficientOfRestitution, particleCollisionTime, kappa,
                                            meffSpherePlane);
   collisionResponse.setStiffnessAndDamping(1, 1, coefficientOfRestitution, particleCollisionTime, kappa,
                                            meffSphereSphere);
   mesa_pd::kernel::AssocToBlock assoc(blocks->getBlockForestPointer());
   mesa_pd::mpi::ReduceProperty reduceProperty;
   mesa_pd::mpi::ReduceContactHistory reduceAndSwapContactHistory;

   // set up coupling functionality
   Vector3< real_t > gravitationalForce(real_t(0), real_t(0),
                                        -(densityParticle - densityFluid) * gravitationalAcceleration * particleVolume);
   lbm_mesapd_coupling::AddForceOnParticlesKernel addGravitationalForce(gravitationalForce);
   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
   lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;
   lbm_mesapd_coupling::LubricationCorrectionKernel lubricationCorrectionKernel(
      kinematicViscosityLB, [](real_t r) { return (real_t(0.001 + real_t(0.00007) * r)) * r; });

   // Assemble boundary block string
   std::string boundariesBlockString = " BoundariesFluid"
                                       "{"
                                       "Border { direction W;    walldistance -1;  flag NoSlip_Fluid; }"
                                       "Border { direction E;    walldistance -1;  flag NoSlip_Fluid; }";

   if (!periodicInY)
   {
      boundariesBlockString += "Border { direction S;    walldistance -1;  flag NoSlip_Fluid; }"
                               "Border { direction N;    walldistance -1;  flag NoSlip_Fluid; }";
   }

   if (!periodicInZ)
   {
      boundariesBlockString += "Border { direction T;    walldistance -1;  flag Density_Fluid; }"
                               "Border { direction B;    walldistance -1;  flag Inflow_Fluid; }";
   }

   boundariesBlockString += "}";

   boundariesBlockString += "\n BoundariesEnergy";
   boundariesBlockString += "{"
                            "Border { direction W;    walldistance -1;  flag Density_Energy_static_cold; }"
                            "Border { direction E;    walldistance -1;  flag Density_Energy_static_cold; }";

   if (!periodicInY)
   {
      boundariesBlockString += "Border { direction S;    walldistance -1;  flag Density_Energy_static_cold; }"
                               "Border { direction N;    walldistance -1;  flag Density_Energy_static_cold; }";
   }

   if (!periodicInZ)
   {
      boundariesBlockString +=
         "Border { direction T;    walldistance -1;  flag Density_Energy_static_cold; }"
         "Border { direction B;    walldistance -1;  flag Density_Energy_static_cold; }"; // Neumann_Energy
   }
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
   auto boundariesConfigFluid         = boundariesCfgFile.getBlock("BoundariesFluid");
   auto boundariesConfigEnergy = boundariesCfgFile.getBlock("BoundariesEnergy");


   // map boundaries into the fluid field simulation
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldFluidID, boundariesConfigFluid);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldFluidID, Fluid_Flag);
   lbm::BC_Fluid_NoSlip noSlip_fluid_bc(blocks, pdfFieldFluidCPUGPUID);
   noSlip_fluid_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldFluidID, NoSlip_Fluid_Flag, Fluid_Flag);
   lbm::BC_Fluid_UBB ubb_fluid_bc(blocks, pdfFieldFluidCPUGPUID, real_t(0), real_t(0), Uc);
   ubb_fluid_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldFluidID, Inflow_Fluid_Flag, Fluid_Flag);
   lbm::BC_Fluid_FreeSlip freeSlip_fluid_bc(blocks, pdfFieldFluidCPUGPUID);
   freeSlip_fluid_bc.fillFromFlagField<FlagField_T>(blocks, flagFieldFluidID, FreeSlip_Fluid_Flag, Fluid_Flag);
   // map boundaries into the energy field simulation

   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldEnergyID, boundariesConfigEnergy);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldEnergyID, Energy_Flag);


   auto EnergyCallback = [](const Cell& pos, const shared_ptr< StructuredBlockForest >& blocks, IBlock& block,BlockDataID densityFluidFieldID, const real_t Cp_f,const real_t Twall) {


      Cell global_cell;
      blocks->transformBlockLocalToGlobalCell(global_cell, block, pos);
      auto densityFluidField = block.getData< DensityField_fluid_T >(densityFluidFieldID);
      real_t rho_f = densityFluidField->get(global_cell);
      return  1*Cp_f * Twall;

   };

   std::function< real_t(const Cell&, const shared_ptr< StructuredBlockForest >&, IBlock&) >
      dynamic_energy_bc = std::bind(EnergyCallback,std::placeholders::_1,std::placeholders::_2,std::placeholders::_3,densityFluidFieldID,Cp_f, Tref);

   lbm::BC_energy_DiffusionDirichlet_dynamic energy_dynamic_bc(blocks, pdfFieldEnergyCPUGPUID, velFieldFluidCPUGPUID,dynamic_energy_bc);
   energy_dynamic_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldEnergyID,
                                                      Density_Energy_Flag_dynamic, Energy_Flag);

   lbm::BC_Energy_Neumann neumann_energy_bc(blocks, pdfFieldEnergyCPUGPUID);
   neumann_energy_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldEnergyID,
                                                      Neumann_Energy_Flag, Energy_Flag);

   lbm::BC_energy_DiffusionDirichlet_static energy_static_bc_cold(blocks,pdfFieldEnergyCPUGPUID,real_t(densityFluid*Cp_f*Tcold*rhoCpRef));
   energy_static_bc_cold.fillFromFlagField< FlagField_T >(blocks, flagFieldEnergyID,
                                                          Density_Energy_Flag_static_cold, Energy_Flag);

   lbm::BC_energy_DiffusionDirichlet_static energy_static_bc_hot(blocks,pdfFieldEnergyCPUGPUID,real_t(densityFluid*Cp_f*Thot*rhoCpRef));
   energy_static_bc_hot.fillFromFlagField< FlagField_T >(blocks, flagFieldEnergyID,
                                                         Density_Energy_Flag_static_hot, Energy_Flag);

///////////////
// TIME LOOP //
///////////////
#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT

   initFluidField(blocks, velFieldFluidID, Uinitialize,domainSizeLB);



   gpu::fieldCpy< gpu::GPUField< real_t >, VelocityField_fluid_T >(blocks, velFieldFluidCPUGPUID, velFieldFluidID);


   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA_fluid(blocks, omega_f);

   PSMSweepCollection psmSweepCollectionFluid(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                              particleAndVolumeFractionSoA_fluid, densityConcentrationFieldCPUGPUID,
                                              particleSubBlockSize);

   ParticleAndVolumeFractionSoA_T< 1 > particleAndVolumeFractionSoA_energy(blocks,omegaT_f);
   PSMSweepCollection psmSweepCollectionTemperature(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                                    particleAndVolumeFractionSoA_energy, densityConcentrationFieldCPUGPUID,
                                                    particleSubBlockSize, true);

   SetParticleTemperaturesSweepp settemperatureparticles(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                particleAndVolumeFractionSoA_energy, densityConcentrationFieldCPUGPUID,particleTemperatureFieldCPUGPUID,true);




   pystencils::initializeConcentrationField initializeConcentrationField(BsFieldID,BFieldID,densityConcentrationFieldID,particleTemperaturesFieldID, Tref);



   // Initialize PDFs

   pystencils::InitializeFluidDomain pdfSetterFluid(
      particleAndVolumeFractionSoA_fluid.BsFieldID, particleAndVolumeFractionSoA_fluid.BFieldID, densityConcentrationFieldCPUGPUID,
      particleAndVolumeFractionSoA_fluid.particleVelocitiesFieldID, pdfFieldFluidCPUGPUID, velFieldFluidCPUGPUID, Tref,
      alphaLB, gravitationalAcceleration, real_t(1), rho_0);


   pystencils::InitializeEnergyDomain pdfSetterEnergy(
      particleAndVolumeFractionSoA_energy.BFieldID, densityConcentrationFieldCPUGPUID,
      pdfFieldEnergyCPUGPUID, velFieldFluidCPUGPUID,
      Cp_f,Cp_s,particleTemperature,rhoCpRef,densityFluid, densityParticle);




#else

   initFluidField(blocks, velFieldFluidCPUGPUID, Uinitialize, domainSize);

   // Map particles into the fluid domain
   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA_fluid(blocks, omega_f);
   PSMSweepCollection psmSweepCollectionFluid(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                              particleAndVolumeFractionSoA_fluid, densityConcentrationFieldCPUGPUID,
                                              particleSubBlockSize);

   ParticleAndVolumeFractionSoA_T< 1 > particleAndVolumeFractionSoA_energy(blocks,omegaT_f);
   PSMSweepCollection psmSweepCollectionTemperature(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                                    particleAndVolumeFractionSoA_energy, densityConcentrationFieldCPUGPUID,
                                                    particleSubBlockSize, true);
   SetParticleTemperaturesSweepp settemperatureparticles(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                                         particleAndVolumeFractionSoA_energy, densityConcentrationFieldCPUGPUID,particleTemperaturesFieldCPUGPUID,
                                                         true);


   pystencils::initializeConcentrationField initializeConcentrationField(particleAndVolumeFractionSoA_energy.BsFieldID,particleAndVolumeFractionSoA_energy.BFieldID,densityConcentrationFieldCPUGPUID,particleTemperaturesFieldCPUGPUID, 0);

   // fluid density boundary condition:

   lbm::BC_Fluid_Density density_fluid_bc(blocks, particleAndVolumeFractionSoA_fluid.BFieldID,densityConcentrationFieldCPUGPUID,pdfFieldFluidCPUGPUID,Tref,alphaLB,real_t(1),gravitationalAcceleration,rho_0);
   density_fluid_bc.fillFromFlagField<FlagField_T>(blocks, flagFieldFluidID, Density_Fluid_Flag, Fluid_Flag);
   // Initialize PDFs

   pystencils::InitializeFluidDomain pdfSetterFluid(
      particleAndVolumeFractionSoA_fluid.BsFieldID, particleAndVolumeFractionSoA_fluid.BFieldID, densityConcentrationFieldCPUGPUID,
      particleAndVolumeFractionSoA_fluid.particleVelocitiesFieldID, pdfFieldFluidCPUGPUID, velFieldFluidCPUGPUID, Tref,
      alphaLB, gravitationalAcceleration, real_t(1), rho_0);


   pystencils::InitializeEnergyDomain pdfSetterEnergy(
      particleAndVolumeFractionSoA_energy.BFieldID, densityConcentrationFieldCPUGPUID,
      pdfFieldEnergyCPUGPUID, velFieldFluidCPUGPUID,
      Cp_f,Cp_s,particleTemperature,rhoCpRef,densityFluid, densityParticle);


#endif

   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollectionFluid.particleMappingSweep(&(*blockIt));
      psmSweepCollectionTemperature.particleMappingSweep(&(*blockIt));
   }
#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   gpu::fieldCpy< BField_T , BFieldGPU_T >(blocks, BFieldID,
                                                              particleAndVolumeFractionSoA_fluid.BFieldID);
   gpu::fieldCpy< BsField_T , BsFieldGPU_T >(blocks, BsFieldID,particleAndVolumeFractionSoA_fluid.BsFieldID);
#endif
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollectionFluid.setParticleVelocitiesSweep(&(*blockIt));
      settemperatureparticles(&(*blockIt));
#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
      gpu::fieldCpy< particleTemperaturesField_T , particleTemperaturesFieldGPU_T >(blocks, particleTemperaturesFieldID,
                                                                             particleTemperatureFieldCPUGPUID);
#endif

      initializeConcentrationField(&(*blockIt));

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
      gpu::fieldCpy< gpu::GPUField< real_t >, DensityField_concentration_T >(blocks, densityConcentrationFieldCPUGPUID,
                                                                             densityConcentrationFieldID);
#endif
      pdfSetterFluid(&(*blockIt));
      pdfSetterEnergy(&(*blockIt));


   }

   ///////////////////////
   // ADD COMMUNICATION //
   //////////////////////

   // Setup of the fluid LBM communication for synchronizing the fluid pdf field between neighboring blocks
#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   gpu::communication::UniformGPUScheme< Stencil_Fluid_T > com_fluid(blocks, sendDirectlyFromGPU, false);
#else
   walberla::blockforest::communication::UniformBufferedScheme< Stencil_Fluid_T > com_fluid(blocks);
#endif
   com_fluid.addPackInfo(make_shared< PackInfoFluid_T >(pdfFieldFluidCPUGPUID));
   auto communication_fluid = std::function< void() >([&]() { com_fluid.communicate(); });

// Setup of the energy LBM communication for synchronizing the energy pdf field between neighboring blocks
#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   gpu::communication::UniformGPUScheme< Stencil_Energy_T > com_energy(blocks, sendDirectlyFromGPU,
                                                                       false);
#else
   walberla::blockforest::communication::UniformBufferedScheme< Stencil_Energy_T > com_energy(blocks);
#endif
   com_energy.addPackInfo(make_shared< PackInfoEnergy_T >(pdfFieldEnergyCPUGPUID));
   auto communication_energy = std::function< void() >([&]() { com_energy.communicate(); });

   // time loop objects for communication with and without hiding

   SweepTimeloop commTimeloop(blocks->getBlockStorage(), numTimeSteps);
   SweepTimeloop timeloop(blocks->getBlockStorage(), numTimeSteps);

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   // objects to get the macroscopic quantities

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
   pystencils::FluidMacroGetter getterSweep_fluid(BFieldID,densityConcentrationFieldID, densityFluidFieldID,
                                                  pdfFieldFluidID, velFieldFluidID, Tref, alphaLB, gravitationalAcceleration,
                                                  rho_0);

   pystencils::EnergyMacroGetter getterSweep_energy(energyFieldID,
                                                    pdfFieldEnergyID);

   pystencils::compute_temperature_field compute_temperature_field(BFieldID,densityConcentrationFieldID,energyFieldID,Cp_f,Cp_s,densityFluid,densityParticle);
#else
   pystencils::FluidMacroGetter getterSweep_fluid(particleAndVolumeFractionSoA_fluid.BFieldID,densityConcentrationFieldCPUGPUID, densityFluidFieldID,
                                                  pdfFieldFluidCPUGPUID, velFieldFluidCPUGPUID, Tref, alphaLB, gravitationalAcceleration,
                                                  rho_0);

   pystencils::EnergyMacroGetter getterSweep_energy(energyFieldCPUGPUID,
                                                    pdfFieldEnergyCPUGPUID);

   pystencils::compute_temperature_field compute_temperature_field(particleAndVolumeFractionSoA_energy.BFieldID,densityConcentrationFieldCPUGPUID,energyFieldCPUGPUID,Cp_f,Cp_s,densityFluid,densityParticle);

#endif


   // vtk output
   if (vtkSpacingParticles != uint_t(0))
   {
      // particles
      auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleUid >("uid");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleTemperature >("temperature");
      // limit output to process-local spheres
      particleVtkOutput->setParticleSelector([sphereShape](const mesa_pd::data::ParticleStorage::iterator& pIt) {
         return pIt->getShapeID() == sphereShape &&
                !(mesa_pd::data::particle_flags::isSet(pIt->getFlags(), mesa_pd::data::particle_flags::GHOST));
      });
      auto particleVtkWriter = vtk::createVTKOutput_PointData(particleVtkOutput, "particles", vtkSpacingParticles, vtkFolder);
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(particleVtkWriter), "VTK (sphere data)");
   }

   if (vtkSpacingFluid != uint_t(0)){
      // Fields
      auto vtkOutput_Fluid = vtk::createVTKOutput_BlockData(blocks, "vtk files fluid", vtkSpacingFluid, 0, false, vtkFolder);

      vtkOutput_Fluid->addBeforeFunction(communication_fluid);

      vtkOutput_Fluid->addBeforeFunction([&]() {
#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
         gpu::fieldCpy< PdfField_fluid_T, gpu::GPUField< real_t > >(blocks, pdfFieldFluidID, pdfFieldFluidCPUGPUID);
         gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, velFieldFluidID,
                                                                         velFieldFluidCPUGPUID);
         gpu::fieldCpy< DensityField_concentration_T, gpu::GPUField< real_t > >(blocks, densityConcentrationFieldID,
                                                                                densityConcentrationFieldCPUGPUID);
         gpu::fieldCpy< GhostLayerField< real_t, 1 >, BFieldGPU_T >(blocks, BFieldID,
                                                                    particleAndVolumeFractionSoA_fluid.BFieldID);
         gpu::fieldCpy< DensityField_energy_T , gpu::GPUField< real_t > >(blocks, energyFieldID,
                                                                         energyFieldCPUGPUID);
         gpu::fieldCpy< PdfField_energy_T , gpu::GPUField< real_t > >(blocks, pdfFieldEnergyID,
                                                                         pdfFieldFluidCPUGPUID);
         gpu::fieldCpy< particleTemperaturesField_T , gpu::GPUField< real_t > >(blocks, particleTemperaturesFieldID,
                                                                     particleTemperatureFieldCPUGPUID);
#endif
         for (auto& block : *blocks)
         {
            getterSweep_fluid(&block);
         }
      });

      auto vtkOutput_Energy =
         vtk::createVTKOutput_BlockData(blocks, "vtk_files_energy", vtkSpacingFluid, 0, false, vtkFolder);

      vtkOutput_Energy->addBeforeFunction(communication_energy);

      vtkOutput_Energy->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            getterSweep_energy(&block);
         }
      });

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< VelocityField_fluid_T > >(velFieldFluidID, "Fluid Velocity"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< BField_T > >(BFieldID, "OverlapFraction"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< particleTemperaturesField_T > >(particleTemperaturesFieldID, "particle temperature field"));
#else
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< VelocityField_fluid_T > >(velFieldFluidCPUGPUID, "Fluid Velocity"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< BField_T > >(particleAndVolumeFractionSoA_fluid.BFieldID, "OverlapFraction"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< particleTemperaturesField_T > >(particleTemperaturesFieldCPUGPUID, "particle temperature field"));
#endif
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_fluid_T > >(densityFluidFieldID, "Fluid Density"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< FlagField_T > >(flagFieldFluidID, "FluidFlagField"));

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
      vtkOutput_Energy->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_concentration_T > >(densityConcentrationFieldID, "temperature"));
      vtkOutput_Energy->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_energy_T > >(energyFieldID, "energy"));
#else
      vtkOutput_Energy->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_energy_T > >(energyFieldCPUGPUID,"energy"));

      vtkOutput_Energy->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_concentration_T > >(densityConcentrationFieldCPUGPUID, "temperature"));
#endif

      vtkOutput_Energy->addCellDataWriter(
         make_shared< field::VTKWriter< FlagField_T > >(flagFieldEnergyID, "EnergyFlagField"));
      if(!writeSlice)
      {
         timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput_Fluid), "VTK output Fluid");
         timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput_Energy), "VTK output Energy");
      }
      else
      {
         const AABB sliceAABB(real_t(0), real_c(domainSize[1]) * real_t(0.5) - real_t(1), real_t(0),
                              real_c(domainSize[0]), real_c(domainSize[1]) * real_t(0.5) + real_t(1),
                              real_c(domainSize[2]));
         const walberla::vtk::AABBCellFilter aabbSliceFilter(sliceAABB);
         field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldFluidID);
         fluidFilter.addFlag(Fluid_Flag);
         walberla::vtk::ChainedFilter combinedSliceFilter;
         combinedSliceFilter.addFilter(fluidFilter);
         combinedSliceFilter.addFilter(aabbSliceFilter);
         vtkOutput_Fluid->addCellInclusionFilter(combinedSliceFilter);
         timeloop.addFuncBeforeTimeStep(walberla::vtk::writeFiles(vtkOutput_Fluid), "VTK (fluid field data)");

         field::FlagFieldCellFilter< FlagField_T > energyFilter(flagFieldEnergyID);
         energyFilter.addFlag(Energy_Flag);
         walberla::vtk::ChainedFilter combinedSliceFilterEnergy;
         combinedSliceFilterEnergy.addFilter(energyFilter);
         combinedSliceFilterEnergy.addFilter(aabbSliceFilter);
         vtkOutput_Energy->addCellInclusionFilter(combinedSliceFilter);
         timeloop.addFuncBeforeTimeStep(walberla::vtk::writeFiles(vtkOutput_Energy), "VTK (energy field data)");
      }
   }
   if (vtkSpacingFluid != uint_t(0)) { vtk::writeDomainDecomposition(blocks, "domain_decomposition", vtkFolder); }

   ///////////////////////////////////////////////////////////////////////////////////////////////
   // add LBM communication, boundary handling and the LBM sweeps to the time loop  for codegen //
   //////////////////////////////////////////////////////////////////////////////////////////////
   pystencils::PSMFluidSweep psmFluidSweep(
      particleAndVolumeFractionSoA_fluid.BsFieldID, particleAndVolumeFractionSoA_fluid.BFieldID, densityConcentrationFieldCPUGPUID,
      particleAndVolumeFractionSoA_fluid.particleForcesFieldID, particleAndVolumeFractionSoA_fluid.particleVelocitiesFieldID,
      pdfFieldFluidCPUGPUID, velFieldFluidCPUGPUID,Tref, alphaLB, gravitationalAcceleration, omega_f, rho_0);


   pystencils::PSMEnergySweep psmEnergySweep(
      particleAndVolumeFractionSoA_energy.BFieldID,densityConcentrationFieldCPUGPUID,energyFieldCPUGPUID,
      pdfFieldEnergyCPUGPUID,velFieldFluidCPUGPUID,Cp_f,Cp_s,Qs,kf,ks,rhoCpRef, densityFluid, densityParticle);


   timeloop.add() << BeforeFunction(communication_fluid, "LBM fluid Communication")
                  << Sweep(deviceSyncWrapper(noSlip_fluid_bc.getSweep()), "Boundary Handling (No slip fluid)");
   timeloop.add() << Sweep(deviceSyncWrapper(ubb_fluid_bc.getSweep()),
                           "Boundary Handling (fluid ubb)");
   timeloop.add() << Sweep(deviceSyncWrapper(freeSlip_fluid_bc.getSweep()),
                           "Boundary Handling (Free slip fluid)");
   timeloop.add() << Sweep(deviceSyncWrapper(density_fluid_bc.getSweep()), "fluid density boundary condition at top");
   // add the energy to the time loop

   timeloop.add() << BeforeFunction(communication_energy, "LBM energy Communication")
                  << Sweep(deviceSyncWrapper(neumann_energy_bc.getSweep()),
                           "Boundary Handling (Energy Neumann)");

   timeloop.add() << Sweep(deviceSyncWrapper(energy_static_bc_hot.getSweep()),
                           "Boundary Handling (Energy static bc hot)");

   timeloop.add() << Sweep(deviceSyncWrapper(energy_static_bc_cold.getSweep()),
                           "Boundary Handling (Energy static bc cold)");


   //addCHTPSMSweepToTimeloop(timeloop, psmSweepCollectionFluid,psmSweepCollectionTemperature, psmFluidSweep,psmEnergySweep);

   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionFluid.particleMappingSweep), "Particle mapping Fluid"); // uses weighting for hydrodynamics specified in Cmakelists file

   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionFluid.setParticleVelocitiesSweep),
                           "Set particle velocities from fluid sweepcollection");
   //timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionTemperature.particleMappingSweep), "Particle mapping Thermal"); // always uses a weighting of 1

   timeloop.add() << Sweep(deviceSyncWrapper(psmFluidSweep), "PSM Fluid sweep");

   timeloop.add() << Sweep(deviceSyncWrapper(psmEnergySweep), "PSM Energy sweep");



   // after both the sweeps, reduce the particle forces.
   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionFluid.reduceParticleForcesSweep),
                           "Reduce particle forces");



   // Add performance logging for fluid
   lbm::PerformanceLogger< FlagField_T > performanceLoggerFluid(blocks, flagFieldFluidID, Fluid_Flag, performanceLogFrequency);
   if (performanceLogFrequency > 0)
   {
      timeloop.addFuncAfterTimeStep(performanceLoggerFluid, "Evaluate performance logging fluid");
   }

   ////////////////////////
   // EXECUTE SIMULATION //
   ////////////////////////

   WcTimingPool timeloopTiming;
   const bool useOpenMP = true;

   real_t linkedCellWidth = linkedCellWidthRation * particleDiameter;
   mesa_pd::data::LinkedCells linkedCells(rpdDomain->getUnionOfLocalAABBs().getExtended(linkedCellWidth),
                                          linkedCellWidth);
   mesa_pd::kernel::InsertParticleIntoLinkedCells ipilc;

   // time loop
   for (uint_t timeStep = 0; timeStep < numTimeSteps; ++timeStep)
   {
      // perform a single simulation step -> this contains LBM and setting of the hydrodynamic interactions

      timeloop.singleStep(timeloopTiming);
      /*if(timeStep % 1000 ==0){
#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
         gpu::fieldCpy< DensityField_concentration_T, gpu::GPUField< real_t > >(blocks, densityConcentrationFieldID,
                                                                                densityConcentrationFieldCPUGPUID);
         real_t  maxresidual = computeResidual(blocks,
                                              olddensityConcentrationFieldCPUGPUID,densityConcentrationFieldID);
         WALBERLA_LOG_INFO_ON_ROOT("max residual for timestep " << timeStep << " is " << maxresidual);
         if(maxresidual <= real_c(resThreshold)){
            WALBERLA_ABORT("simulation reached minimum residual threshold threshold  " << maxresidual << " aborting");
         }
#else
         real_t  maxresidual = computeResidual(blocks,
                                              olddensityConcentrationFieldCPUGPUID,densityConcentrationFieldCPUGPUID);
         WALBERLA_LOG_INFO_ON_ROOT("max residual for timestep " << timeStep << " is " << maxresidual);
         if(maxresidual <= real_c(resThreshold)){
            WALBERLA_ABORT("simulation reached minimum residual threshold threshold  " << maxresidual << " aborting");
         }
#endif

      }*/

      if (particleBarriers) WALBERLA_MPI_BARRIER();
      timeloopTiming["RPD forEachParticle assoc"].start();
      ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, assoc, *accessor);
      if (particleBarriers) WALBERLA_MPI_BARRIER();
      timeloopTiming["RPD forEachParticle assoc"].end();
      timeloopTiming["RPD reduceProperty HydrodynamicForceTorqueNotification"].start();
      reduceProperty.operator()< mesa_pd::HydrodynamicForceTorqueNotification >(*ps);
      if (particleBarriers) WALBERLA_MPI_BARRIER();
      timeloopTiming["RPD reduceProperty HydrodynamicForceTorqueNotification"].end();

      if (timeStep == 0)
      {
         lbm_mesapd_coupling::InitializeHydrodynamicForceTorqueForAveragingKernel
            initializeHydrodynamicForceTorqueForAveragingKernel;
         timeloopTiming["RPD forEachParticle initializeHydrodynamicForceTorqueForAveragingKernel"].start();
         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor,
                             initializeHydrodynamicForceTorqueForAveragingKernel, *accessor);
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD forEachParticle initializeHydrodynamicForceTorqueForAveragingKernel"].end();
      }
      timeloopTiming["RPD forEachParticle averageHydrodynamicForceTorque"].start();
      ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, averageHydrodynamicForceTorque,
                          *accessor);
      if (particleBarriers) WALBERLA_MPI_BARRIER();
      timeloopTiming["RPD forEachParticle averageHydrodynamicForceTorque"].end();

      for (auto subCycle = uint_t(0); subCycle < numberOfParticleSubCycles; ++subCycle)
      {
         if(useIntegrators)
         {
            timeloopTiming["RPD forEachParticle vvIntegratorPreForce"].start();
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPreForce,
                                *accessor);
            if (particleBarriers) WALBERLA_MPI_BARRIER();
            timeloopTiming["RPD forEachParticle vvIntegratorPreForce"].end();
         }
         timeloopTiming["RPD syncCall"].start();
         syncCall();
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD syncCall"].end();

         timeloopTiming["RPD linkedCells.clear"].start();
         linkedCells.clear();
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD linkedCells.clear"].end();
         timeloopTiming["RPD forEachParticle ipilc"].start();
         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectAll(), *accessor, ipilc, *accessor, linkedCells);
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD forEachParticle ipilc"].end();

         if (useLubricationForces)
         {
            // lubrication correction
            timeloopTiming["RPD forEachParticlePairHalf lubricationCorrectionKernel"].start();
            linkedCells.forEachParticlePairHalf(
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
            if (particleBarriers) WALBERLA_MPI_BARRIER();
            timeloopTiming["RPD forEachParticlePairHalf lubricationCorrectionKernel"].end();
         }

         // collision response
         timeloopTiming["RPD forEachParticlePairHalf collisionResponse"].start();
         linkedCells.forEachParticlePairHalf(
            useOpenMP, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
            [&collisionResponse, &rpdDomain, timeStepSizeRPD](const size_t idx1, const size_t idx2, auto& ac) {
               mesa_pd::collision_detection::AnalyticContactDetection acd;
               mesa_pd::kernel::DoubleCast double_cast;
               mesa_pd::mpi::ContactFilter contact_filter;
               if (double_cast(idx1, idx2, ac, acd, ac))
               {
                  if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                  {
                     collisionResponse(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(),
                                       acd.getContactNormal(), acd.getPenetrationDepth(), timeStepSizeRPD);
                  }
               }
            },
            *accessor);
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD forEachParticlePairHalf collisionResponse"].end();

         timeloopTiming["RPD reduceProperty reduceAndSwapContactHistory"].start();
         reduceAndSwapContactHistory(*ps);
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD reduceProperty reduceAndSwapContactHistory"].end();

         // add hydrodynamic force
         lbm_mesapd_coupling::AddHydrodynamicInteractionKernel addHydrodynamicInteraction;
         timeloopTiming["RPD forEachParticle addHydrodynamicInteraction + addGravitationalForce"].start();
         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, addHydrodynamicInteraction,
                             *accessor);

         ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, addGravitationalForce, *accessor);
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD forEachParticle addHydrodynamicInteraction + addGravitationalForce"].end();

         timeloopTiming["RPD reduceProperty ForceTorqueNotification"].start();
         reduceProperty.operator()< mesa_pd::ForceTorqueNotification >(*ps);
         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["RPD reduceProperty ForceTorqueNotification"].end();

         if(useIntegrators)
         {
            timeloopTiming["RPD forEachParticle vvIntegratorPostForce"].start();
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPostForce,
                                *accessor);
            if (particleBarriers) WALBERLA_MPI_BARRIER();
            timeloopTiming["RPD forEachParticle vvIntegratorPostForce"].end();
         }
      }

      timeloopTiming["RPD syncCall"].start();
      syncCall();
      if (particleBarriers) WALBERLA_MPI_BARRIER();
      timeloopTiming["RPD syncCall"].end();

      timeloopTiming["RPD forEachParticle resetHydrodynamicForceTorque"].start();
      ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque,
                          *accessor);
      if (particleBarriers) WALBERLA_MPI_BARRIER();
      timeloopTiming["RPD forEachParticle resetHydrodynamicForceTorque"].end();

      if (infoSpacing != 0 && timeStep % infoSpacing == 0)
      {
         timeloopTiming["Evaluate infos"].start();

         auto particleInfo = evaluateParticleInfo(*accessor);
         WALBERLA_LOG_INFO_ON_ROOT(particleInfo);
         //if (mpi::MPIManager::instance()->rank() == 0) { (writeVelocityToFile(particleInfo, timeStep)); }

#ifdef WALBERLA_BUILD_WITH_GPU_SUPPORT
         gpu::fieldCpy< PdfField_fluid_T, gpu::GPUField< real_t > >(blocks, pdfFieldFluidID, pdfFieldFluidCPUGPUID);
         gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, velFieldFluidID,
                                                                         velFieldFluidCPUGPUID);
         gpu::fieldCpy< DensityField_concentration_T, gpu::GPUField< real_t > >(blocks, densityConcentrationFieldID,
                                                                                densityConcentrationFieldCPUGPUID);
         gpu::fieldCpy< GhostLayerField< real_t, 1 >, BFieldGPU_T >(blocks, BFieldID,
                                                                    particleAndVolumeFractionSoA_fluid.BFieldID);

         for (auto& block : *blocks)
         {
            getterSweep_fluid(&block);
         }
         auto fluidInfo = evaluateFluidInfo(blocks, densityFluidFieldID, velFieldFluidID);
         WALBERLA_LOG_INFO_ON_ROOT(fluidInfo);
#else
         auto fluidInfo = evaluateFluidInfo(blocks, densityFluidFieldID, velFieldFluidCPUGPUID);
         for (auto& block : *blocks)
         {
            getterSweep_fluid(&block);
         }
         WALBERLA_LOG_INFO_ON_ROOT(fluidInfo);
#endif

         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["Evaluate infos"].end();
      }
   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}
} // namespace MaterialTransport

int main(int argc, char** argv) { MaterialTransport::main(argc, argv); }