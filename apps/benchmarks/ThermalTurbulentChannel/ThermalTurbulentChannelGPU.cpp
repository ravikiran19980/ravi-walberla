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
#include "mesa_pd/mpi/ClearGhostOwnerSync.h"
#include "mesa_pd/mpi/ClearNextNeighborSync.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/mpi/notifications/HydrodynamicForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "sqlite/SQLite.h"

#include "vtk/all.h"

#include "./utilities/InitializerFunctions.h"
#include "FluidMacroGetter.h"
#include "GeneralInfoHeader.h"
#include "PSMFluidSweep.h"
#include "PackInfoFluid.h"
#include "PackInfoTemperature.h"
#ifdef run_with_temperature
#include "TemperatureMacroGetter.h"
#endif
#include "math.h"
#include <fstream>
#include <iomanip>
#include "randomPoints.h"
#include "./utilities/settemperaturesweep.h"
#include "HeatEvaluators.h"
#include "turbulentFlowUtilities.h"
//#include "WelfordVelocity.h"
//#include "WelfordTemperature.h"

namespace MaterialTransport
{
///////////
// USING //
///////////

using namespace walberla;
using namespace lbm_mesapd_coupling::psm::gpu;
typedef pystencils::PackInfoFluid PackInfoFluid_T;
typedef pystencils::PackInfoTemperature PackInfoTemperature_T;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

// Field Types
using ScalarField_T = field::GhostLayerField< real_t, 1 >;
using VectorField_T = field::GhostLayerField< real_t, Stencil_Fluid_T::D >;
using TensorField_T = field::GhostLayerField< real_t, Stencil_Fluid_T::D*Stencil_Fluid_T::D >;
using WelfordSweepVelocity_T = pystencils::WelfordVelocity;
#ifdef run_with_temperature
using WelfordSweepTemperature_T = pystencils::WelfordTemperature;
#endif


///////////
// FLAGS //
///////////

// Fluid Flags
const FlagUID Fluid_Flag("Fluid");
const FlagUID Density_Fluid_Flag("Density_Fluid");
const FlagUID NoSlip_Fluid_Flag("NoSlip_Fluid");
const FlagUID Inflow_Fluid_Flag_top("Inflow_Fluid_Top");
const FlagUID Inflow_Fluid_Flag_bottom("Inflow_Fluid_Bottom");
const FlagUID FreeSlip_Fluid_Flag("Free_Slip_Fluid");


// Temperature Flags
const FlagUID Temperature_Flag("Temperature");
const FlagUID Density_Temperature_Flag_dynamic("Density_Temperature_dynamic");
const FlagUID Density_Temperature_Flag_static_cold("Density_Temperature_static_cold");
const FlagUID Density_Temperature_Flag_static_hot("Density_Temperature_static_hot");
const FlagUID NoSlip_Temperature_Flag("NoSlip_Temperature");
const FlagUID Inflow_Temperature_Flag("Inflow_Temperature");
const FlagUID Neumann_Temperature_Flag("Neumann_Temperature");

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
      real_t velMagnitude   = ac.getLinearVelocity(i)[2];
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
#ifdef run_with_temperature
   real_t maxTemperature  = 0_r;
   real_t minTemperature  = 0_r;
#endif

   void allReduce()
   {
      walberla::mpi::allReduceInplace(numFluidCells, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(averageVelocity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(maximumVelocity, walberla::mpi::MAX);
      ;
      walberla::mpi::allReduceInplace(averageDensity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(maximumDensity, walberla::mpi::MAX);
#ifdef run_with_temperature
      walberla::mpi::allReduceInplace(maxTemperature, walberla::mpi::MAX);
      walberla::mpi::allReduceInplace(minTemperature, walberla::mpi::MIN);
#endif

      averageVelocity /= real_c(numFluidCells);
      averageDensity /= real_c(numFluidCells);
   }
};

std::ostream& operator<<(std::ostream& os, FluidInfo const& m)
{
#ifdef run_with_temperature
   return os << "Fluid Info: numFluidCells = " << m.numFluidCells << ", uAvg = " << m.averageVelocity
             << ", uMax = " << m.maximumVelocity << ", densityAvg = " << m.averageDensity
             << ", densityMax = " << m.maximumDensity << ", TMax = " << m.maxTemperature << ", TMin = " << m.minTemperature;
#else
   return os << "Fluid Info: numFluidCells = " << m.numFluidCells << ", uAvg = " << m.averageVelocity
             << ", uMax = " << m.maximumVelocity << ", densityAvg = " << m.averageDensity
             << ", densityMax = " << m.maximumDensity;
#endif
}

void renameFile(const std::string& oldName, const std::string& newName)
{
   int result = std::rename(oldName.c_str(), newName.c_str());
   if (result != 0)
   {
      WALBERLA_LOG_WARNING_ON_ROOT("Could not rename file " << oldName << " to " << newName
                                                            << " with error code " << result);
   }
}

#ifdef run_with_temperature
FluidInfo evaluateFluidInfo(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& densityFieldID,
                            const BlockDataID& velocityFieldID, const BlockDataID &temperatureFieldID)
{
   FluidInfo info;

   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      auto densityField  = blockIt->getData< DensityField_fluid_T >(densityFieldID);
      auto velocityField = blockIt->getData< VelocityField_fluid_T >(velocityFieldID);
      auto temperatureField = blockIt->getData< DensityField_temperature_T >(temperatureFieldID);

      WALBERLA_FOR_ALL_CELLS_XYZ(
         densityField, ++info.numFluidCells; Vector3< real_t > velocity(
            velocityField->get(x, y, z, 0), velocityField->get(x, y, z, 1), velocityField->get(x, y, z, 2));
         real_t density = densityField->get(x, y, z); real_t velMagnitude = std::abs(velocityField->get(x, y, z, 0));
         real_t temperature = temperatureField->get(x,y,z);
         info.averageVelocity += velMagnitude; info.maximumVelocity = std::max(info.maximumVelocity, velMagnitude);
         info.averageDensity += density; info.maximumDensity        = std::max(info.maximumDensity, density);
         info.maxTemperature = std::max(info.maxTemperature, temperature);
         info.minTemperature = std::min(info.minTemperature, temperature);)

   }
   info.allReduce();
   return info;
}
#else
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
         real_t density = densityField->get(x, y, z); real_t velMagnitude = std::abs(velocityField->get(x, y, z, 0));
         info.averageVelocity += velMagnitude; info.maximumVelocity = std::max(info.maximumVelocity, velMagnitude);
         info.averageDensity += density; info.maximumDensity        = std::max(info.maximumDensity, density);)
   }
   info.allReduce();
   return info;
}
#endif



//////////
// MAIN //
//////////

int main(int argc, char** argv)
{
   Environment env(argc, argv);
   auto cfgFile = env.config();
   if (!cfgFile) { WALBERLA_ABORT("Usage: " << argv[0] << " path-to-configuration-file \n"); }



   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));
   WALBERLA_LOG_INFO_ON_ROOT("compiler flags: " << std::string(WALBERLA_COMPILER_FLAGS));
   WALBERLA_LOG_INFO_ON_ROOT("build machine: " << std::string(WALBERLA_BUILD_MACHINE));
   WALBERLA_LOG_INFO_ON_ROOT(*cfgFile);

   // read all parameters from the config file

   Config::BlockHandle physicalSetup       = cfgFile->getBlock("PhysicalSetup");
   const real_t xSize                      = physicalSetup.getParameter< real_t >("xSize");
   const real_t ySize                      = physicalSetup.getParameter< real_t >("ySize");
   const real_t zSize                      = physicalSetup.getParameter< real_t >("zSize");
   const bool periodicInX                  = physicalSetup.getParameter< bool >("periodicInX");
   const bool periodicInY                  = physicalSetup.getParameter< bool >("periodicInY");
   const bool periodicInZ                  = physicalSetup.getParameter< bool >("periodicInZ");
   const real_t densityFluid               = physicalSetup.getParameter< real_t >("densityFluid");
   const real_t particleDiameter           = physicalSetup.getParameter< real_t >("particleDiameter");
   const real_t densityParticle            = physicalSetup.getParameter< real_t >("densityParticle");
   const real_t dynamicFrictionCoefficient = physicalSetup.getParameter< real_t >("dynamicFrictionCoefficient");
   const real_t coefficientOfRestitution   = physicalSetup.getParameter< real_t >("coefficientOfRestitution");
   const real_t collisionTimeFactor        = physicalSetup.getParameter< real_t >("collisionTimeFactor");
   const uint_t simulationTimeFactor       = physicalSetup.getParameter< uint_t >("simulationTimeFactor");

   Config::BlockHandle numericalSetup = cfgFile->getBlock("NumericalSetup");
   const uint_t numXBlocks            = numericalSetup.getParameter< uint_t >("numXBlocks");
   const uint_t numYBlocks            = numericalSetup.getParameter< uint_t >("numYBlocks");
   const uint_t numZBlocks            = numericalSetup.getParameter< uint_t >("numZBlocks");

   WALBERLA_CHECK_EQUAL(numXBlocks * numYBlocks * numZBlocks, uint_t(MPIManager::instance()->numProcesses()),
                        "When using GPUs, the number of blocks ("
                           << numXBlocks * numYBlocks * numZBlocks << ") has to match the number of MPI processes ("
                           << uint_t(MPIManager::instance()->numProcesses()) << ")");
   /*if ((periodicInX && numXBlocks == 1) || (periodicInY && numYBlocks == 1) || (periodicInZ && numZBlocks == 1))
   {
      WALBERLA_ABORT("The number of blocks must be greater than 1 in periodic dimensions.")
   }*/

   const bool useLubricationForces        = numericalSetup.getParameter< bool >("useLubricationForces");
   const bool useParticles                = numericalSetup.getParameter< bool >("useParticles");
   const uint_t numberOfParticleSubCycles = numericalSetup.getParameter< uint_t >("numberOfParticleSubCycles");
   const bool useIntegrators              = numericalSetup.getParameter< bool >("useIntegrators");
   const Vector3< uint_t > particleSubBlockSize =
      numericalSetup.getParameter< Vector3< uint_t > >("particleSubBlockSize");
   const real_t linkedCellWidthRation = numericalSetup.getParameter< real_t >("linkedCellWidthRation");
   const bool particleBarriers        = numericalSetup.getParameter< bool >("particleBarriers");
   const Vector3< real_t > generationDomainFraction =
      numericalSetup.getParameter< Vector3< real_t > >("generationDomainFraction");

   const real_t volfraction = numericalSetup.getParameter< real_t >("volfraction");
   WALBERLA_CHECK(!useParticles || volfraction > real_t(0),
                  "useParticles is true, but volfraction is equal to 0, must be greater than 0");

   Config::BlockHandle turbulenceSetup   = cfgFile->getBlock("TurbulenceSetup");
   const real_t target_bulk_Reynolds     = turbulenceSetup.getParameter< real_t >("target_bulk_Reynolds");
   const real_t target_friction_Reynolds = turbulenceSetup.getParameter< real_t >("target_friction_Reynolds");
   const real_t target_bulk_velocity     = turbulenceSetup.getParameter< real_t >("target_bulk_velocity");
   const real_t center_line_velocity     = turbulenceSetup.getParameter< real_t >("center_line_velocity");
   const uint_t nTurnovers               = turbulenceSetup.getParameter< uint_t >("nTurnovers");

#ifdef run_with_temperature
   Config::BlockHandle TemperatureSetup = cfgFile->getBlock("TemperatureSetup");
   const real_t Thot                    = TemperatureSetup.getParameter< real_t >("Thot");
   const real_t Tcold                   = TemperatureSetup.getParameter< real_t >("Tcold");
   const real_t Tref                    = TemperatureSetup.getParameter< real_t >("Tref");
   const real_t Tparticle               = TemperatureSetup.getParameter< real_t >("Tparticle");
   const real_t Pr                      = TemperatureSetup.getParameter< real_t >("PrandtlNumber");
   const real_t diffusivityRatio        = TemperatureSetup.getParameter< real_t >("diffusivityRatio");
   const real_t Gr                      = TemperatureSetup.getParameter< real_t >("Gr");
#endif

   Config::BlockHandle vtk_params      = cfgFile->getBlock("vtk_params");
   const uint_t infoSpacing            = vtk_params.getParameter< uint_t >("infoSpacing");
   const uint_t vtkSpacingParticles    = vtk_params.getParameter< uint_t >("vtkSpacingParticles");
   const uint_t vtkSpacingFluid        = vtk_params.getParameter< uint_t >("vtkSpacingFluid");
   const std::string vtkFolder         = vtk_params.getParameter< std::string >("vtkFolder");
   const bool writeSlice               = vtk_params.getParameter< bool >("writeSlice");

   Config::BlockHandle checkpoint_params      = cfgFile->getBlock("checkpoint_params");
   const bool startFromCheckPointFile         = checkpoint_params.getParameter< bool >("startFromCheckPointFile");
   const std::string checkpointingFileName    = checkpoint_params.getParameter< std::string >("checkpointingFileName");
   const uint_t checkPointingFrequency        = checkpoint_params.getParameter< uint_t >("checkPointingFrequency");
   const bool writeContinuousCheckPoints      = checkpoint_params.getParameter< bool >("writeContinuousCheckPoints");

   Config::BlockHandle statisticsParams    = cfgFile->getBlock("statistics_params");
   const uint_t outputFrequency            = statisticsParams.getParameter< uint_t >("outputFrequency");
   const real_t convergenceTolerance       = statisticsParams.getParameter< real_t >("convergenceTolerance");

   Config::BlockHandle performance_params    = cfgFile->getBlock("performance_params");
   const uint_t performanceLogFrequency   = performance_params.getParameter< uint_t >("performanceLogFrequency");
   const bool sendDirectlyFromGPU         = performance_params.getParameter< bool >("sendDirectlyFromGPU");

   const uint_t planeAveragingtimeBlock    = statisticsParams.getParameter< uint_t >("planeAveragingtimeBlock");
   const uint_t heatFluxAveragingtimeBlock = statisticsParams.getParameter< uint_t >("heatFluxAveragingtimeBlock");

   // convert SI units to simulation (LBM) units and check setup

   Vector3< uint_t > domainSize(uint_c(xSize), uint_c(ySize),
                                uint_c(zSize ));
   WALBERLA_LOG_INFO_ON_ROOT("domain size is " << domainSize);
   WALBERLA_CHECK_EQUAL(domainSize[codegen::flow_axis], xSize, "domain size in x is not divisible by given dx");
   WALBERLA_CHECK_EQUAL(domainSize[codegen::wall_axis], ySize, "domain size in y is not divisible by given dx");
   WALBERLA_CHECK_EQUAL(domainSize[codegen::remaining_axis], zSize, "domain size in z is not divisible by given dx");

   Vector3< uint_t > cellsPerBlockPerDirection(domainSize[codegen::flow_axis] / numXBlocks,
                                               domainSize[codegen::wall_axis] / numYBlocks,
                                               domainSize[codegen::remaining_axis] / numZBlocks);

   WALBERLA_CHECK_EQUAL(domainSize[codegen::flow_axis], cellsPerBlockPerDirection[0] * numXBlocks,
                        "number of cells in x of " << domainSize[codegen::flow_axis]
                                                   << " is not divisible by given number of blocks in x direction");
   WALBERLA_CHECK_EQUAL(domainSize[codegen::wall_axis], cellsPerBlockPerDirection[1] * numYBlocks,
                        "number of cells in y of " << domainSize[codegen::wall_axis]
                                                   << " is not divisible by given number of blocks in y direction");
   WALBERLA_CHECK_EQUAL(domainSize[codegen::remaining_axis], cellsPerBlockPerDirection[2] * numZBlocks,
                        "number of cells in z of " << domainSize[codegen::remaining_axis]
                                                   << " is not divisible by given number of blocks in z direction");

   WALBERLA_CHECK_GREATER_EQUAL(
      particleDiameter, 5_r,
      "Your numerical resolution is below 5 cells per diameter and thus too small for such simulations!");

   real_t densityRatio = densityParticle / densityFluid;

   // in simulation units: dt = 1, dx = 1, densityFluid = 1

   const real_t particleVolume   = math::pi / 6_r * particleDiameter * particleDiameter * particleDiameter;
   const real_t poissonsRatio         = real_t(0.22);
   const real_t kappa                 = real_t(2) * (real_t(1) - poissonsRatio) / (real_t(2) - poissonsRatio);
   const real_t particleCollisionTime = collisionTimeFactor * particleDiameter;

   Vector3< uint_t > domainSizeLB;
   Vector3< real_t > Uinitialize(target_bulk_velocity, 0, 0);
   const real_t domainVolume = domainSize[codegen::flow_axis] * domainSize[codegen::wall_axis] *
                               domainSize[codegen::remaining_axis];
   const uint_t numParticles = uint_c((volfraction*domainVolume)/(particleVolume));
   WALBERLA_LOG_INFO_ON_ROOT(numParticles << " particles will be created");
   const real_t T_conversion = real_t(1);
   const real_t channel_half_width = real_c(domainSize[codegen::wall_axis]) / 2.0;
#ifdef run_with_temperature
   // conversion for the various temperature quantities:
   const real_t rho_0               = densityFluid;
   const real_t particleTemperature = Tparticle;
#endif

   // calculation of target friction velocity from the thesis of Eschghinejadfard

   const real_t vonkarman_kappa = 2.5;
   const real_t B = 5.5;
   const real_t target_friction_velocity  = center_line_velocity/(vonkarman_kappa*std::log(target_friction_Reynolds) + B);
   const real_t turnOverPeriod = channel_half_width/target_friction_velocity;

   const real_t kinematicViscosityLB = (target_friction_velocity*channel_half_width)/target_friction_Reynolds;

#ifdef run_with_temperature
   const real_t thermalDiffusivityFluid_LB = kinematicViscosityLB / Pr;
   const real_t thermalDiffusivityParticle_LB = thermalDiffusivityFluid_LB;

#endif
   const real_t omega_f  = lbm::collision_model::omegaFromViscosity(kinematicViscosityLB);
#ifdef run_with_temperature
   const real_t omegaT_f = lbm::collision_model::omegaFromViscosity(thermalDiffusivityFluid_LB);
   const real_t omegaT_s = lbm::collision_model::omegaFromViscosity(thermalDiffusivityParticle_LB);
#endif
   const uint_t numTimeSteps              = uint_c(simulationTimeFactor *turnOverPeriod);
   const uint_t samplingInterval          = uint_c(0.04 * turnOverPeriod);
   uint_t startTimeStep                   = uint_t(0);

   if (startFromCheckPointFile)
   {
      WALBERLA_ROOT_SECTION()
      {
         std::ifstream checkpointConfigIS(checkpointingFileName + "_config.txt");
         if (!checkpointConfigIS.is_open())
         {
            WALBERLA_LOG_WARNING_ON_ROOT("Could not open checkpoint config file " << checkpointingFileName + "_config.txt");
         }
         else
         {
            checkpointConfigIS >> startTimeStep;
            if (!checkpointConfigIS.good() && !checkpointConfigIS.eof())
            {
               WALBERLA_ABORT("Could not parse timestep from checkpoint config file "
                              << checkpointingFileName + "_config.txt");
            }
         }
      }
      walberla::mpi::broadcastObject(startTimeStep);
      WALBERLA_LOG_INFO_ON_ROOT("Restarting simulation from checkpoint at time step " << startTimeStep);
   }




   WALBERLA_LOG_INFO_ON_ROOT("Known Quantities are    ");
   WALBERLA_LOG_INFO_ON_ROOT("density particle LB is " << densityParticle);
   WALBERLA_LOG_INFO_ON_ROOT("density fluid LB is " << densityFluid);
   WALBERLA_LOG_INFO_ON_ROOT("Particle Diameter is = " << particleDiameter);

   WALBERLA_LOG_INFO_ON_ROOT("------------------------------");
   WALBERLA_LOG_INFO_ON_ROOT("Extracted Quantities are;   ");
   WALBERLA_LOG_INFO_ON_ROOT("Target Bulk Velocity is " << target_bulk_velocity);
#ifdef run_with_temperature
   WALBERLA_LOG_INFO_ON_ROOT("Temperature Relaxation rate fluid is " << omegaT_f);
#endif
   WALBERLA_LOG_INFO_ON_ROOT("Hydrodynamic Relaxation rate  fluid is " << omega_f);
   WALBERLA_LOG_INFO_ON_ROOT("Sanity checks------------------------------");
#ifdef run_with_temperature
   WALBERLA_LOG_INFO_ON_ROOT("Prandtl number = " << (kinematicViscosityLB / thermalDiffusivityFluid_LB));
   WALBERLA_LOG_INFO_ON_ROOT("thermal diffusivity alpha fluid = " <<  thermalDiffusivityFluid_LB);
   WALBERLA_LOG_INFO_ON_ROOT("thermal diffusivity alpha particle = " <<  thermalDiffusivityParticle_LB);
#endif



   // outputting turbulent related parameters
   WALBERLA_LOG_INFO_ON_ROOT("target friction Reynolds number = " <<  target_friction_Reynolds);
   WALBERLA_LOG_INFO_ON_ROOT("target_friction_velocity = " <<  target_friction_velocity);
   WALBERLA_LOG_INFO_ON_ROOT("turnOverPeriod = " <<  turnOverPeriod);
   WALBERLA_LOG_INFO_ON_ROOT("sampling interval = " <<  samplingInterval);
   WALBERLA_LOG_INFO_ON_ROOT("kinematic viscosity = " <<  kinematicViscosityLB);
   WALBERLA_LOG_INFO_ON_ROOT("viscous length scale = " <<  kinematicViscosityLB/target_friction_velocity);

   /////////////////////////////////////////////
   //// driving force calculation parameters ///
   ////////////////////////////////////////////

   ForceCalculatorParameters forceParams;
   forceParams.domainSize = Vector3<uint_t>(
      uint_t(xSize),
      uint_t(ySize),
      uint_t(zSize)
   );

   forceParams.wallAxis = codegen::wall_axis;  // wall-normal direction along y

   forceParams.channelHalfWidth = real_c(domainSize[codegen::wall_axis]) / 2.0;

   forceParams.targetBulkVelocity = target_bulk_velocity;

   // example conversion if needed
   forceParams.targetFrictionVelocity = target_friction_velocity;





   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   shared_ptr< StructuredBlockForest > blocks;
   if (startFromCheckPointFile == false)
   {
      blocks = blockforest::createUniformBlockGrid(numXBlocks, numYBlocks, numZBlocks, cellsPerBlockPerDirection[0],
                                                   cellsPerBlockPerDirection[1], cellsPerBlockPerDirection[2],
                                                   real_t(1), uint_t(0), false, false, periodicInX, periodicInY,
                                                   periodicInZ, // periodicity
                                                   false);
      blocks->createCellBoundingBoxes();

      WALBERLA_LOG_INFO_ON_ROOT("Writing block forest to file!");
      blocks->getBlockForest().saveToFile(checkpointingFileName + "_forest.txt");
   }
   else
   {
      blocks = blockforest::createUniformBlockGrid(checkpointingFileName + "_forest.txt",
                                                   cellsPerBlockPerDirection[0], cellsPerBlockPerDirection[1],
                                                   cellsPerBlockPerDirection[2], false);
   }

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
   BlockDataID particleStorageID;

   if (startFromCheckPointFile && std::filesystem::exists(checkpointingFileName + "_mesa.txt"))
   {
      WALBERLA_LOG_INFO_ON_ROOT("Initializing " << numParticles << " particles from checkpointing file!");
      particleStorageID =
         blocks->loadBlockData(checkpointingFileName + "_mesa.txt", mesa_pd::domain::createBlockForestDataHandling(ps),
                               "Particle Storage");
      mesa_pd::mpi::ClearNextNeighborSync clearNextNeighborSync;
      clearNextNeighborSync(*accessor);

      mesa_pd::mpi::ClearGhostOwnerSync CGOS;
      CGOS(*accessor);
   }
   else
   {
      particleStorageID = blocks->addBlockData(mesa_pd::domain::createBlockForestDataHandling(ps), "Particle Storage");
   }

   auto sphereShape         = ss->create< mesa_pd::data::Sphere >(particleDiameter * real_t(0.5));
   ss->shapes[sphereShape]->updateMassAndInertia(densityParticle);

   // prevent particles from interfering with inflow and outflow by putting the bounding planes slightly in front
   if (useParticles && !startFromCheckPointFile && !std::filesystem::exists(checkpointingFileName + "_mesa.txt"))
   {
      const real_t planeOffsetFromInflow  = 1_r;
      const real_t planeOffsetFromOutflow = 1_r;
      createPlaneSetup(ps, ss, simulationDomain, periodicInX, periodicInY, periodicInZ, planeOffsetFromInflow,
                       planeOffsetFromOutflow);
      // Create spheres

      const int rank = mpi::MPIManager::instance()->rank();

      std::vector< math::Vector3<real_t> > positions;
      if (rank == 0)
      {
         const unsigned base_seed = 123456u; // no rank in the seed!
         std::seed_seq seq{ base_seed };
         std::mt19937 gen(seq);

         // min center-to-center distance = particleDiameter (or a bit more)
         const real_t minCenterDistance = particleDiameter;
         real_t boundarymargin          = minCenterDistance / 2;
         positions = generatePositionsSimple(simulationDomain, numParticles, minCenterDistance, boundarymargin, gen);

         if (positions.size() != numParticles)
         {
            WALBERLA_ABORT("Requested " << numParticles << " but only placed " << positions.size()
                                        << " with min spacing " << minCenterDistance
                                        << ". Enlarge domain or reduce spacing.");
         }
      }
      walberla::mpi::broadcastObject(positions);
      uint_t particlecount = 0;
      for (const auto& pos : positions)
      {
         if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pos))
         {
            mesa_pd::data::Particle&& p = *ps->create();
            p.setPosition(pos);
            p.setInteractionRadius(particleDiameter * real_t(0.5));
            p.setOwner(mpi::MPIManager::instance()->rank());
            p.setShapeID(sphereShape);
            p.setType(1);
#ifdef run_with_temperature
            p.setTemperature(particleTemperature);
#endif
         }
         particlecount += 1;
         if (particlecount == numParticles) break;
      }
   }


   ////////////////////////
   // ADD DATA TO BLOCKS //
   ///////////////////////

   // Setting initial PDFs to nan helps to detect bugs in the initialization/BC handling


   ////////////////////////////////////////////////
   // Fluid related fields creation on CPU       //
   ///////////////////////////////////////////////


   BlockDataID pdfFieldFluidID;
   BlockDataID pdfFieldFluidGPUID;

   if (startFromCheckPointFile == true)
   {

      auto dataHandling = make_shared< field::DefaultBlockDataHandling< PdfField_fluid_T > >(
         blocks, uint_t(1), real_c(std::nan("")), field::fzyx);

      pdfFieldFluidID = blocks->loadBlockData(checkpointingFileName + "_lbm.txt", dataHandling, "pdf field",false);
      pdfFieldFluidGPUID = gpu::addGPUFieldToStorage< PdfField_fluid_T >(blocks, pdfFieldFluidID, "pdf fluid field GPU");

      gpu::fieldCpy<gpu::GPUField< real_t > , PdfField_fluid_T>(blocks,pdfFieldFluidGPUID, pdfFieldFluidID );


   }
   else
   {
      pdfFieldFluidID =
         field::addToStorage< PdfField_fluid_T >(blocks, "pdf fluid field CPU", real_c(std::nan("")), field::fzyx);
      pdfFieldFluidGPUID=
      gpu::addGPUFieldToStorage< PdfField_fluid_T >(blocks, pdfFieldFluidID, "pdf fluid field GPU");
   }


   BlockDataID velFieldFluidID =
      field::addToStorage< VelocityField_fluid_T >(blocks, "velocity fluid field CPU", real_t(0), field::fzyx);
   BlockDataID velFieldFluidGPUID =
      gpu::addGPUFieldToStorage< VelocityField_fluid_T >(blocks, velFieldFluidID, "velocity fluid field GPU");

   BlockDataID densityFluidFieldID =
      field::addToStorage< DensityField_fluid_T >(blocks, "density fluid field", real_t(0), field::fzyx);
   BlockDataID densityFluidFieldGPUID =
      gpu::addGPUFieldToStorage< DensityField_fluid_T >(blocks, densityFluidFieldID, "density fluid field GPU");

   BlockDataID flagFieldFluidID       = field::addFlagFieldToStorage< FlagField_T >(blocks, "fluid flag field");

   // Welford fields for shared velocity statistics
   BlockDataID meanVelFieldID =
      field::addToStorage< VelocityField_fluid_T >(blocks, "mean velocity field CPU", real_t(0), field::fzyx);
   BlockDataID meanVelFieldGPUID =
      gpu::addGPUFieldToStorage< VelocityField_fluid_T >(blocks, meanVelFieldID, "mean velocity field GPU");

   BlockDataID sosVelFieldID =
      field::addToStorage< TensorField_T >(blocks, "sum of squares velocity field CPU", real_t(0), field::fzyx);
   BlockDataID sosVelFieldGPUID =
      gpu::addGPUFieldToStorage< TensorField_T >(blocks, sosVelFieldID, "sum of squares velocity field GPU");



#ifdef run_with_temperature
   ////////////////////////////////////////////////
   // Temperature related fields creation on CPU //
   ///////////////////////////////////////////////
   BlockDataID pdfFieldTemperatureID;
   BlockDataID pdfFieldTemperatureGPUID;

   if (startFromCheckPointFile == true)
   {
      auto dataHandlingTemperature = make_shared< field::DefaultBlockDataHandling< PdfField_temperature_T > >(
         blocks, uint_t(1), real_c(std::nan("")), field::fzyx);
      pdfFieldTemperatureID = blocks->loadBlockData(checkpointingFileName + "_lbm_temperature.txt",
                                                    dataHandlingTemperature, "pdf field");
      pdfFieldTemperatureGPUID =
         gpu::addGPUFieldToStorage< PdfField_temperature_T >(blocks, pdfFieldTemperatureID, "pdf temperature field GPU");
      gpu::fieldCpy< PdfField_temperature_T, gpu::GPUField< real_t > >(blocks, pdfFieldTemperatureID,
                                                                       pdfFieldTemperatureGPUID);
   }
   else
   {
      pdfFieldTemperatureID = field::addToStorage< PdfField_temperature_T >(
         blocks, "pdf temperature field CPU", real_c(std::nan("")), field::fzyx);
      pdfFieldTemperatureGPUID =
         gpu::addGPUFieldToStorage< PdfField_temperature_T >(blocks, pdfFieldTemperatureID, "pdf temperature field GPU");
   }

   BlockDataID temperatureFieldID = field::addToStorage< DensityField_temperature_T >(
      blocks, "temperature field", real_t(0), field::fzyx);
   BlockDataID temperatureFieldGPUID =
      gpu::addGPUFieldToStorage< DensityField_temperature_T >(blocks, temperatureFieldID, "temperature field GPU");

   BlockDataID particleTemperaturesFieldID = field::addToStorage< particleTemperaturesField_T >(
      blocks, "particle temperatures field CPU", real_t(0), field::fzyx, uint_t(1), true);
   BlockDataID particleTemperaturesFieldGPUID = gpu::addGPUFieldToStorage< particleTemperaturesField_T >(
      blocks, particleTemperaturesFieldID, "particle temperatures field GPU");


   BlockDataID flagFieldTemperatureID = field::addFlagFieldToStorage< FlagField_T >(blocks, "temperature flag field");


   // temperature fields
   BlockDataID meanTemperatureFieldID =
      field::addToStorage< DensityField_temperature_T >(blocks, "mean temperature field CPU", real_t(0), field::fzyx);
   BlockDataID meanTemperatureFieldGPUID =
      gpu::addGPUFieldToStorage< DensityField_temperature_T >(blocks, meanTemperatureFieldID, "mean temperature field GPU");

   BlockDataID sosTemperatureFieldID =
      field::addToStorage< DensityField_temperature_T >(blocks, "sum of squares temperature field CPU", real_t(0), field::fzyx);
   BlockDataID sosTemperatureFieldGPUID =
      gpu::addGPUFieldToStorage< DensityField_temperature_T >(blocks, sosTemperatureFieldID, "sum of squares temperature field GPU");
   #endif


   /////////////////////////////////////////////////////////
   // other fields creation on CPU required for sampling //
   ///////////////////////////////////////////////////////

   // fraction fields
   BlockDataID BFieldID   = field::addToStorage< BField_T >(blocks, "B field CPU", real_t(0), field::fzyx, uint_t(1), true);
   BlockDataID BsFieldID  = field::addToStorage< BsField_T >(blocks, "Bs field CPU", real_t(0), field::fzyx, uint_t(1), true);


   WcTimer checkpointTimer;

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
                                        -(densityParticle - densityFluid) * real_t(0) * particleVolume);
   lbm_mesapd_coupling::AddForceOnParticlesKernel addGravitationalForce(gravitationalForce);
   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
   lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;
   lbm_mesapd_coupling::LubricationCorrectionKernel lubricationCorrectionKernel(
      kinematicViscosityLB, [](real_t r) { return (real_t(0.001 + real_t(0.00007) * r)) * r; });

   // Assemble boundary block string
   std::string boundariesBlockString = " BoundariesFluid";
   boundariesBlockString             += "{";

   if (!periodicInX)
   {
      boundariesBlockString += "\t Border { direction W;    walldistance -1;  flag NoSlip_Fluid; }\n"
                               "\t Border { direction E;    walldistance -1;  flag NoSlip_Fluid; }\n";
   }

   if (!periodicInY)
   {
      boundariesBlockString += "\t Border { direction S;    walldistance -1;  flag NoSlip_Fluid; }\n"
                               "\t Border { direction N;    walldistance -1;  flag NoSlip_Fluid; }\n";
   }

   if (!periodicInZ)
   {
      boundariesBlockString += "\t Border { direction T;    walldistance -1;  flag NoSlip_Fluid; }\n"
                               "\t Border { direction B;    walldistance -1;  flag NoSlip_Fluid; }\n";
   }

   boundariesBlockString += "}";

   boundariesBlockString += "\n BoundariesTemperature";
   boundariesBlockString +=    "{";

   if(!periodicInX)
   {
      boundariesBlockString += "\t Border { direction W;    walldistance -1;  flag Neumann_Temperature; }\n"
                               "\t Border { direction E;    walldistance -1;  flag Neumann_Temperature; }\n";
   }
   if (!periodicInY)
   {
      boundariesBlockString += "Border { direction S;    walldistance -1;  flag Neumann_Temperature; }"
                               "Border { direction N;    walldistance -1;  flag Neumann_Temperature; }";
   }

   if (!periodicInZ)
   {
      boundariesBlockString +=
         "Border { direction T;    walldistance -1;  flag Density_Temperature_static_cold; }"
         "Border { direction B;    walldistance -1;  flag Density_Temperature_static_hot; }"; // Neumann_Energy
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
   auto boundariesConfigTemperature = boundariesCfgFile.getBlock("BoundariesTemperature");


   // map boundaries into the fluid field simulation
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldFluidID, boundariesConfigFluid);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldFluidID, Fluid_Flag);
   lbm::BC_Fluid_NoSlip noSlip_fluid_bc(blocks, pdfFieldFluidGPUID);
   noSlip_fluid_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldFluidID, NoSlip_Fluid_Flag, Fluid_Flag);

   lbm::BC_Fluid_FreeSlip freeSlip_fluid_bc(blocks, pdfFieldFluidGPUID);
   freeSlip_fluid_bc.fillFromFlagField<FlagField_T>(blocks, flagFieldFluidID, FreeSlip_Fluid_Flag, Fluid_Flag);

   #ifdef run_with_temperature
   // map boundaries into the temperature field simulation

   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldTemperatureID, boundariesConfigTemperature);
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldTemperatureID, Temperature_Flag);



   lbm::BC_Temperature_Neumann neumann_temperature_bc(blocks, pdfFieldTemperatureGPUID);
   neumann_temperature_bc.fillFromFlagField< FlagField_T >(blocks, flagFieldTemperatureID,
                                                      Neumann_Temperature_Flag, Temperature_Flag);

   lbm::BC_Temperature_DiffusionDirichlet_static temperature_static_bc_cold(
      blocks, pdfFieldTemperatureGPUID, real_t(Tcold));
   temperature_static_bc_cold.fillFromFlagField< FlagField_T >(blocks, flagFieldTemperatureID,
                                                          Density_Temperature_Flag_static_cold, Temperature_Flag);

   lbm::BC_Temperature_DiffusionDirichlet_static temperature_static_bc_hot(
      blocks, pdfFieldTemperatureGPUID, real_t(Thot));
   temperature_static_bc_hot.fillFromFlagField< FlagField_T >(blocks, flagFieldTemperatureID,
                                                         Density_Temperature_Flag_static_hot, Temperature_Flag);
   #endif


   // create the welford fluid and temperature sweep objects:

   WelfordSweepVelocity_T welfordVelocitySweep(meanVelFieldGPUID, sosVelFieldGPUID, velFieldFluidGPUID, 0_r);
   #ifdef run_with_temperature
   WelfordSweepTemperature_T welfordTemperatureSweep(meanTemperatureFieldGPUID, sosTemperatureFieldGPUID, temperatureFieldGPUID, 0_r);
   #endif

   ////////////////////////////////////
   // Initialize the PDFs and Fields //
   ///////////////////////////////////

   // Velocity field setup
   setVelocityFieldsAsmuth<VectorField_T>(
      blocks, velFieldFluidID, meanVelFieldID,
      target_friction_velocity, uint_c(forceParams.channelHalfWidth),
      5.5_r, 0.4_r, kinematicViscosityLB,
      forceParams.wallAxis, codegen::flow_axis );
   gpu::fieldCpy< gpu::GPUField< real_t >, VelocityField_fluid_T >(blocks, velFieldFluidGPUID, velFieldFluidID);

   // Map particles into the fluid domain
   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA_fluid(blocks, omega_f);
   PSMSweepCollection psmSweepCollectionFluid(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                              particleAndVolumeFractionSoA_fluid,
                                              particleSubBlockSize);



   #ifdef run_with_temperature
   ParticleAndVolumeFractionSoA_T< 1 > particleAndVolumeFractionSoA_temperature(blocks,omegaT_f);
   PSMSweepCollection psmSweepCollectionTemperature(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                                    particleAndVolumeFractionSoA_temperature,
                                                    particleSubBlockSize);
   SetParticleTemperaturesSweepp settemperatureparticles(blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(),
                                                         particleAndVolumeFractionSoA_temperature,
                                                         temperatureFieldGPUID,
                                                         particleTemperaturesFieldGPUID,
                                                         true);
   #endif

   // create the force and bulk velocity calculator:
   // use B-field as mask to exclude particle-occupied fractions from bulk velocity.
#ifdef run_with_temperature
   ForceCalculator< VectorField_T, BField_T > forceCalculator(blocks, velFieldFluidID,
                                                                    particleAndVolumeFractionSoA_temperature.BFieldID,
                                                                    forceParams);
#else
   ForceCalculator< VectorField_T, BField_T > forceCalculator(blocks, velFieldFluidID,
                                                                    particleAndVolumeFractionSoA_fluid.BFieldID,
                                                                    forceParams);
#endif

   // calculate the initial force for initialization:
   forceCalculator.setBulkVelocity(forceParams.targetBulkVelocity);
   const auto initialForce = forceCalculator.getCurrentDrivingForce();

   // Initialize PDFs

   pystencils::InitializeFluidDomain pdfSetterFluid(
      particleAndVolumeFractionSoA_fluid.BsFieldID, particleAndVolumeFractionSoA_fluid.BFieldID,
      particleAndVolumeFractionSoA_fluid.particleVelocitiesFieldID,
      pdfFieldFluidGPUID, velFieldFluidGPUID, initialForce,
      real_t(1));

   #ifdef run_with_temperature
   pystencils::InitializeTemperatureDomain pdfSetterTemperature(
      pdfFieldTemperatureGPUID, temperatureFieldGPUID, velFieldFluidGPUID);
   #endif


   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollectionFluid.particleMappingSweep(&(*blockIt));
      #ifdef run_with_temperature
      psmSweepCollectionTemperature.particleMappingSweep(&(*blockIt));
      #endif
   }

   #ifdef run_with_temperature
   # TODO
   /*walberla::initConcentrationFieldCoutte(blocks, temperatureFieldID,particleAndVolumeFractionSoA_fluid.BFieldID,
                                domainSize);*/
   #endif

   pystencils::FluidMacroGetter getterSweep_fluid(particleAndVolumeFractionSoA_fluid.BFieldID, densityFluidFieldGPUID,
                                                  pdfFieldFluidGPUID, velFieldFluidGPUID, initialForce);
   #ifdef run_with_temperature
   pystencils::TemperatureMacroGetter getterSweep_temperature(pdfFieldTemperatureGPUID, temperatureFieldGPUID);
   #endif


   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      psmSweepCollectionFluid.setParticleVelocitiesSweep(&(*blockIt));
      #ifdef run_with_temperature
      settemperatureparticles(&(*blockIt));
      gpu::fieldCpy< particleTemperaturesFieldGPU_T, particleTemperaturesField_T >(blocks, particleTemperaturesFieldGPUID,
                                                                                   particleTemperaturesFieldID);
      gpu::fieldCpy< gpu::GPUField< real_t >, DensityField_temperature_T >(blocks, temperatureFieldGPUID,
                                                                           temperatureFieldID);
      #endif
      if (!startFromCheckPointFile)
      {
         pdfSetterFluid(&(*blockIt));
      }
      #ifdef run_with_temperature
      if (!startFromCheckPointFile)
      {
         pdfSetterTemperature(&(*blockIt));
      }
      #endif
   }

   if (startFromCheckPointFile)
   {
      for (auto& block : *blocks)
      {
         getterSweep_fluid(&block);
         #ifdef run_with_temperature
         getterSweep_temperature(&block);
         #endif
      }
      gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, velFieldFluidID, velFieldFluidGPUID);
      gpu::fieldCpy< DensityField_fluid_T, gpu::GPUField< real_t > >(blocks, densityFluidFieldID,
                                                                     densityFluidFieldGPUID);
#ifdef run_with_temperature
      gpu::fieldCpy< DensityField_temperature_T, gpu::GPUField< real_t > >(blocks, temperatureFieldID,
                                                                          temperatureFieldGPUID);
#endif
   }

   ///////////////////////
   // ADD COMMUNICATION //
   //////////////////////

   // Setup of the fluid LBM communication for synchronizing the fluid pdf field between neighboring blocks
   gpu::communication::UniformGPUScheme< Stencil_Fluid_T > com_fluid(blocks, sendDirectlyFromGPU, false);
   com_fluid.addPackInfo(make_shared< PackInfoFluid_T >(pdfFieldFluidGPUID));
   auto communication_fluid = std::function< void() >([&]() { com_fluid.communicate(); });

   #ifdef run_with_temperature
   gpu::communication::UniformGPUScheme< Stencil_Temperature_T > com_temperature(blocks, sendDirectlyFromGPU, false);
   com_temperature.addPackInfo(make_shared< PackInfoTemperature_T >(pdfFieldTemperatureGPUID));
   auto communication_temperature = std::function< void() >([&]() { com_temperature.communicate(); });
   #endif

   // time loop objects for communication with and without hiding

   SweepTimeloop commTimeloop(blocks->getBlockStorage(), numTimeSteps);
   SweepTimeloop timeloop(blocks->getBlockStorage(), numTimeSteps);
   timeloop.setCurrentTimeStep(startTimeStep);
   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

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
         for (auto& block : *blocks)
         {
            getterSweep_fluid(&block);
         }
         gpu::fieldCpy< PdfField_fluid_T, gpu::GPUField< real_t > >(blocks, pdfFieldFluidID, pdfFieldFluidGPUID);
         gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, velFieldFluidID, velFieldFluidGPUID);
         gpu::fieldCpy< DensityField_fluid_T, gpu::GPUField< real_t > >(blocks, densityFluidFieldID,
                                                                        densityFluidFieldGPUID);
      });
#ifdef run_with_temperature
      auto vtkOutput_Temperature =
         vtk::createVTKOutput_BlockData(blocks, "vtk_files_temperature", vtkSpacingFluid, 0, false, vtkFolder);

      vtkOutput_Temperature->addBeforeFunction(communication_temperature);

      vtkOutput_Temperature->addBeforeFunction([&]() {
         for (auto& block : *blocks)
         {
            getterSweep_temperature(&block);
         }
         gpu::fieldCpy< PdfField_temperature_T, gpu::GPUField< real_t > >(blocks, pdfFieldTemperatureID,
                                                                          pdfFieldTemperatureGPUID);
      });
#endif

      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< PdfField_fluid_T > >(pdfFieldFluidID, "fluid pdf field"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< VelocityField_fluid_T > >(velFieldFluidID, "instantaneous Fluid Velocity"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< VelocityField_fluid_T > >(meanVelFieldID, "mean Fluid velocity"));
      vtkOutput_Fluid->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_fluid_T > >(densityFluidFieldID, "Fluid Density"));
      //vtkOutput_Fluid->addCellDataWriter(
      //   make_shared< field::VTKWriter< FlagField_T > >(flagFieldFluidID, "FluidFlagField"));


         auto flagOutput = vtk::createVTKOutput_BlockData(
            blocks, "flag_writer", 1, 1, false, "vtk_out", "simulation_step",
            false, true, true, false
         );
         auto flagWriter = std::make_shared<field::VTKWriter<FlagField_T>>(flagFieldFluidID, "flag_field_fluid");
         flagOutput->addCellDataWriter(flagWriter);
         flagOutput->write();


#ifdef run_with_temperature
      vtkOutput_Temperature->addCellDataWriter(
         make_shared< field::VTKWriter< PdfField_temperature_T > >(pdfFieldTemperatureID, "temperature pdf field"));
      vtkOutput_Temperature->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_temperature_T > >(temperatureFieldID, "instantaneous temperature field"));

      vtkOutput_Temperature->addCellDataWriter(
         make_shared< field::VTKWriter< FlagField_T > >(flagFieldTemperatureID, "TemperatureFlagField"));
      vtkOutput_Temperature->addCellDataWriter(
         make_shared< field::VTKWriter< DensityField_temperature_T > >(meanTemperatureFieldID, "mean temperature field"));
#endif

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput_Fluid), "VTK output Fluid");
#ifdef run_with_temperature
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(vtkOutput_Temperature), "VTK output Temperature");
#endif


      if(writeSlice){
         const AABB sliceAABB(real_t(0), real_c(domainSize[codegen::wall_axis]) * real_t(0.5) - real_t(1), real_t(0),
                              real_c(domainSize[codegen::flow_axis]),
                              real_c(domainSize[codegen::wall_axis]) * real_t(0.5) + real_t(1),
                              real_c(domainSize[codegen::remaining_axis]));
         const walberla::vtk::AABBCellFilter aabbSliceFilter(sliceAABB);
         field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldFluidID);
         fluidFilter.addFlag(Fluid_Flag);
         walberla::vtk::ChainedFilter combinedSliceFilter;
         combinedSliceFilter.addFilter(fluidFilter);
         combinedSliceFilter.addFilter(aabbSliceFilter);
         vtkOutput_Fluid->addCellInclusionFilter(combinedSliceFilter);
         timeloop.addFuncBeforeTimeStep(walberla::vtk::writeFiles(vtkOutput_Fluid), "VTK (fluid field data)");

#ifdef run_with_temperature
         field::FlagFieldCellFilter< FlagField_T > temperatureFilter(flagFieldTemperatureID);
         temperatureFilter.addFlag(Temperature_Flag);
         walberla::vtk::ChainedFilter combinedSliceFilterTemperature;
         combinedSliceFilterTemperature.addFilter(temperatureFilter);
         combinedSliceFilterTemperature.addFilter(aabbSliceFilter);
         vtkOutput_Temperature->addCellInclusionFilter(combinedSliceFilter);
         timeloop.addFuncBeforeTimeStep(walberla::vtk::writeFiles(vtkOutput_Temperature), "VTK (temperature field data)");
#endif
      }

   }
   if (vtkSpacingFluid != uint_t(0)) { vtk::writeDomainDecomposition(blocks, "domain_decomposition", vtkFolder); }

   ////////////////////////////////////////////////////////////////////////////////////////////////
   // add LBM communication, boundary handling and the LBM sweeps to the time loop              //
   //////////////////////////////////////////////////////////////////////////////////////////////

   pystencils::PSMFluidSweep psmFluidSweep(
      particleAndVolumeFractionSoA_fluid.BsFieldID, particleAndVolumeFractionSoA_fluid.BFieldID,
      particleAndVolumeFractionSoA_fluid.particleForcesFieldID, particleAndVolumeFractionSoA_fluid.particleVelocitiesFieldID,
      pdfFieldFluidGPUID, velFieldFluidGPUID, initialForce, omega_f);

   #ifdef run_with_temperature
   pystencils::PSMTemperatureSweep psmTemperatureSweep(
      pdfFieldTemperatureGPUID, temperatureFieldGPUID, velFieldFluidGPUID, omegaT_f, 1, 1, 1);
   #endif


   // setting new force:

   auto setNewForce = [&](const real_t newForce) {
      psmFluidSweep.setForcex(newForce);
   };
   auto psmFluidSweeplamda = [&psmFluidSweep](IBlock * block) {
      psmFluidSweep(block);
   };




   // Add performance logging
   lbm::PerformanceLogger< FlagField_T > performanceLogger(blocks, flagFieldFluidID, Fluid_Flag, performanceLogFrequency);
   if (performanceLogFrequency > 0)
   {
      timeloop.addFuncAfterTimeStep(performanceLogger, "Evaluate performance logging");
   }

   //////////////////////////////////////////
   // Creating the necessary class objects //
   //////////////////////////////////////////

   WcTimingPool timeloopTiming;
   const bool useOpenMP = true;

   real_t linkedCellWidth = linkedCellWidthRation * particleDiameter;
   mesa_pd::data::LinkedCells linkedCells(rpdDomain->getUnionOfLocalAABBs().getExtended(linkedCellWidth),
                                          linkedCellWidth);
   mesa_pd::kernel::InsertParticleIntoLinkedCells ipilc;

   ////////////////////////////////////////////////////////
   /// needed to monitor stationary state or convergence ///
   ////////////////////////////////////////////////////////
   WallStatistics wall_statistics(domainSize[codegen::wall_axis], 1, kinematicViscosityLB, outputFrequency);

   //////////////////////////////////////////////////////////////////////////////////////////////
   /// for obtaining plane velocity and temperature averages required to compute heat budgets ///
   //////////////////////////////////////////////////////////////////////////////////////////////

   MeanPlaneAverager meanPlaneAverager(uint_c(domainSize[codegen::wall_axis]));

   //////////////////////////////////////////
   // for computation of heatflux budgets //
   ////////////////////////////////////////

   #ifdef run_with_temperature
   HeatFluxBudgets heatFluxBudgets(uint_c(domainSize[codegen::wall_axis]),1, thermalDiffusivityFluid_LB,
                                   thermalDiffusivityParticle_LB, heatFluxAveragingtimeBlock, meanPlaneAverager);
   #endif

   //////////////////////////////////////////////////
   /// for computing particle and fluid statistics //
   /////////////////////////////////////////////////

   PlaneAveragedProfiles<VectorField_T> planeAveragedProfiles_velocity(blocks, velFieldFluidID,
                                                                        codegen::wall_axis, domainSize);
   #ifdef run_with_temperature
   PlaneAveragedProfiles<ScalarField_T> planeAveragedProfiles_temperature(blocks, temperatureFieldID,
                                                                          codegen::wall_axis, domainSize);
   #endif



   uint_t printchecker = 0;
   auto postProcessingLamdas = [&]() {


      if ( wall_statistics.getWallStatisticsConvergence() == true && timeloop.getCurrentTimeStep() >= uint_c(nTurnovers * turnOverPeriod) && timeloop.getCurrentTimeStep() % samplingInterval == 0  )
      {
         printchecker += 1;
         if (printchecker == 1)
         {
            WALBERLA_LOG_INFO_ON_ROOT("starting the phase statistics ");

         }


         gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, velFieldFluidID, velFieldFluidGPUID);

         // spatial and temporal averaging of the particle and fluid statistics
         #ifdef run_with_temperature

         gpu::fieldCpy< BField_T, gpu::GPUField< real_t > >(blocks, BFieldID, particleAndVolumeFractionSoA_temperature.BFieldID);
         planeAveragedProfiles_velocity.computeFluidParticleAveragedVectors(BFieldID);

         #else
         gpu::fieldCpy< BField_T, gpu::GPUField< real_t > >(blocks, BFieldID, particleAndVolumeFractionSoA_fluid.BFieldID);
         planeAveragedProfiles_velocity.computeFluidParticleAveragedVectors(BFieldID);
         #endif

      }

      if (timeloop.getCurrentTimeStep() == numTimeSteps - uint_c(turnOverPeriod))
      {
         WALBERLA_LOG_INFO_ON_ROOT("at time Step "<< timeloop.getCurrentTimeStep() << " writing the phase statistics to file: output/phase_statistics.txt  ")
         // computation of fluid and particle avg, rms, reynolds stresses quantities
         gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, velFieldFluidID, velFieldFluidGPUID);
         planeAveragedProfiles_velocity.computeFluidParticleRMS();

         WALBERLA_ROOT_SECTION()
         {
            std::ofstream velocityOS;
            velocityOS << std::fixed << std::setprecision(6);
            velocityOS.open("output/phase_statistics.txt", std::ios::out);
            auto printRow = [&](auto&&... args) {
               ((velocityOS << std::setw(12) << args), ...);
               velocityOS << "\n";
            };

            // plotting everything in normalized wall units
            printRow("y", "y+", "Ux_f", "Uy_f", "Uz_f", "UU_f", "UV_f", "UW_f", "VU_f", "VV_f", "VW_f", "WU_f", "WV_f",
                     "WW_f", "Ux_p", "Uy_p", "Uz_p", "UU_p", "UV_p", "UW_p", "VU_p", "VV_p", "VW_p", "WU_p", "WV_p",
                     "WW_p");
            printRow(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

            for (uint_t idx = 0; idx < domainSize[codegen::wall_axis]; ++idx)
            {
               velocityOS << std::setw(12) << idx;

               // fluid averaged velocities
               for (uint_t i = 0; i < VectorField_T::F_SIZE; ++i)
               {
                  velocityOS << std::setw(12) <<
                  planeAveragedProfiles_velocity.getFluidAVGProfile()[idx * VectorField_T::F_SIZE + i];
               }

               // fluid rms profiles
               for (uint_t i = 0; i < VectorField_T::F_SIZE; ++i)
               {
                  for (uint_t j = 0; j < VectorField_T::F_SIZE; ++j)
                  {
                     velocityOS << std::setw(12)
                                << planeAveragedProfiles_velocity
                                      .getFluidRMSProfile()[idx * VectorField_T::F_SIZE * VectorField_T::F_SIZE +
                                                            i * VectorField_T::F_SIZE + j];
                  }
               }

               // particle averaged velocities
               for (uint_t i = 0; i < VectorField_T::F_SIZE; ++i)
               {
                  velocityOS << std::setw(12) <<
                  planeAveragedProfiles_velocity.getParticleAVGProfile()[idx * VectorField_T::F_SIZE + i];
               }

               // particle rms profiles
               for (uint_t i = 0; i < VectorField_T::F_SIZE; ++i)
               {
                  for (uint_t j = 0; j < VectorField_T::F_SIZE; ++j)
                  {
                     velocityOS << std::setw(12)
                                << planeAveragedProfiles_velocity
                                      .getParticleRMSProfile()[idx * VectorField_T::F_SIZE * VectorField_T::F_SIZE +
                                                            i * VectorField_T::F_SIZE + j];
                  }
               }

               velocityOS << "\n";
            }
            velocityOS.flush();
            velocityOS.close();
            WALBERLA_ABORT("simulation completed and the averaged results have been written to the file:  phase_statistics.txt");
         }
      }
   };

   #ifdef run_with_temperature
   auto postProcessingLamdaswithTemperature = [&]() {


      if (timeloop.getCurrentTimeStep() >= nTurnovers*turnOverPeriod && wall_statistics.getWallStatisticsConvergence() ==true )
      {
         if (meanPlaneAverager.getTimeCounter() < 2 * planeAveragingtimeBlock)
         {
            if (meanPlaneAverager.getTimeCounter() % 1000 == 0) // just for debugging
            {
               WALBERLA_LOG_INFO_ON_ROOT("still in  the pre part  " << meanPlaneAverager.getTimeCounter())
            }
            meanPlaneAverager(blocks, velFieldFluidID, temperatureFieldID,
                              particleAndVolumeFractionSoA_temperature.BFieldID);
         }
         if (meanPlaneAverager.getTimeCounter() == 2 * planeAveragingtimeBlock)
         {
            if (meanPlaneAverager.getTimeCounter() % 10000 == 0)
            {
               WALBERLA_LOG_INFO_ON_ROOT("entered the averagin part")
            }
            heatFluxBudgets(blocks, velFieldFluidID, temperatureFieldID,
                            particleAndVolumeFractionSoA_temperature.BFieldID);

            planeAveragedProfiles_velocity.computeFluidParticleAveragedVectors(
               particleAndVolumeFractionSoA_temperature.BFieldID);
            planeAveragedProfiles_temperature.computeFluidParticleAveragedVectors(
               particleAndVolumeFractionSoA_temperature.BFieldID);

            if (heatFluxBudgets.getTimeSampleCount() == real_c(heatFluxAveragingtimeBlock))
            {
               // averaging is done so write out the different quantities to text files here:

               planeAveragedProfiles_velocity.computeFluidParticleRMS(); // compute the rms quantities
               planeAveragedProfiles_temperature.computeFluidParticleRMS();
               WALBERLA_ROOT_SECTION()
               {
                  std::ofstream velocityOS;
                  velocityOS.open("statistics.txt", std::ios::out);
                  velocityOS << "height\t Uf_x\t Uf_y\t Uf_z\t Up_x\t Up_y\t Up_z\t Urms_fx\t Urms_fy\t Urms_fz\t "
                                "Urms_px\t Urms_py\t Urms_pz\t Tf\t Tp\t Trms_f\t Trms_p\t  \n";
                  for (uint_t idx = 0; idx < domainSize[codegen::wall_axis]; ++idx)
                  {
                     velocityOS << idx << "\t";
                     for (uint_t i = 0; i < VectorField_T::F_SIZE; ++i)
                     {
                        velocityOS
                           << planeAveragedProfiles_velocity.getFluidAVGProfile()[idx * VectorField_T::F_SIZE + i]
                           << "\t";
                        velocityOS
                           << planeAveragedProfiles_velocity.getParticleAVGProfile()[idx * VectorField_T::F_SIZE + i]
                           << "\t";
                        velocityOS
                           << planeAveragedProfiles_velocity.getParticleRMSProfile()[idx * VectorField_T::F_SIZE + i]
                           << "\t";
                        velocityOS
                           << planeAveragedProfiles_velocity.getParticleRMSProfile()[idx * VectorField_T::F_SIZE + i]
                           << "\t";
                     }

                     velocityOS << planeAveragedProfiles_temperature.getFluidAVGProfile()[idx] << "\t";
                     velocityOS << planeAveragedProfiles_temperature.getParticleAVGProfile()[idx] << "\t";
                     velocityOS << planeAveragedProfiles_temperature.getParticleRMSProfile()[idx] << "\t";
                     velocityOS << planeAveragedProfiles_temperature.getParticleRMSProfile()[idx] << "\t";

                     velocityOS << "\n";
                  }
                  WALBERLA_ABORT("simulation completed and the averaged results have been written to the file:  statistics.txt");
               }
            }
         }
      }
   };
   #endif

   auto welfordPhasesSweepLambda =
      std::function< void(IBlock*) >([&](IBlock* block) {
            welfordVelocitySweep(block);
            #ifdef run_with_temperature
            welfordTemperatureSweep(block);
            #endif

         if (timeloop.getCurrentTimeStep() == numTimeSteps - uint_c(turnOverPeriod))
            {
            WALBERLA_LOG_INFO_ON_ROOT("at time Step "<< timeloop.getCurrentTimeStep() << " writing the welford statistics to file: output/welford_statistics.txt");
               // computation of welford statistics
               gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, meanVelFieldID,
                                                                                meanVelFieldGPUID);
               gpu::fieldCpy< TensorField_T, gpu::GPUField< real_t > >(blocks, sosVelFieldID, sosVelFieldGPUID);

               reduceWelfordFields<VectorField_T, TensorField_T> reduceWelford_velocity(
                  blocks, meanVelFieldID, sosVelFieldID, codegen::wall_axis,
                  uint_c(domainSize[codegen::wall_axis]), welfordVelocitySweep);

               reduceWelford_velocity();
               auto welford_mean_velocity = reduceWelford_velocity.getPlaneMeans();
               auto welford_sos_velocity = reduceWelford_velocity.getPlaneSoSMeans();

               // here a function to reduce welford fields to vectors is needed
               WALBERLA_ROOT_SECTION()
               {


                  std::ofstream velocityOS;
                  velocityOS << std::fixed << std::setprecision(6);
                  velocityOS.open("welford_statistics.txt", std::ios::out);
                  auto printRow = [&](auto&&... args) {
                     ((velocityOS << std::setw(12) << args), ...);
                     velocityOS << "\n";
                  };

                  printRow("y", "Ux_f", "Uy_f", "Uz_f", "UU", "UV", "UW", "VU", "VV", "VW", "WU", "WV",
                           "WW");

                  for (uint_t idx = 0; idx < domainSize[codegen::wall_axis]; ++idx)
                  {
                     velocityOS << std::setw(12) << idx;

                     // fluid averaged velocities
                     for (uint_t i = 0; i < VectorField_T::F_SIZE; ++i)
                     {
                        velocityOS
                           << std::setw(12)
                           << welford_mean_velocity[idx * VectorField_T::F_SIZE + i];
                     }

                     // fluid rms profiles
                     for (uint_t i = 0; i < VectorField_T::F_SIZE; ++i)
                     {
                        for (uint_t j = 0; j < VectorField_T::F_SIZE; ++j)
                        {
                           velocityOS << std::setw(12)
                                      << welford_sos_velocity[idx * VectorField_T::F_SIZE * VectorField_T::F_SIZE +
                                                                  i * VectorField_T::F_SIZE + j];
                        }
                     }

                     velocityOS << "\n";
                  }
                  velocityOS.flush();
                  velocityOS.close();
                 // WALBERLA_ABORT(
                   //  "simulation completed and the averaged results have been written to the file:  welford_statistics.txt");
               }
            }



      });

   auto wallstatisticsLamda = [&]() {
      if (timeloop.getCurrentTimeStep() % samplingInterval == 0)
      {
#ifdef run_with_temperature
         gpu::fieldCpy< DensityField_temperature_T, gpu::GPUField< real_t > >(blocks, meanTemperatureFieldID,
                                                                              meanTemperatureFieldGPUID);
         wall_statistics(blocks, meanTemperatureFieldID, meanVelFieldID, timeloop.getCurrentTimeStep(), Tcold, Thot,
                         convergenceTolerance);
#else
         gpu::fieldCpy< VelocityField_fluid_T, gpu::GPUField< real_t > >(blocks, meanVelFieldID, meanVelFieldGPUID);
         wall_statistics(blocks, meanVelFieldID, timeloop.getCurrentTimeStep(), convergenceTolerance);
      }
#endif
      };
   ///////////////////////////////////
   // add everything to the timeloop//
   ///////////////////////////////////

   timeloop.add() << BeforeFunction(communication_fluid, "LBM fluid Communication")
                  << Sweep(deviceSyncWrapper(noSlip_fluid_bc.getSweep()), "Boundary Handling (No slip fluid)");

   timeloop.add() << Sweep(deviceSyncWrapper(freeSlip_fluid_bc.getSweep()), "Boundary Handling (Free slip fluid)");

   // add the temperature to the time loop

#ifdef run_with_temperature
   timeloop.add() << BeforeFunction(communication_temperature, "LBM temperature Communication")
                  << Sweep(deviceSyncWrapper(neumann_temperature_bc.getSweep()),
                           "Boundary Handling (Temperature Neumann)");

   timeloop.add() << Sweep(deviceSyncWrapper(temperature_static_bc_hot.getSweep()),
                           "Boundary Handling (Temperature static bc hot)");

   timeloop.add() << Sweep(deviceSyncWrapper(temperature_static_bc_cold.getSweep()),
                           "Boundary Handling (Temperature static bc cold)");
#endif

   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionFluid.particleMappingSweep),
                           "Particle mapping Fluid"); // uses weighting for hydrodynamics specified in Cmakelists file

   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionFluid.setParticleVelocitiesSweep),
                           "Set particle velocities from fluid sweepcollection");
#ifdef run_with_temperature
   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionTemperature.particleMappingSweep),
                           "Particle mapping Thermal"); // always uses a weighting of 1
#endif

   // compute the force before the psm fluid sweep.

   timeloop.add() //<< BeforeFunction([&]() { forceCalculator.calculateBulkVelocity(); }, "bulk velocity calculation")
                  << BeforeFunction(
                        [&]() {
                           // const auto newForce = forceCalculator.calculateDrivingForce();
                           setNewForce(initialForce);
                        },
                        "new force setter")
                  << Sweep([](IBlock*) {}, "new force setter");

   timeloop.add() << Sweep(psmFluidSweeplamda, "PSM Fluid sweep");

#ifdef run_with_temperature
   timeloop.add() << Sweep(deviceSyncWrapper(psmTemperatureSweep), "PSM Temperature sweep");
#endif

   // after both the sweeps, reduce the particle forces.
   timeloop.add() << Sweep(deviceSyncWrapper(psmSweepCollectionFluid.reduceParticleForcesSweep),
                           "Reduce particle forces");


   bool resetCounters = false;
   timeloop.addFuncAfterTimeStep(postProcessingLamdas, "custom HeatFlux, velocity statistics and other post processing");
   timeloop.add() << BeforeFunction(
                        [&]() {
                           if (wall_statistics.getWallStatisticsConvergence() == true && resetCounters == false)
                           {
                              WALBERLA_LOG_INFO_ON_ROOT("entering loop again should print only once")
                              resetCounters = true;
                              welfordVelocitySweep.setCounter(0);
#ifdef run_with_temperature
                              welfordTemperatureSweep.setCounter(0);
#endif

                              for (auto& block : *blocks)
                              {
#ifdef run_with_temperature
                                 auto* sosTemperatureField =
                                    block.template getData< ScalarField_T >(sosTemperatureFieldID);

                                 sosTemperatureField->setWithGhostLayer(0.0);
#endif
                                 auto* sosVelocityField = block.template getData< TensorField_T >(sosVelFieldID);
                                 sosVelocityField->setWithGhostLayer(0.0);
                              }
                           }
                           welfordVelocitySweep.setCounter(welfordVelocitySweep.getCounter() + real_c(1.0));
#ifdef run_with_temperature
                           welfordTemperatureSweep.setCounter(welfordTemperatureSweep.getCounter() + real_c(1.0));
#endif
                           //WALBERLA_LOG_INFO_ON_ROOT("current counter welfor is   " << welfordVelocitySweep.getCounter());
                        },
                        "welford velocity sweep")
                  << Sweep(welfordPhasesSweepLambda, "welford velocity sweep");


   timeloop.addFuncAfterTimeStep(wallstatisticsLamda, "wall statistics computations");

   for (uint_t timeStep = startTimeStep; timeStep < numTimeSteps; ++timeStep)
   {
      //WALBERLA_LOG_INFO_ON_ROOT("reached here 0001");
      if (checkPointingFrequency > uint_t(0) && (timeStep % checkPointingFrequency == 0) && timeStep != startTimeStep)
      {

         WALBERLA_LOG_INFO_ON_ROOT("writing checkpoint at timestep " << timeStep << " (pre-step)");
         checkpointTimer.start();
         gpu::fieldCpy<PdfField_fluid_T,gpu::GPUField< real_t >>(blocks,pdfFieldFluidID,pdfFieldFluidGPUID);
         blocks->saveBlockData(checkpointingFileName + "_lbm_tmp.txt", pdfFieldFluidID,false);
         checkpointTimer.end();
         WALBERLA_LOG_INFO_ON_ROOT("time to write checkpoint fluid is  " << checkpointTimer.last());
#ifdef run_with_temperature
         gpu::fieldCpy< PdfField_temperature_T, gpu::GPUField< real_t > >(blocks, pdfFieldTemperatureID,
                                                                          pdfFieldTemperatureGPUID);
         blocks->saveBlockData(checkpointingFileName + "_lbm_tmp_temperature.txt", pdfFieldTemperatureID, false);
#endif
         if (useParticles)
         {
            blocks->saveBlockData(checkpointingFileName + "_mesa_tmp.txt", particleStorageID);
         }

         WALBERLA_ROOT_SECTION()
         {
            std::ofstream checkpointConfigOS(checkpointingFileName + "_config_tmp.txt",
                                             std::ios::out | std::ios::trunc);
            WALBERLA_CHECK(checkpointConfigOS.is_open(), "Could not open checkpoint config tmp file "
                                                            << checkpointingFileName + "_config_tmp.txt"
                                                            << " for writing");
            checkpointConfigOS << timeStep << "\n";
            checkpointConfigOS.close();
         }
         WALBERLA_MPI_BARRIER();

         WALBERLA_ROOT_SECTION()
         {
            if (writeContinuousCheckPoints)
            {
               renameFile(checkpointingFileName + "_lbm_tmp.txt", checkpointingFileName + "_lbm_" + std::to_string(timeStep) + ".txt");
#ifdef run_with_temperature
               renameFile(checkpointingFileName + "_lbm_tmp_temperature.txt",
                          checkpointingFileName + "_lbm_temperature_" + std::to_string(timeStep) + ".txt");
#endif
               if (useParticles)
               {
                  renameFile(checkpointingFileName + "_mesa_tmp.txt",
                             checkpointingFileName + "_mesa_" + std::to_string(timeStep) + ".txt");
               }
               renameFile(checkpointingFileName + "_config_tmp.txt", checkpointingFileName + "_config_" + std::to_string(timeStep) + ".txt");

               if (timeStep >= 2 * checkPointingFrequency)
               {
                  uint_t oldStep = timeStep - 2 * checkPointingFrequency;
                  std::string oldBase_lbm = checkpointingFileName + "_lbm_" + std::to_string(oldStep);
                  std::string oldBase_config = checkpointingFileName + "_config_" + std::to_string(oldStep);
                  std::string oldBase_particles = checkpointingFileName + "_mesa_" + std::to_string(oldStep);
#ifdef run_with_temperature
                  std::string oldBase_temperature = checkpointingFileName + "_lbm_temperature_" + std::to_string(oldStep);
#endif

                  if (std::filesystem::exists(oldBase_lbm + ".txt")) { std::filesystem::remove(oldBase_lbm + ".txt"); }
#ifdef run_with_temperature
                  if (std::filesystem::exists(oldBase_temperature + ".txt"))
                  {
                     std::filesystem::remove(oldBase_temperature + ".txt");
                  }
#endif
                  if (std::filesystem::exists(oldBase_config + ".txt"))
                  {
                     std::filesystem::remove(oldBase_config + ".txt");
                  }
                  if (useParticles && std::filesystem::exists(oldBase_particles + ".txt"))
                  {
                     std::filesystem::remove(oldBase_particles + ".txt");
                  }
               }
            }
            else
            {
               renameFile(checkpointingFileName + "_lbm_tmp.txt", checkpointingFileName + "_lbm.txt");
#ifdef run_with_temperature
               renameFile(checkpointingFileName + "_lbm_tmp_temperature.txt",
                          checkpointingFileName + "_lbm_temperature.txt");
#endif
               if (useParticles)
               {
                  renameFile(checkpointingFileName + "_mesa_tmp.txt", checkpointingFileName + "_mesa.txt");
               }
               renameFile(checkpointingFileName + "_config_tmp.txt", checkpointingFileName + "_config.txt");
            }
         }
      }

      // perform a single simulation step -> this contains LBM and setting of the hydrodynamic interactions
      timeloop.singleStep(timeloopTiming);

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

         #ifdef run_with_temperature
         gpu::fieldCpy< DensityField_temperature_T,gpu::GPUField< real_t >  >(blocks,temperatureFieldID, temperatureFieldGPUID);
         gpu::fieldCpy<VelocityField_fluid_T ,gpu::GPUField< real_t >  >(blocks,velFieldFluidID, velFieldFluidGPUID);
         gpu::fieldCpy< DensityField_fluid_T, gpu::GPUField< real_t > >(blocks, densityFluidFieldID,
                                                                    densityFluidFieldGPUID);
         auto fluidInfo =
            evaluateFluidInfo(blocks, densityFluidFieldID, velFieldFluidID,temperatureFieldID);
         #else
         gpu::fieldCpy<VelocityField_fluid_T ,gpu::GPUField< real_t >  >(blocks,velFieldFluidID, velFieldFluidGPUID);
         gpu::fieldCpy< DensityField_fluid_T, gpu::GPUField< real_t > >(blocks, densityFluidFieldID,
                                                                    densityFluidFieldGPUID);
         auto fluidInfo =
           evaluateFluidInfo(blocks, densityFluidFieldID, velFieldFluidID);

         #endif
         WALBERLA_LOG_INFO_ON_ROOT(fluidInfo);

         if (particleBarriers) WALBERLA_MPI_BARRIER();
         timeloopTiming["Evaluate infos"].end();
      }
   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}
} // namespace MaterialTransport

int main(int argc, char** argv) { MaterialTransport::main(argc, argv); }
