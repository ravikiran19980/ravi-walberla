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
#include "core/math/all.h"
#include "core/math/extern/exprtk.h"
#include "core/timing/RemainingTimeLogger.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "geometry/InitBoundaryHandling.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "lbm/PerformanceLogger.h"
#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypesCodegen.h"

#include "mesa_pd/data/LinkedCells.h"

#include "mesa_pd/domain/BlockForestDataHandling.h"

#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "sqlite/SQLite.h"

#include "vtk/all.h"

#include <fstream>
#include <filesystem>
#include <iomanip>

#include "FluidMacroGetter.h"
#include "GeneralInfoHeader.h"

#include <memory>
#include <cmath>
#include <string>
#include <blockforest/all.h>
#include <core/all.h>
#include <domain_decomposition/all.h>
#include <field/all.h>
#include <geometry/all.h>
#include <timeloop/all.h>
#include <lbm/all.h>


#include "math.h"


namespace MaterialTransport
{

///////////
// USING //
///////////

using namespace walberla;
using namespace lbm_mesapd_coupling::psm::gpu;
using namespace pystencils;
typedef PackInfoFluid PackInfoFluid_T;
typedef PackInfoTemperature PackInfoTemperature_T;

using flag_t      = uint8_t;
using FlagField_T = FlagField< flag_t >;

// Field Types
using ScalarField_T = GhostLayerField< real_t, 1 >;
using VectorField_T = GhostLayerField< real_t, Stencil_Fluid_T::D >;
using TensorField_T = GhostLayerField< real_t, Stencil_Fluid_T::D*Stencil_Fluid_T::D >;
using WelfordSweepVelocity_T = WelfordVelocity;
using WelfordSweepTemperature_T = WelfordTemperature;


///////////
// FLAGS //
///////////


struct FluidInfo
{
   uint_t numFluidCells   = 0;
   real_t averageVelocity = 0_r;
   real_t maximumVelocity = 0_r;
   real_t averageDensity  = 0_r;
   real_t maximumDensity  = 0_r;
   real_t maxTemperature  = 0_r;
   real_t minTemperature  = 0_r;

   void allReduce()
   {
      walberla::mpi::allReduceInplace(numFluidCells, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(averageVelocity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(maximumVelocity, walberla::mpi::MAX);
      ;
      walberla::mpi::allReduceInplace(averageDensity, walberla::mpi::SUM);
      walberla::mpi::allReduceInplace(maximumDensity, walberla::mpi::MAX);
      walberla::mpi::allReduceInplace(maxTemperature, walberla::mpi::MAX);
      walberla::mpi::allReduceInplace(minTemperature, walberla::mpi::MIN);

      averageVelocity /= real_c(numFluidCells);
      averageDensity /= real_c(numFluidCells);
   }
};

std::ostream& operator<<(std::ostream& os, FluidInfo const& m)
{
   return os << "Fluid Info: numFluidCells = " << m.numFluidCells << ", uAvg = " << m.averageVelocity
             << ", uMax = " << m.maximumVelocity << ", densityAvg = " << m.averageDensity
             << ", densityMax = " << m.maximumDensity << ", TMax = " << m.maxTemperature << ", TMin = " << m.minTemperature;
}

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
         real_t density = densityField->get(x, y, z); real_t velMagnitude = std::abs(velocityField->get(x, y, z, 0));//velocity.length();
         real_t temperature = temperatureField->get(x,y,z);
         info.averageVelocity += velMagnitude; info.maximumVelocity = std::max(info.maximumVelocity, velMagnitude);
         info.averageDensity += density; info.maximumDensity        = std::max(info.maximumDensity, density);
         info.maxTemperature = std::max(info.maxTemperature, temperature);
         info.minTemperature = std::min(info.minTemperature, temperature);)

   }
   info.allReduce();
   return info;
}
struct ForceCalculatorParameters
{
   Vector3<uint_t> domainSize;
   uint_t wallAxis;

   real_t channelHalfWidth;
   real_t targetBulkVelocity;
   real_t targetFrictionVelocity;
};

template< typename VelocityField_T >
   class ForceCalculator {

    public:
      ForceCalculator(const std::weak_ptr<StructuredBlockStorage> & blocks, const BlockDataID meanVelocityId,
                      const ForceCalculatorParameters & parameter)
         : blocks_(blocks), meanVelocityId_(meanVelocityId), channelHalfWidth_(real_c(parameter.channelHalfWidth)),
           targetBulkVelocity_(parameter.targetBulkVelocity), targetFrictionVelocity_(parameter.targetFrictionVelocity)
      {
         const auto & domainSize = parameter.domainSize;
         WALBERLA_LOG_INFO_ON_ROOT("target friction velocity is "  << targetFrictionVelocity_);
         Cell maxCell;
         maxCell[parameter.wallAxis] = int_c(parameter.channelHalfWidth) - 1;
         maxCell[flowDirection_] = int_c(domainSize[flowDirection_]) - 1;
         const auto remainingIdx = 3 - parameter.wallAxis - flowDirection_;
         maxCell[remainingIdx] = int_c(domainSize[remainingIdx]) - 1;
         ci_ = CellInterval(Cell{}, maxCell);

         numCells_ = real_c(parameter.channelHalfWidth * domainSize[flowDirection_] * domainSize[remainingIdx]);
      }

      real_t bulkVelocity() const { return bulkVelocity_; }
      void setBulkVelocity(const real_t bulkVelocity) { bulkVelocity_ = bulkVelocity; }

      void calculateBulkVelocity() {

         // reset bulk velocity
         bulkVelocity_ = 0_r;

         auto blocks = blocks_.lock();
         WALBERLA_CHECK_NOT_NULLPTR(blocks)

         for( auto block = blocks->begin(); block != blocks->end(); ++block) {

            auto * meanVelocityField = block->template getData<VelocityField_T>(meanVelocityId_);
            WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)

            auto fieldSize = meanVelocityField->xyzSize();
            CellInterval localCi;
            blocks->transformGlobalToBlockLocalCellInterval(localCi, *block, ci_);
            fieldSize.intersect(localCi);

            auto * slicedField = meanVelocityField->getSlicedField(fieldSize);
            WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)

            for(auto fieldIt = slicedField->beginXYZ(); fieldIt != slicedField->end(); ++fieldIt) {
               const auto localMean = fieldIt[flowDirection_];
               bulkVelocity_ += localMean;
            }

         }

         mpi::allReduceInplace< real_t >(bulkVelocity_, mpi::SUM);
         bulkVelocity_ /= numCells_;

      }

      real_t calculateDrivingForce() const {

         // forcing term as in Malaspinas (2014) "Wall model for large-eddy simulation based on the lattice Boltzmann method"
         const auto force = targetFrictionVelocity_ * targetFrictionVelocity_ / channelHalfWidth_
                            + (targetBulkVelocity_ - bulkVelocity_) * targetBulkVelocity_ / channelHalfWidth_;

         return force;
      }

    private:
      const std::weak_ptr<StructuredBlockStorage> blocks_{};
      const BlockDataID meanVelocityId_{};

      const uint_t flowDirection_{};
      const real_t channelHalfWidth_{};
      const real_t targetBulkVelocity_{};
      const real_t targetFrictionVelocity_{};

      CellInterval ci_{};

      real_t numCells_{};
      real_t bulkVelocity_{};
   };

namespace boundaries {
void createBoundaryConfig(ForceCalculatorParameters &parameters, Config::Block & boundaryBlock) {

   auto & bottomWall = boundaryBlock.createBlock("Border");
   bottomWall.addParameter("direction", stencil::dirToString[stencil::directionFromAxis(parameters.wallAxis, true)]);
   bottomWall.addParameter("walldistance", "-1");

   bottomWall.addParameter("flag", "NoSlip");


   auto & topWall = boundaryBlock.createBlock("Border");
   topWall.addParameter("direction", stencil::dirToString[stencil::directionFromAxis(parameters.wallAxis, false)]);
   topWall.addParameter("walldistance", "-1");
   topWall.addParameter("flag", "NoSlip");
}
}

template<typename VelocityField_T>
   void setVelocityFieldsAsmuth( const std::weak_ptr<StructuredBlockStorage>& forest,
                                 const BlockDataID & velocityFieldId, const BlockDataID & meanVelocityFieldId,
                                 const real_t frictionVelocity, const uint_t channel_half_width,
                                 const real_t B, const real_t kappa, const real_t viscosity,
                                 const uint_t wallAxis, const uint_t flowAxis ) {

      auto blocks = forest.lock();
      WALBERLA_CHECK_NOT_NULLPTR(blocks)

      const auto domainSize = blocks->getDomain().max();
      const auto delta = real_c(channel_half_width);
      const auto remAxis = 3 - wallAxis - flowAxis;

      for( auto block = blocks->begin(); block != blocks->end(); ++block ) {

         auto * velocityField = block->template getData<VelocityField_T>(velocityFieldId);
         WALBERLA_CHECK_NOT_NULLPTR(velocityField)

         auto * meanVelocityField = block->template getData<VelocityField_T>(meanVelocityFieldId);
         WALBERLA_CHECK_NOT_NULLPTR(meanVelocityField)

         const auto ci = velocityField->xyzSizeWithGhostLayer();
         for(auto cellIt = ci.begin(); cellIt != ci.end(); ++cellIt) {

            Cell globalCell(*cellIt);
            blocks->transformBlockLocalToGlobalCell(globalCell, *block);
            Vector3<real_t> cellCenter;
            blocks->getCellCenter(cellCenter, globalCell);

            const auto y = cellCenter[wallAxis];
            const auto rel_x = cellCenter[flowAxis] / domainSize[flowAxis];
            const auto rel_z = cellCenter[remAxis] / domainSize[remAxis];

            const real_t pos = std::max(delta - std::abs(y - delta - 1_r), 0.05_r);
            const auto rel_y = pos / delta;

            auto initialVel = frictionVelocity * (std::log(frictionVelocity * pos / viscosity) / kappa + B);

            Vector3<real_t> vel;
            vel[flowAxis] = initialVel;

            vel[remAxis] = 2_r * frictionVelocity / kappa * std::sin(math::pi * 16_r * rel_x) *
                           std::sin(math::pi * 8_r * rel_y) / (std::pow(rel_y, 2_r) + 1_r);

            vel[wallAxis] = 8_r * frictionVelocity / kappa *
                            (std::sin(math::pi * 8_r * rel_z) * std::sin(math::pi * 8_r * rel_y) +
                             std::sin(math::pi * 8_r * rel_x)) / (std::pow(0.5_r * delta - pos, 2_r) + 1_r);

            for(uint_t d = 0; d < 3; ++d) {
               velocityField->get(*cellIt, d) = vel[d];
               meanVelocityField->get(*cellIt, d) = vel[d];
            }

         }
      }

   } // function setVelocityFieldsHenrik

//////////
// MAIN //
//////////

int main(int argc, char** argv)
{
   Environment env(argc, argv);
   auto cfgFile = env.config();
   if (!cfgFile) { WALBERLA_ABORT("Usage: " << argv[0] << " path-to-configuration-file \n"); }

   WALBERLA_ROOT_SECTION()
   {
      const std::filesystem::path outputPath("output");
      if (!std::filesystem::exists(outputPath))
      {
         std::filesystem::create_directories(outputPath);
      }
   }


   WALBERLA_LOG_INFO_ON_ROOT("one direction is   " << stencil::dirToString[stencil::directionFromAxis(1, true)]);
   WALBERLA_LOG_INFO_ON_ROOT("one direction is   " << stencil::dirToString[stencil::directionFromAxis(1, false)]);
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
   if ((periodicInX && numXBlocks == 1) || (periodicInY && numYBlocks == 1) || (periodicInZ && numZBlocks == 1))
   {
      WALBERLA_ABORT("The number of blocks must be greater than 1 in periodic dimensions.")
   }

   const bool useLubricationForces        = numericalSetup.getParameter< bool >("useLubricationForces");
   const uint_t numberOfParticleSubCycles = numericalSetup.getParameter< uint_t >("numberOfParticleSubCycles");
   const bool useIntegrators              = numericalSetup.getParameter< bool >("useIntegrators");
   const Vector3< uint_t > particleSubBlockSize =
      numericalSetup.getParameter< Vector3< uint_t > >("particleSubBlockSize");
   const real_t linkedCellWidthRation = numericalSetup.getParameter< real_t >("linkedCellWidthRation");
   const bool particleBarriers        = numericalSetup.getParameter< bool >("particleBarriers");
   const Vector3< real_t > generationDomainFraction =
      numericalSetup.getParameter< Vector3< real_t > >("generationDomainFraction");

   const real_t volfraction = numericalSetup.getParameter< real_t >("volfraction");

   const bool writeSlice          = numericalSetup.getParameter< bool >("writeSlice");
   const bool sendDirectlyFromGPU = numericalSetup.getParameter< bool >("sendDirectlyFromGPU");

   Config::BlockHandle turbulenceSetup   = cfgFile->getBlock("TurbulenceSetup");
   const real_t target_bulk_Reynolds     = turbulenceSetup.getParameter< real_t >("target_bulk_Reynolds");
   const real_t target_friction_Reynolds = turbulenceSetup.getParameter< real_t >("target_friction_Reynolds");
   const real_t target_bulk_velocity     = turbulenceSetup.getParameter< real_t >("target_bulk_velocity");
   const real_t center_line_velocity     = turbulenceSetup.getParameter< real_t >("center_line_velocity");
   const uint_t nTurnovers               = turbulenceSetup.getParameter< uint_t >("nTurnovers");


   Config::BlockHandle TemperatureSetup = cfgFile->getBlock("TemperatureSetup");
   const real_t Thot                    = TemperatureSetup.getParameter< real_t >("Thot");
   const real_t Tcold                   = TemperatureSetup.getParameter< real_t >("Tcold");
   const real_t Tref                    = TemperatureSetup.getParameter< real_t >("Tref");
   const real_t Tparticle               = TemperatureSetup.getParameter< real_t >("Tparticle");
   const real_t Pr                      = TemperatureSetup.getParameter< real_t >("PrandtlNumber");
   const real_t diffusivityRatio        = TemperatureSetup.getParameter< real_t >("diffusivityRatio");
   const real_t Gr                      = TemperatureSetup.getParameter< real_t >("Gr");

   Config::BlockHandle outputSetup      = cfgFile->getBlock("Output");
   const uint_t infoSpacing             = outputSetup.getParameter< real_t >("infoSpacing");
   const real_t vtkSpacingParticles     = outputSetup.getParameter< real_t >("vtkSpacingParticles");
   const real_t vtkSpacingFluid         = outputSetup.getParameter< real_t >("vtkSpacingFluid");
   const std::string vtkFolder          = outputSetup.getParameter< std::string >("vtkFolder");
   const uint_t performanceLogFrequency = outputSetup.getParameter< uint_t >("performanceLogFrequency");

   Config::BlockHandle statisticsParams    = cfgFile->getBlock("statistics_params");
   const real_t convergenceTolerance       = statisticsParams.getParameter< real_t >("convergenceTolerance");
   const uint_t outputFrequency            = statisticsParams.getParameter< uint_t >("outputFrequency");
   const uint_t planeAveragingtimeBlock    = statisticsParams.getParameter< uint_t >("planeAveragingtimeBlock");
   const uint_t heatFluxAveragingtimeBlock = statisticsParams.getParameter< uint_t >("heatFluxAveragingtimeBlock");


   // convert SI units to simulation (LBM) units and check setup

   Vector3< uint_t > domainSize(uint_c(xSize), uint_c(ySize),
                                uint_c(zSize ));
   WALBERLA_LOG_INFO_ON_ROOT("domain size is " << domainSize);
   WALBERLA_CHECK_EQUAL(domainSize[0], xSize, "domain size in x is not divisible by given dx");
   WALBERLA_CHECK_EQUAL(domainSize[1], ySize, "domain size in y is not divisible by given dx");
   WALBERLA_CHECK_EQUAL(domainSize[2], zSize, "domain size in z is not divisible by given dx");

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
      particleDiameter, 5_r,
      "Your numerical resolution is below 5 cells per diameter and thus too small for such simulations!");

   real_t densityRatio = densityParticle / densityFluid;

   // in simulation units: dt = 1, dx = 1, densityFluid = 1

   const real_t particleVolume   = math::pi / 6_r * particleDiameter * particleDiameter * particleDiameter;
   const real_t poissonsRatio         = real_t(0.22);
   const real_t kappa                 = real_t(2) * (real_t(1) - poissonsRatio) / (real_t(2) - poissonsRatio);
   const real_t particleCollisionTime = collisionTimeFactor * particleDiameter;

   const real_t domainVolume = domainSize[0] * domainSize[1] * domainSize[2];
   const uint_t numParticles = uint_c((volfraction*domainVolume)/(particleVolume));
   WALBERLA_LOG_INFO_ON_ROOT(numParticles << " particles will be created");
   const real_t T_conversion = real_t(1);
   // conversion for the various temperature quantities:
   const real_t rho_0               = densityFluid;
   const real_t particleTemperature = Tparticle;
   const real_t channel_half_width = real_c(domainSize[codegen::wall_axis]/2);

   // calculation of target friction velocity from the thesis of Eschghinejadfard

   const real_t vonkarman_kappa = 2.5;
   const real_t B = 5.5;
   const real_t target_friction_velocity  = center_line_velocity/(vonkarman_kappa*std::log(target_friction_Reynolds) + B);
   const real_t turnOverPeriod = channel_half_width/target_friction_velocity;

   const real_t kinematicViscosityLB = (target_friction_velocity*channel_half_width)/target_friction_Reynolds;

   const real_t thermalDiffusivityFluid_LB = kinematicViscosityLB / Pr;
   const real_t thermalDiffusivityParticle_LB = thermalDiffusivityFluid_LB;


   const real_t omega_f  = lbm::collision_model::omegaFromViscosity(kinematicViscosityLB);
   const real_t omegaT_f = lbm::collision_model::omegaFromViscosity(thermalDiffusivityFluid_LB);
   const real_t omegaT_s = lbm::collision_model::omegaFromViscosity(thermalDiffusivityParticle_LB);
   const uint_t numTimeSteps              =  uint_c(simulationTimeFactor *turnOverPeriod);




   WALBERLA_LOG_INFO_ON_ROOT("total number of timeSteps in simulation " << numTimeSteps);
   WALBERLA_LOG_INFO_ON_ROOT("density particle LB is " << densityParticle);
   WALBERLA_LOG_INFO_ON_ROOT("density fluid LB is " << densityFluid);
   WALBERLA_LOG_INFO_ON_ROOT("Particle Diameter is = " << particleDiameter);

   WALBERLA_LOG_INFO_ON_ROOT("------------------------------");
   WALBERLA_LOG_INFO_ON_ROOT("Extracted Quantities are;   ");
   WALBERLA_LOG_INFO_ON_ROOT("Target Bulk Velocity is " << target_bulk_velocity);
   WALBERLA_LOG_INFO_ON_ROOT("Temperature Relaxation rate fluid is " << omegaT_f);
   WALBERLA_LOG_INFO_ON_ROOT("Hydrodynamic Relaxation rate  fluid is " << omega_f);
   WALBERLA_LOG_INFO_ON_ROOT("Prandtl number = " << (kinematicViscosityLB / thermalDiffusivityFluid_LB));
   WALBERLA_LOG_INFO_ON_ROOT("thermal diffusivity alpha fluid = " <<  thermalDiffusivityFluid_LB);
   WALBERLA_LOG_INFO_ON_ROOT("thermal diffusivity alpha particle = " <<  thermalDiffusivityParticle_LB);



   // outputting turbulent related parameters
   WALBERLA_LOG_INFO_ON_ROOT("Channel half width  " << channel_half_width);
   WALBERLA_LOG_INFO_ON_ROOT("target friction Reynolds number = " <<  target_friction_Reynolds);
   WALBERLA_LOG_INFO_ON_ROOT("target_friction_velocity = " <<  target_friction_velocity);
   WALBERLA_LOG_INFO_ON_ROOT("turnOverPeriod = " <<  turnOverPeriod);
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

   forceParams.wallAxis = codegen::wall_axis;  // example if Y is wall direction

   forceParams.channelHalfWidth = uint_c(ySize / 2.0);

   forceParams.targetBulkVelocity = target_bulk_velocity;

   // example conversion if needed
   forceParams.targetFrictionVelocity = target_friction_velocity;





   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

// domain creation
      std::shared_ptr<StructuredBlockForest> blocks;
      {
         Vector3< uint_t > numBlocks;
         Vector3< uint_t > cellsPerBlock;
         blockforest::calculateCellDistribution(domainSize,
                                                uint_c(mpi::MPIManager::instance()->numProcesses()),
                                                numBlocks, cellsPerBlock);
         Vector3<uint_t> periodicity{true,false,true};

         //const auto & periodicity = periodicity;
         //auto & domainSize = domainSize;
         const Vector3<uint_t> newDomainSize(numBlocks[0] * cellsPerBlock[0], numBlocks[1] * cellsPerBlock[1], numBlocks[2] * cellsPerBlock[2]);

         if(domainSize != newDomainSize) {
            domainSize = newDomainSize;
            WALBERLA_LOG_WARNING_ON_ROOT("\nWARNING: Domain size has changed due to the chosen domain decomposition.\n")
         }

         SetupBlockForest sforest;

         sforest.addWorkloadMemorySUIDAssignmentFunction( blockforest::uniformWorkloadAndMemoryAssignment );

         sforest.init( AABB(0_r, 0_r, 0_r, real_c(domainSize[0]), real_c(domainSize[1]), real_c(domainSize[2])),
                       numBlocks[0], numBlocks[1], numBlocks[2], periodicity[0], periodicity[1], periodicity[2] );

         // calculate process distribution

         const memory_t memoryLimit = numeric_cast< memory_t >( sforest.getNumberOfBlocks() );

         const blockforest::GlobalLoadBalancing::MetisConfiguration< SetupBlock > metisConfig(
            true, false, std::bind( blockforest::cellWeightedCommunicationCost, std::placeholders::_1, std::placeholders::_2,
                                    cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2] ) );

         sforest.calculateProcessDistribution_Default( uint_c( MPIManager::instance()->numProcesses() ), memoryLimit,
                                                       "hilbert", 10, false, metisConfig );

         if( !MPIManager::instance()->rankValid() )
            MPIManager::instance()->useWorldComm();

         // create StructuredBlockForest (encapsulates a newly created BlockForest)

         WALBERLA_LOG_INFO_ON_ROOT("SetupBlockForest created successfully:\n" << sforest)

         sforest.writeVTKOutput("domain_decomposition");

         auto bf = std::make_shared< BlockForest >( uint_c( MPIManager::instance()->rank() ), sforest, false );

         blocks = std::make_shared< StructuredBlockForest >( bf, cellsPerBlock[0], cellsPerBlock[1], cellsPerBlock[2] );
         blocks->createCellBoundingBoxes();

      }

   ////////////////////////
   // ADD DATA TO BLOCKS //
   ///////////////////////

   // Setting initial PDFs to nan helps to detect bugs in the initialization/BC handling


   ////////////////////////////////////////////////
   // Fluid related fields creation on CPU       //
   ///////////////////////////////////////////////


   BlockDataID pdfFieldFluidID =
      field::addToStorage< PdfField_fluid_T >(blocks, "pdf fluid field CPU", real_c(std::nan("")), field::fzyx);

   BlockDataID velFieldFluidID =
      field::addToStorage< VelocityField_fluid_T >(blocks, "velocity fluid field CPU", real_t(0), field::fzyx);

   BlockDataID densityFluidFieldID =
      field::addToStorage< DensityField_fluid_T >(blocks, "density fluid field", real_t(1), field::fzyx);

   BlockDataID flagFieldFluidID       = field::addFlagFieldToStorage< FlagField_T >(blocks, "fluid flag field");



   BlockDataID temperatureFieldID = field::addToStorage< DensityField_temperature_T >(
      blocks, "temperature field", real_t(0), field::fzyx);

   /////////////////////////////////////////////
   // Welford fields for overall statistics  //
   ///////////////////////////////////////////



   // velocity fields
   BlockDataID meanVelFieldID =
      field::addToStorage< VelocityField_fluid_T >(blocks, "mean velocity field CPU", real_t(0), field::fzyx);

   BlockDataID sosVelFieldID =
      field::addToStorage< TensorField_T >(blocks, "sum of squares velocity field CPU", real_t(0), field::fzyx);


   Config::Block boundaryBlock;
   boundaries::createBoundaryConfig(forceParams, boundaryBlock);


   // map boundaries into the fluid field simulation
   using NoSlip_T = lbm::BC_Fluid_NoSlip;
   NoSlip_T noSlip(blocks, pdfFieldFluidID);
   const FlagUID fluidFlagUID("Fluid");
   geometry::initBoundaryHandling< FlagField_T >(*blocks, flagFieldFluidID, Config::BlockHandle(&boundaryBlock));
   geometry::setNonBoundaryCellsToDomain< FlagField_T >(*blocks, flagFieldFluidID, fluidFlagUID);

   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldFluidID, FlagUID("NoSlip"), fluidFlagUID);









   ////////////////////////////////////
   // Initialize the PDFs and Fields //
   ///////////////////////////////////

   // Velocity field setup
   setVelocityFieldsAsmuth<VectorField_T>(
      blocks, velFieldFluidID, meanVelFieldID,
      target_friction_velocity, uint_c(forceParams.channelHalfWidth),
      5.5_r, 0.41_r, kinematicViscosityLB,
      forceParams.wallAxis, 0 );


   // create the force and bulk velocity calculator:
   // use B-field as mask to exclude particle-occupied fractions from bulk velocity.
   ForceCalculator<VectorField_T> forceCalculator(blocks, velFieldFluidID, forceParams);

   //////////////
   /// Setter ///
   //////////////

   WALBERLA_LOG_INFO_ON_ROOT("Setting up fields...")

   // Velocity field setup
   setVelocityFieldsAsmuth<VectorField_T>(
      blocks, velFieldFluidID, meanVelFieldID,
      target_friction_velocity, uint_c(forceParams.channelHalfWidth),
      5.5_r, 0.41_r, kinematicViscosityLB,
      forceParams.wallAxis, 0 );

   forceCalculator.setBulkVelocity(forceParams.targetBulkVelocity);
   const auto initialForce = forceCalculator.calculateDrivingForce();
   using StreamCollideSweep_T = pystencils::TurbulentChannel_Sweep;
   StreamCollideSweep_T streamCollideSweep(pdfFieldFluidID, velFieldFluidID, initialForce, omega_f);

   // Initialize PDFs

   using Setter_T = pystencils::TurbulentChannel_Setter;
   Setter_T pdfSetter(pdfFieldFluidID, velFieldFluidID, initialForce);



   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      pdfSetter(blockIt.get());
   }

   ///////////////////////
   // ADD COMMUNICATION //
   //////////////////////

   // Setup of the fluid LBM communication for synchronizing the fluid pdf field between neighboring blocks

   walberla::blockforest::communication::UniformBufferedScheme< Stencil_Fluid_T > com_fluid(blocks);
   com_fluid.addPackInfo(make_shared< PackInfoFluid_T >(pdfFieldFluidID));



   // time loop objects for communication with and without hiding
   SweepTimeloop timeloop(blocks->getBlockStorage(), numTimeSteps);
   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");







   ////////////////////////////////////////////////////////////////////////////////////////////////
   // add LBM communication, boundary handling and the LBM sweeps to the time loop              //
   //////////////////////////////////////////////////////////////////////////////////////////////

   // setting new force:

   auto setNewForce = [&](const real_t newForce) {
      streamCollideSweep.setForcex(newForce);
   };
   auto streamCollideLambda = [&streamCollideSweep](IBlock * block) {
      streamCollideSweep(block);
   };

   //////////////////////////////////////////
   // Creating the necessary class objects //
   //////////////////////////////////////////

   WcTimingPool timeloopTiming;
   const bool useOpenMP = true;

   ///////////////////////////////////
   // add everything to the timeloop//
   ///////////////////////////////////

   timeloop.add() << BeforeFunction(com_fluid, "LBM fluid Communication")

                  << BeforeFunction([&]() { forceCalculator.calculateBulkVelocity(); }, "bulk velocity calculation")
                  << BeforeFunction(
                        [&]() {
                           const auto newForce = forceCalculator.calculateDrivingForce();
                           //WALBERLA_LOG_INFO_ON_ROOT("new force is  " << newForce);
                           setNewForce(newForce);
                        },
                        "new force setter")
                  << Sweep([](IBlock*) {}, "new force setter");
   timeloop.add() << Sweep((noSlip), "Boundary Handling (No slip fluid)");
   timeloop.add() << Sweep((streamCollideLambda), "streamcollide Fluid sweep");




   for (uint_t timeStep = 0; timeStep < numTimeSteps; ++timeStep)
   {

      // perform a single simulation step -> this contains LBM and setting of the hydrodynamic interactions
      timeloop.singleStep(timeloopTiming);
      auto fluidInfo =
           evaluateFluidInfo(blocks,densityFluidFieldID ,velFieldFluidID,temperatureFieldID);
      if (timeStep%1000 == 0)
      {
         WALBERLA_LOG_INFO_ON_ROOT(fluidInfo);
      }


   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}
} // namespace MaterialTransport

int main(int argc, char** argv) { MaterialTransport::main(argc, argv); }
