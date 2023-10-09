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
//! \file LubricationForceEvaluationPSM.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/debug/TestSubsystem.h"
#include "core/logging/all.h"
#include "core/math/all.h"
#include "core/mpi/Broadcast.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/waLBerlaBuildInfo.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "geometry/InitBoundaryHandling.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/PSMSweepCollectionGPU.h"
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
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "vtk/all.h"

#include <functional>

#include "InitializeDomainForPSM.h"
#include "PSMPackInfo.h"
#include "PSMSweep.h"
#include "PSM_FreeSlip.h"
#include "PSM_InfoHeader.h"
#include "PSM_MacroGetter.h"

namespace lubrication_force_evaluation
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
const FlagUID FreeSlip_Flag("FreeSlip");

//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief Evaluates the hydrodynamic force for the lubrication case for sphere-sphere and sphere-wall case.
 *
 * 4 different setups are available that change the relative velocity to investigate the different components
 * individually. All particles are fixed but have a small velocity which is a valid assumption in Stokes flow. The
 * simulations are run until steady state is reached.
 *
 * see also Rettinger, Ruede 2020 for the details
 */
//*******************************************************************************************************************

int main(int argc, char** argv)
{
   debug::enterTestMode();

   mpi::Environment env(argc, argv);

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));

   bool sphSphTest            = true;
   bool fileIO                = true;
   uint_t vtkIOFreq           = 0;
   std::string fileNameEnding = "";
   std::string baseFolder     = "vtk_out_Lubrication";

   real_t radius         = real_t(5);
   real_t ReynoldsNumber = real_t(1e-2);
   real_t tau            = real_t(1);
   real_t gapSize        = real_t(0);

   // 1: translation in normal direction -> normal Lubrication force
   // 2: translation in tangential direction -> tangential Lubrication force and torque
   // 3: rotation around tangential direction -> force & torque
   // 4: rotation around normal direction -> torque
   uint_t setup = 1;

   for (int i = 1; i < argc; ++i)
   {
      if (std::strcmp(argv[i], "--sphWallTest") == 0)
      {
         sphSphTest = false;
         continue;
      }
      if (std::strcmp(argv[i], "--noLogging") == 0)
      {
         fileIO = false;
         continue;
      }
      if (std::strcmp(argv[i], "--vtkIOFreq") == 0)
      {
         vtkIOFreq = uint_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--setup") == 0)
      {
         setup = uint_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--baseFolder") == 0)
      {
         baseFolder = argv[++i];
         continue;
      }
      if (std::strcmp(argv[i], "--diameter") == 0)
      {
         radius = real_t(0.5) * real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--gapSize") == 0)
      {
         gapSize = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--tau") == 0)
      {
         tau = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--fileName") == 0)
      {
         fileNameEnding = argv[++i];
         continue;
      }
      WALBERLA_ABORT("Unrecognized command line argument found: " << argv[i]);
   }

   ///////////////////////////
   // SIMULATION PROPERTIES //
   ///////////////////////////

   uint_t xSize = uint_c(real_t(24) * radius);
   uint_t ySize = uint_c(real_t(24) * radius);
   uint_t zSize = uint_c(real_t(24) * radius);

   uint_t xBlocks = uint_c(1);      // number of blocks in x-direction
   uint_t yBlocks = uint_c(1);      // number of blocks in y-direction
   uint_t zBlocks = uint_c(1);      // number of blocks in z-direction

   uint_t xCells = xSize / xBlocks; // number of cells in x-direction on each block
   uint_t yCells = ySize / yBlocks; // number of cells in y-direction on each block
   uint_t zCells = zSize / zBlocks; // number of cells in z-direction on each block

   // Perform missing variable calculations
   real_t omega    = real_t(1) / tau;
   real_t nu       = walberla::lbm::collision_model::viscosityFromOmega(omega);
   real_t velocity = ReynoldsNumber * nu / (real_t(2) * radius);

   uint_t timesteps = uint_c(10000);

   real_t fStokes = real_t(6) * math::pi * nu * radius * velocity;
   real_t tStokes = real_t(8) * math::pi * nu * radius * radius * velocity;

   WALBERLA_LOG_INFO_ON_ROOT_SECTION()
   {
      std::stringstream ss;

      if (sphSphTest)
      {
         ss << "-------------------------------------------------------\n"
            << "   Parameters for the sphere-sphere lubrication test \n"
            << "-------------------------------------------------------\n";
      }
      else
      {
         ss << "-------------------------------------------------------\n"
            << "   Parameters for the sphere-wall lubrication test \n"
            << "-------------------------------------------------------\n";
      }
      ss << " omega        = " << omega << "\n"
         << " radius       = " << radius << "\n"
         << " velocity     = " << velocity << "\n"
         << " Re           = " << ReynoldsNumber << "\n"
         << " gap size     = " << gapSize << "\n"
         << " time steps   = " << timesteps << "\n"
         << " fStokes      = " << fStokes << "\n"
         << " setup        = " << setup << "\n"
         << "-------------------------------------------------------\n"
         << " domainSize = " << xSize << " x " << ySize << " x " << zSize << "\n"
         << " blocks     = " << xBlocks << " x " << yBlocks << " x " << zBlocks << "\n"
         << " blockSize  = " << xCells << " x " << yCells << " x " << zCells << "\n"
         << "-------------------------------------------------------\n";
      WALBERLA_LOG_INFO(ss.str());
   }

   auto blocks = blockforest::createUniformBlockGrid(xBlocks, yBlocks, zBlocks, xCells, yCells, zCells, real_t(1), 0,
                                                     false, false, sphSphTest, true, true, // periodicity
                                                     false);

   //////////////////
   // RPD COUPLING //
   //////////////////

   auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

   // init data structures
   auto ps                  = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto ss                  = walberla::make_shared< mesa_pd::data::ShapeStorage >();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);

   auto sphereShape = ss->create< mesa_pd::data::Sphere >(radius);

   uint_t id1(0);
   uint_t id2(0);

   uint_t randomSeed = uint_c(std::chrono::system_clock::now().time_since_epoch().count());
   mpi::broadcastObject(randomSeed); // root process chooses seed and broadcasts it
   std::mt19937 randomNumberGenerator(static_cast< unsigned int >(randomSeed));

   Vector3< real_t > domainCenter(real_c(xSize) * real_t(0.5), real_c(ySize) * real_t(0.5),
                                  real_c(zSize) * real_t(0.5));
   Vector3< real_t > offsetVector(math::realRandom< real_t >(real_t(0), real_t(1), randomNumberGenerator),
                                  math::realRandom< real_t >(real_t(0), real_t(1), randomNumberGenerator),
                                  math::realRandom< real_t >(real_t(0), real_t(1), randomNumberGenerator));

   if (sphSphTest)
   {
      Vector3< real_t > pos1 = domainCenter + offsetVector;
      if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pos1))
      {
         mesa_pd::data::Particle&& p = *ps->create();
         p.setPosition(pos1);
         p.setInteractionRadius(radius);
         p.setOwner(mpi::MPIManager::instance()->rank());
         p.setShapeID(sphereShape);
         if (setup == 1)
         {
            // only normal translational vel
            p.setLinearVelocity(Vector3< real_t >(velocity, real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
         }
         else if (setup == 2)
         {
            // only tangential translational velocity
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), velocity));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
         }
         else if (setup == 3)
         {
            // only rotation around axis perpendicular to center line
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), velocity / radius, real_t(0)));
         }
         else if (setup == 4)
         {
            // only rotation around center line
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(velocity / radius, real_t(0), real_t(0)));
         }
         id1 = p.getUid();
      }

      Vector3< real_t > pos2 = pos1 + Vector3< real_t >(real_t(2) * radius + gapSize, real_t(0), real_t(0));
      if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pos2))
      {
         mesa_pd::data::Particle&& p = *ps->create();
         p.setPosition(pos2);
         p.setInteractionRadius(radius);
         p.setOwner(mpi::MPIManager::instance()->rank());
         p.setShapeID(sphereShape);
         if (setup == 1)
         {
            // only normal translational vel
            p.setLinearVelocity(Vector3< real_t >(-velocity, real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
         }
         else if (setup == 2)
         {
            // only tangential translational velocity
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), -velocity));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
         }
         else if (setup == 3)
         {
            // only rotation around axis perpendicular to center line
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), velocity / radius, real_t(0)));
         }
         else if (setup == 4)
         {
            // only rotation around center line
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(-velocity / radius, real_t(0), real_t(0)));
         }
         id2 = p.getUid();
      }

      mpi::allReduceInplace(id1, mpi::SUM);
      mpi::allReduceInplace(id2, mpi::SUM);

      WALBERLA_LOG_INFO_ON_ROOT("pos sphere 1 = " << pos1);
      WALBERLA_LOG_INFO_ON_ROOT("pos sphere 2 = " << pos2);
   }
   else
   {
      // sphere-wall test

      Vector3< real_t > referenceVector(offsetVector[0], domainCenter[1], domainCenter[2]);

      // create two planes
      mesa_pd::data::Particle&& p0 = *ps->create(true);
      p0.setPosition(referenceVector);
      p0.setInteractionRadius(std::numeric_limits< real_t >::infinity());
      p0.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(1, 0, 0)));
      p0.setOwner(mpi::MPIManager::instance()->rank());
      mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
      mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
      id2 = p0.getUid();

      mesa_pd::data::Particle&& p1 = *ps->create(true);
      p1.setPosition(Vector3< real_t >(real_c(xSize), 0, 0));
      p1.setInteractionRadius(std::numeric_limits< real_t >::infinity());
      p1.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(-1, 0, 0)));
      p1.setOwner(mpi::MPIManager::instance()->rank());
      mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
      mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

      Vector3< real_t > pos1 = referenceVector + Vector3< real_t >(radius + gapSize, real_t(0), real_t(0));
      if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), pos1))
      {
         mesa_pd::data::Particle&& p = *ps->create();
         p.setPosition(pos1);
         p.setInteractionRadius(radius);
         p.setOwner(mpi::MPIManager::instance()->rank());
         p.setShapeID(sphereShape);
         if (setup == 1)
         {
            // only normal translational vel
            p.setLinearVelocity(Vector3< real_t >(-velocity, real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
         }
         else if (setup == 2)
         {
            // only tangential translational velocity
            p.setLinearVelocity(Vector3< real_t >(real_t(0), velocity, real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
         }
         else if (setup == 3)
         {
            // only rotation around axis perpendicular to center line
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(real_t(0), velocity / radius, real_t(0)));
         }
         else if (setup == 4)
         {
            // only rotation around center line
            p.setLinearVelocity(Vector3< real_t >(real_t(0), real_t(0), real_t(0)));
            p.setAngularVelocity(Vector3< real_t >(velocity / radius, real_t(0), real_t(0)));
         }
         id1 = p.getUid();
      }

      mpi::allReduceInplace(id1, mpi::SUM);
      // id2 is globally known

      WALBERLA_LOG_INFO_ON_ROOT("pos plane = " << referenceVector);
      WALBERLA_LOG_INFO_ON_ROOT("pos sphere = " << pos1);
   }

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // add PDF field
   BlockDataID pdfFieldID =
      field::addToStorage< PdfField_T >(blocks, "pdf field (fzyx)", real_c(std::nan("")), field::fzyx);
   BlockDataID pdfFieldGPUID = gpu::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldID, "pdf field gpu");
   // add flag field
   BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

   BlockDataID densityFieldID = field::addToStorage< DensityField_T >(blocks, "Density", real_t(0), field::fzyx);
   BlockDataID velFieldID     = field::addToStorage< VelocityField_T >(blocks, "Velocity", real_t(0), field::fzyx);

   // assemble boundary block string
   std::string boundariesBlockString = " Boundaries"
                                       "{"
                                       "Border { direction T;    walldistance -1;  flag FreeSlip; }"
                                       "Border { direction B;    walldistance -1;  flag FreeSlip; }"
                                       "Border { direction N;    walldistance -1;  flag FreeSlip; }"
                                       "Border { direction S;    walldistance -1;  flag FreeSlip; }"
                                       "}";

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

   lbm::PSM_FreeSlip freeSlip(blocks, pdfFieldGPUID);
   freeSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldID, FreeSlip_Flag, Fluid_Flag);

   // set up RPD functionality
   std::function< void(void) > syncCall = [&ps, &rpdDomain]() {
      mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
      syncNextNeighborFunc(*ps, *rpdDomain);
   };

   syncCall();

   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;

   real_t lubricationCutOffDistanceNormal                  = real_t(2) / real_t(3);
   real_t lubricationCutOffDistanceTangentialTranslational = real_t(0.5);
   real_t lubricationCutOffDistanceTangentialRotational    = real_t(0.5);
   lbm_mesapd_coupling::LubricationCorrectionKernel lubricationCorrectionKernel(
      nu, [](real_t) { return real_t(0); }, lubricationCutOffDistanceNormal,
      lubricationCutOffDistanceTangentialTranslational, lubricationCutOffDistanceTangentialRotational);
   real_t maximalCutOffDistance =
      std::max(lubricationCutOffDistanceNormal, std::max(lubricationCutOffDistanceTangentialTranslational,
                                                         lubricationCutOffDistanceTangentialRotational));

   lbm_mesapd_coupling::RegularParticlesSelector sphereSelector;

   ///////////////
   // TIME LOOP //
   ///////////////

   ParticleAndVolumeFractionSoA_T< Weighting > particleAndVolumeFractionSoA(blocks, omega);
   PSMSweepCollectionGPU psmSweepCollection(blocks, accessor, sphereSelector, particleAndVolumeFractionSoA, 10);
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

   // create the timeloop
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   // setup of the LBM communication for synchronizing the pdf field between neighboring blocks
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, 0, false);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldGPUID));
   auto communication = std::function< void() >([&]() { com.communicate(nullptr); });

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   pystencils::PSM_MacroGetter getterSweep(densityFieldID, pdfFieldID, velFieldID, real_t(0.0), real_t(0.0),
                                           real_t(0.0));

   if (vtkIOFreq != uint_t(0))
   {
      // spheres
      auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleOwner >("owner");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      auto particleVtkWriter =
         vtk::createVTKOutput_PointData(particleVtkOutput, "Particles", vtkIOFreq, baseFolder, "simulation_step");
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(particleVtkWriter), "VTK (sphere data)");

      // flag field (written only once in the first time step, ghost layers are also written)
      auto flagFieldVTK =
         vtk::createVTKOutput_BlockData(blocks, "flag_field", vtkIOFreq, FieldGhostLayers, false, baseFolder);
      flagFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< FlagField_T > >(flagFieldID, "FlagField"));
      vtk::writeFiles(flagFieldVTK)();

      // pdf field
      auto pdfFieldVTK = vtk::createVTKOutput_BlockData(blocks, "fluid_field", vtkIOFreq, 0, false, baseFolder);

      pdfFieldVTK->addBeforeFunction(communication);

      pdfFieldVTK->addBeforeFunction([&]() {
         gpu::fieldCpy< PdfField_T, gpu::GPUField< real_t > >(blocks, pdfFieldID, pdfFieldGPUID);
         for (auto& block : *blocks)
            getterSweep(&block);
      });

      field::FlagFieldCellFilter< FlagField_T > fluidFilter(flagFieldID);
      fluidFilter.addFlag(Fluid_Flag);
      pdfFieldVTK->addCellInclusionFilter(fluidFilter);

      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< VelocityField_T > >(velFieldID, "Velocity"));
      pdfFieldVTK->addCellDataWriter(make_shared< field::VTKWriter< DensityField_T > >(densityFieldID, "Density"));

      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(pdfFieldVTK), "VTK (fluid field data)");
   }

   // add LBM communication function and boundary handling sweep (does the hydro force calculations and the no-slip
   // treatment)

   timeloop.add() << BeforeFunction(communication, "LBM Communication")
                  << Sweep(freeSlip.getSweep(), "Boundary Handling (FreeSlip)");

   pystencils::PSMSweep PSMSweep(particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
                                 particleAndVolumeFractionSoA.particleForcesFieldID,
                                 particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0.0),
                                 real_t(0.0), real_t(0.0), omega);
   addPSMSweepsToTimeloop(timeloop, psmSweepCollection, PSMSweep);

   ////////////////////////
   // EXECUTE SIMULATION //
   ////////////////////////

   WcTimingPool timeloopTiming;

   Vector3< real_t > hydForce(0.);
   Vector3< real_t > lubForce(0.);
   Vector3< real_t > hydTorque(0.);
   Vector3< real_t > lubTorque(0.);

   real_t curForceNorm  = real_t(0);
   real_t oldForceNorm  = real_t(0);
   real_t curTorqueNorm = real_t(0);
   real_t oldTorqueNorm = real_t(0);

   real_t convergenceLimit = real_t(1e-5);

   // time loop
   for (uint_t i = 1; i <= timesteps; ++i)
   {
      // perform a single simulation step -> this contains LBM and setting of the hydrodynamic interactions
      timeloop.singleStep(timeloopTiming);

      // lubrication correction
      mesa_pd::collision_detection::AnalyticContactDetection acd;
      acd.getContactThreshold() = maximalCutOffDistance;

      {
         auto idx1 = accessor->uidToIdx(id1);
         if (idx1 != accessor->getInvalidIdx())
         {
            auto idx2 = accessor->uidToIdx(id2);
            if (idx2 != accessor->getInvalidIdx())
            {
               mesa_pd::kernel::DoubleCast double_cast;
               mesa_pd::mpi::ContactFilter contact_filter;
               if (double_cast(idx1, idx2, *accessor, acd, *accessor))
               {
                  if (contact_filter(acd.getIdx1(), acd.getIdx2(), *accessor, acd.getContactPoint(), *rpdDomain))
                  {
                     double_cast(acd.getIdx1(), acd.getIdx2(), *accessor, lubricationCorrectionKernel, *accessor,
                                 acd.getContactNormal(), acd.getPenetrationDepth());
                  }
               }
            }
         }
      }

      if (i % 100 == 0 && i > 1)
      {
         oldForceNorm  = curForceNorm;
         oldTorqueNorm = curTorqueNorm;

         hydForce.reset();
         lubForce.reset();
         hydTorque.reset();
         lubTorque.reset();

         auto idx1 = accessor->uidToIdx(id1);
         if (idx1 != accessor->getInvalidIdx())
         {
            hydForce  = accessor->getHydrodynamicForce(idx1);
            lubForce  = accessor->getForce(idx1);
            hydTorque = accessor->getHydrodynamicTorque(idx1);
            lubTorque = accessor->getTorque(idx1);
         }

         WALBERLA_MPI_SECTION()
         {
            mpi::allReduceInplace(hydForce, mpi::SUM);
            mpi::reduceInplace(lubForce, mpi::SUM);
            mpi::allReduceInplace(hydTorque, mpi::SUM);
            mpi::reduceInplace(lubTorque, mpi::SUM);
         }

         curForceNorm  = hydForce.length();
         curTorqueNorm = hydTorque.length();

         real_t forceDiff  = std::fabs((curForceNorm - oldForceNorm) / oldForceNorm);
         real_t torqueDiff = std::fabs((curTorqueNorm - oldTorqueNorm) / oldTorqueNorm);

         WALBERLA_LOG_INFO_ON_ROOT("F/Fs = " << hydForce / fStokes << " ( " << forceDiff
                                             << " ), T/Ts = " << hydTorque / tStokes << " ( " << torqueDiff << " )");

         if (i == 100)
         {
            WALBERLA_LOG_INFO_ON_ROOT("Flub = " << lubForce << ", Tlub = " << lubTorque);
            WALBERLA_LOG_INFO_ON_ROOT("Flub/Fs = " << lubForce / fStokes << ", Tlub/Ts = " << lubTorque / tStokes);
         }

         if (forceDiff < convergenceLimit && torqueDiff < convergenceLimit)
         {
            WALBERLA_LOG_INFO_ON_ROOT("Force and torque norms converged - terminating simulation");
            break;
         }
      }

      // reset forces
      ps->forEachParticle(
         false, mesa_pd::kernel::SelectAll(), *accessor,
         [](const size_t idx, auto& ac) {
            ac.getForceRef(idx)  = Vector3< real_t >(real_t(0));
            ac.getTorqueRef(idx) = Vector3< real_t >(real_t(0));
         },
         *accessor);
      ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor);
   }

   if (fileIO)
   {
      std::string loggingFileName(baseFolder + "/Logging_");
      std::string executableName = argv[0];
      size_t lastSlash           = executableName.find_last_of("/\\");
      if (lastSlash != std::string::npos) { loggingFileName += executableName.substr(lastSlash + 1); }
      else { loggingFileName += executableName; }
      if (sphSphTest)
         loggingFileName += "_SphSph";
      else
         loggingFileName += "_SphPla";
      loggingFileName += "_Setup" + std::to_string(setup);
      loggingFileName += "_gapSize" + std::to_string(uint_c(gapSize * real_t(100)));
      loggingFileName += "_radius" + std::to_string(uint_c(radius));
      if (!fileNameEnding.empty()) loggingFileName += "_" + fileNameEnding;
      loggingFileName += ".txt";

      WALBERLA_ROOT_SECTION()
      {
         std::ofstream file1;
         file1.open(loggingFileName.c_str(), std::ofstream::app);
         file1.setf(std::ios::unitbuf);
         file1.precision(15);
         file1 << radius << " " << gapSize << " " << fStokes << " " << hydForce[0] << " " << hydForce[1] << " "
               << hydForce[2] << " " << lubForce[0] << " " << lubForce[1] << " " << lubForce[2] << " " << hydTorque[0]
               << " " << hydTorque[1] << " " << hydTorque[2] << " " << lubTorque[0] << " " << lubTorque[1] << " "
               << lubTorque[2] << std::endl;
         file1.close();
      }
   }

   timeloopTiming.logResultOnRoot();

   return EXIT_SUCCESS;
}

} // namespace lubrication_force_evaluation

int main(int argc, char** argv) { lubrication_force_evaluation::main(argc, argv); }
