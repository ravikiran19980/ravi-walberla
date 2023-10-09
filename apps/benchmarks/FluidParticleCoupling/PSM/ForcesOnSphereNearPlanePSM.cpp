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
//! \file ForcesOnSphereNearPlanePSM.cpp
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//
//======================================================================================================================

#include "blockforest/Initialization.h"

#include "core/Environment.h"
#include "core/debug/TestSubsystem.h"
#include "core/logging/all.h"
#include "core/math/all.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/waLBerlaBuildInfo.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "geometry/InitBoundaryHandling.h"

#include "gpu/AddGPUFieldToStorage.h"
#include "gpu/DeviceSelectMPI.h"
#include "gpu/communication/UniformGPUScheme.h"

#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypesGPU.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/cuda/PSMSweepCollectionGPU.h"
#include "lbm_mesapd_coupling/utility/AverageHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/InitializeHydrodynamicForceTorqueForAveragingKernel.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"

#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/HydrodynamicForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "vtk/all.h"

#include <functional>

#include "PSMPackInfo.h"
#include "PSMSweep.h"
#include "PSM_InfoHeader.h"
#include "PSM_MacroGetter.h"
#include "PSM_MacroSetter.h"
#include "PSM_NoSlip.h"
#include "PSM_UBB.h"

namespace forces_on_sphere_near_plane
{

///////////
// USING //
///////////

using namespace walberla;
using walberla::uint_t;
using namespace lbm_mesapd_coupling::psm::gpu;

//////////////
// TYPEDEFS //
//////////////

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

typedef pystencils::PSMPackInfo PackInfo_T;

const uint_t FieldGhostLayers = 1;

///////////
// FLAGS //
///////////

const FlagUID Fluid_Flag("Fluid");
const FlagUID NoSlip_Flag("NoSlip");
const FlagUID Velocity_Flag("Velocity");

template< typename ParticleAccessor_T >
class SpherePropertyLogger
{
 public:
   SpherePropertyLogger(const shared_ptr< ParticleAccessor_T >& ac, walberla::id_t sphereUid,
                        const std::string& fileName, bool fileIO, real_t dragNormalizationFactor,
                        real_t liftNormalizationFactor, real_t physicalTimeScale)
      : ac_(ac), sphereUid_(sphereUid), fileName_(fileName), fileIO_(fileIO),
        dragNormalizationFactor_(dragNormalizationFactor), liftNormalizationFactor_(liftNormalizationFactor),
        physicalTimeScale_(physicalTimeScale)
   {
      if (fileIO_)
      {
         WALBERLA_ROOT_SECTION()
         {
            std::ofstream file;
            file.open(fileName_.c_str());
            file << "#\t t\t Cd\t Cl\t fX\t fY\t fZ\t tX\t tY\t tZ\n";
            file.close();
         }
      }
   }

   void operator()(const uint_t timestep)
   {
      Vector3< real_t > force(real_t(0));
      Vector3< real_t > torque(real_t(0));

      size_t idx = ac_->uidToIdx(sphereUid_);
      if (idx != ac_->getInvalidIdx())
      {
         if (!mesa_pd::data::particle_flags::isSet(ac_->getFlags(idx), mesa_pd::data::particle_flags::GHOST))
         {
            force  = ac_->getHydrodynamicForce(idx);
            torque = ac_->getHydrodynamicTorque(idx);
         }
      }

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace(force, mpi::SUM);
         mpi::allReduceInplace(torque, mpi::SUM);
      }

      if (fileIO_) writeToFile(timestep, force, torque);

      dragForce_ = force[0];
      liftForce_ = force[2];
   }

   real_t getDragForce() { return dragForce_; }

   real_t getLiftForce() { return liftForce_; }

   real_t getDragCoefficient() { return dragForce_ / dragNormalizationFactor_; }

   real_t getLiftCoefficient() { return liftForce_ / liftNormalizationFactor_; }

 private:
   void writeToFile(uint_t timestep, const Vector3< real_t >& force, const Vector3< real_t >& torque)
   {
      WALBERLA_ROOT_SECTION()
      {
         std::ofstream file;
         file.open(fileName_.c_str(), std::ofstream::app);

         file << timestep << "\t" << real_c(timestep) / physicalTimeScale_ << "\t"
              << force[0] / dragNormalizationFactor_ << "\t" << force[2] / liftNormalizationFactor_ << "\t" << force[0]
              << "\t" << force[1] << "\t" << force[2] << "\t" << torque[0] << "\t" << torque[1] << "\t" << torque[2]
              << "\n";
         file.close();
      }
   }

   shared_ptr< ParticleAccessor_T > ac_;
   const walberla::id_t sphereUid_;
   std::string fileName_;
   bool fileIO_;
   real_t dragForce_, liftForce_;
   real_t dragNormalizationFactor_, liftNormalizationFactor_;
   real_t physicalTimeScale_;
};

void initializeCouetteProfile(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& velFieldID,
                              const real_t& domainHeight, const real_t wallVelocity)
{
   for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
   {
      auto velField = blockIt->getData< VelocityField_T >(velFieldID);

      WALBERLA_FOR_ALL_CELLS_XYZ(velField,

                                 const Vector3< real_t > coord =
                                    blocks->getBlockLocalCellCenter(*blockIt, Cell(x, y, z));

                                 Vector3< real_t > velocity(real_c(0));

                                 velocity[0] = wallVelocity * coord[2] / domainHeight;

                                 velField->get(x, y, z, 0) = velocity[0]; velField->get(x, y, z, 1) = velocity[1];
                                 velField->get(x, y, z, 2)                                          = velocity[2];)
   }
}

void logFinalResult(const std::string& fileName, real_t Re, real_t wallDistance, real_t diameter, uint_t domainLength,
                    uint_t domainWidth, uint_t domainHeight, real_t dragCoeff, real_t dragCoeffRef, real_t liftCoeff,
                    real_t liftCoeffRef, uint_t timestep, real_t nonDimTimestep)
{
   WALBERLA_ROOT_SECTION()
   {
      std::ofstream file;
      file.open(fileName.c_str(), std::ofstream::app);

      file << Re << "\t " << wallDistance << "\t " << diameter << "\t " << domainLength << "\t " << domainWidth << "\t "
           << domainHeight << "\t " << dragCoeff << "\t " << dragCoeffRef << "\t "
           << std::abs(dragCoeff - dragCoeffRef) / dragCoeffRef << "\t " << liftCoeff << "\t " << liftCoeffRef << "\t "
           << std::abs(liftCoeff - liftCoeffRef) / liftCoeffRef << "\t " << timestep << "\t " << nonDimTimestep << "\n";
      file.close();
   }
}

//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief Testcase that evaluates the drag and lift force on a sphere that is close to the bottom plane in shear flow
 *
 * see overview paper:
 * Zeng, Najjar, Balachandar, Fischer - "Forces on a finite-sized particle located close to a wall in a linear shear
 * flow", 2009
 *
 * contains references to:
 * Leighton, Acrivos - "The lift on a small sphere touching a plane in the presence of a simple shear flow", 1985
 * Zeng, Balachandar, Fischer - "Wall-induced forces on a rigid sphere at finite Reynolds number", 2005
 *
 * CFD-IBM simulations in:
 * Lee, Balachandar - "Drag and lift forces on a spherical particle moving on a wall in a shear flow at finite Re", 2010
 *
 * Description:
 *  - Domain size [x, y, z] = [48 x 16 x 8 ] * diameter = [L(ength), W(idth), H(eight)]
 *  - horizontally periodic, bounded by two bcs in z-direction
 *  - top bc is constant wall velocity -> shear flow
 *  - sphere is placed in the vicinity of the bottom plane at [ L/2 + xOffset, W/2 + yOffset, wallDistance * diameter]
 *  - distance of sphere center to the bottom plane is crucial parameter
 *  - viscosity is adjusted to match specified Reynolds number = shearRate * diameter * wallDistance / viscosity
 *  - dimensionless drag and lift forces are evaluated and written to logging file
 */
//*******************************************************************************************************************

int main(int argc, char** argv)
{
   debug::enterTestMode();

   mpi::Environment env(argc, argv);
   gpu::selectDeviceBasedOnMpiRank();

   WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));

   ///////////////////
   // Customization //
   ///////////////////

   // simulation control
   bool fileIO                   = true;
   uint_t vtkIOFreq              = 0;
   std::string baseFolderVTK     = "vtk_out_ForcesNearPlane";
   std::string baseFolderLogging = ".";

   // physical setup
   real_t diameter = real_t(20); // cells per diameter -> determines overall resolution
   real_t normalizedWallDistance =
      real_t(1);                 // distance of the sphere center to the bottom wall, normalized by the diameter
   real_t ReynoldsNumberShear = real_t(1); // = shearRate * wallDistance * diameter / viscosity

   // numerical parameters
   real_t maximumNonDimTimesteps  = real_t(100); // maximum number of non-dimensional time steps
   real_t xOffsetOfSpherePosition = real_t(0);   // offset in x-direction of sphere position
   real_t yOffsetOfSpherePosition = real_t(0);   // offset in y-direction of sphere position
   real_t wallVelocity            = real_t(0.1);

   bool initializeVelocityProfile = false;

   real_t relativeChangeConvergenceEps = real_t(1e-5);
   real_t physicalCheckingFrequency    = real_t(0.1);

   // command line arguments
   for (int i = 1; i < argc; ++i)
   {
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
      if (std::strcmp(argv[i], "--diameter") == 0)
      {
         diameter = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--timesteps") == 0)
      {
         maximumNonDimTimesteps = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--wallDistance") == 0)
      {
         normalizedWallDistance = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--Re") == 0)
      {
         ReynoldsNumberShear = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--velocity") == 0)
      {
         wallVelocity = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--xOffset") == 0)
      {
         xOffsetOfSpherePosition = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--yOffset") == 0)
      {
         yOffsetOfSpherePosition = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--baseFolderVTK") == 0)
      {
         baseFolderVTK = argv[++i];
         continue;
      }
      if (std::strcmp(argv[i], "--baseFolderLogging") == 0)
      {
         baseFolderLogging = argv[++i];
         continue;
      }
      if (std::strcmp(argv[i], "--initializeVelocityProfile") == 0)
      {
         initializeVelocityProfile = true;
         continue;
      }
      if (std::strcmp(argv[i], "--convergenceLimit") == 0)
      {
         relativeChangeConvergenceEps = real_c(std::atof(argv[++i]));
         continue;
      }
      if (std::strcmp(argv[i], "--checkingFrequency") == 0)
      {
         physicalCheckingFrequency = real_c(std::atof(argv[++i]));
         continue;
      }
      WALBERLA_ABORT("Unrecognized command line argument found: " << argv[i]);
   }

   WALBERLA_CHECK_GREATER_EQUAL(normalizedWallDistance, real_t(0.5));
   WALBERLA_CHECK_GREATER_EQUAL(ReynoldsNumberShear, real_t(0));
   WALBERLA_CHECK_GREATER_EQUAL(diameter, real_t(0));

   //////////////////////////
   // NUMERICAL PARAMETERS //
   //////////////////////////

   const real_t domainLength = real_t(48) * diameter; // x
   const real_t domainWidth  = real_t(16) * diameter; // y
   const real_t domainHeight = real_t(8) * diameter;  // z

   Vector3< uint_t > domainSize(uint_c(std::ceil(domainLength)), uint_c(std::ceil(domainWidth)),
                                uint_c(std::ceil(domainHeight)));

   const real_t wallDistance        = diameter * normalizedWallDistance;
   const real_t shearRate           = wallVelocity / domainHeight;
   const real_t velAtSpherePosition = shearRate * wallDistance;
   const real_t viscosity           = velAtSpherePosition * diameter / ReynoldsNumberShear;

   const real_t relaxationTime = real_t(1) / lbm::collision_model::omegaFromViscosity(viscosity);

   const real_t densityFluid = real_t(1);

   const real_t dx = real_t(1);

   const real_t physicalTimeScale = diameter / velAtSpherePosition;
   const uint_t timesteps         = uint_c(maximumNonDimTimesteps * physicalTimeScale);

   const real_t omega = real_t(1) / relaxationTime;

   Vector3< real_t > initialPosition(domainLength * real_t(0.5) + xOffsetOfSpherePosition,
                                     domainWidth * real_t(0.5) + yOffsetOfSpherePosition, wallDistance);

   WALBERLA_LOG_INFO_ON_ROOT("Setup:");
   WALBERLA_LOG_INFO_ON_ROOT(" - domain size = " << domainSize);
   WALBERLA_LOG_INFO_ON_ROOT(" - normalized wall distance = " << normalizedWallDistance);
   WALBERLA_LOG_INFO_ON_ROOT(" - shear rate = " << shearRate);
   WALBERLA_LOG_INFO_ON_ROOT(" - wall velocity = " << wallVelocity);
   WALBERLA_LOG_INFO_ON_ROOT(" - Reynolds number (shear rate based) = "
                             << ReynoldsNumberShear << ", vel at sphere pos = " << velAtSpherePosition);
   WALBERLA_LOG_INFO_ON_ROOT(" - density = " << densityFluid);
   WALBERLA_LOG_INFO_ON_ROOT(" - viscosity = " << viscosity << " -> omega = " << omega
                                               << " , tau = " << relaxationTime);
   WALBERLA_LOG_INFO_ON_ROOT(" - sphere diameter = " << diameter << ", position = " << initialPosition
                                                     << " ( xOffset = " << xOffsetOfSpherePosition
                                                     << ", yOffset = " << yOffsetOfSpherePosition << " )");
   WALBERLA_LOG_INFO_ON_ROOT(" - base folder VTK = " << baseFolderVTK
                                                     << ", base folder logging = " << baseFolderLogging);

   ///////////////////////////
   // BLOCK STRUCTURE SETUP //
   ///////////////////////////

   Vector3< uint_t > blocksPerDirection(uint_t(MPIManager::instance()->numProcesses()), 1, 1);
   WALBERLA_CHECK_EQUAL(blocksPerDirection[0] * blocksPerDirection[1] * blocksPerDirection[2],
                        uint_t(MPIManager::instance()->numProcesses()),
                        "When using GPUs, the number of blocks ("
                           << blocksPerDirection[0] * blocksPerDirection[1] * blocksPerDirection[2]
                           << ") has to match the number of MPI processes ("
                           << uint_t(MPIManager::instance()->numProcesses()) << ")");

   // Vector3<uint_t> blocksPerDirection( 3, 3, 1 );

   WALBERLA_CHECK(domainSize[0] % blocksPerDirection[0] == 0 && domainSize[1] % blocksPerDirection[1] == 0 &&
                  domainSize[2] % blocksPerDirection[2] == 0);
   Vector3< uint_t > blockSizeInCells(domainSize[0] / (blocksPerDirection[0]), domainSize[1] / (blocksPerDirection[1]),
                                      domainSize[2] / (blocksPerDirection[2]));

   AABB simulationDomain(real_t(0), real_t(0), real_t(0), real_c(domainSize[0]), real_c(domainSize[1]),
                         real_c(domainSize[2]));
   auto blocks = blockforest::createUniformBlockGrid(
      blocksPerDirection[0], blocksPerDirection[1], blocksPerDirection[2], blockSizeInCells[0], blockSizeInCells[1],
      blockSizeInCells[2], dx, 0, false, false, true, true, false, // periodicity
      false);

   WALBERLA_LOG_INFO_ON_ROOT(" - blocks = " << blocksPerDirection << ", block size = " << blockSizeInCells);

   // write domain decomposition to file
   if (vtkIOFreq > 0) { vtk::writeDomainDecomposition(blocks, "initial_domain_decomposition", baseFolderVTK); }

   /////////////////
   // PE COUPLING //
   /////////////////

   auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

   // init data structures
   auto ps                  = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
   auto ss                  = walberla::make_shared< mesa_pd::data::ShapeStorage >();
   using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
   auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);

   // create sphere and store Uid
   auto sphereShape = ss->create< mesa_pd::data::Sphere >(diameter * real_t(0.5));

   walberla::id_t sphereUid = 0;
   if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), initialPosition))
   {
      mesa_pd::data::Particle&& p = *ps->create();
      p.setPosition(initialPosition);
      p.setInteractionRadius(diameter * real_t(0.5));
      p.setOwner(mpi::MPIManager::instance()->rank());
      p.setShapeID(sphereShape);
      sphereUid = p.getUid();
   }
   mpi::allReduceInplace(sphereUid, mpi::SUM);

   // set up RPD functionality
   std::function< void(void) > syncCall = [ps, rpdDomain]() {
      mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
      syncNextNeighborFunc(*ps, *rpdDomain);
   };
   syncCall();

   mesa_pd::mpi::ReduceProperty reduceProperty;

   ///////////////////////
   // ADD DATA TO BLOCKS //
   ////////////////////////

   // add PDF field
   BlockDataID pdfFieldID =
      field::addToStorage< PdfField_T >(blocks, "pdf field (fzyx)", real_c(std::nan("")), field::fzyx);
   BlockDataID pdfFieldGPUID = gpu::addGPUFieldToStorage< PdfField_T >(blocks, pdfFieldID, "pdf field GPU");

   BlockDataID densityFieldID = field::addToStorage< DensityField_T >(blocks, "Density", real_t(0), field::fzyx);
   BlockDataID velFieldID     = field::addToStorage< VelocityField_T >(blocks, "Velocity", real_t(0), field::fzyx);

   // add flag field
   BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field", FieldGhostLayers);

   // set up coupling functionality
   lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
   lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;

   // initialize Couette velocity profile in whole domain
   if (initializeVelocityProfile)
   {
      WALBERLA_LOG_INFO_ON_ROOT("Initializing Couette velocity profile.");
      initializeCouetteProfile(blocks, velFieldID, domainHeight, wallVelocity);
      pystencils::PSM_MacroSetter setterSweep(pdfFieldID, velFieldID, real_t(0), real_t(0), real_t(0));
      for (auto& block : *blocks)
         setterSweep(&block);
   }
   gpu::fieldCpy< gpu::GPUField< real_t >, PdfField_T >(blocks, pdfFieldGPUID, pdfFieldID);

   // assemble boundary block string
   std::string boundariesBlockString = " Boundaries"
                                       "{"
                                       "Border { direction T;    walldistance -1;  flag Velocity; }"
                                       "Border { direction B;    walldistance -1;  flag NoSlip; }"
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
   lbm::PSM_NoSlip noSlip(blocks, pdfFieldGPUID);
   noSlip.fillFromFlagField< FlagField_T >(blocks, flagFieldID, NoSlip_Flag, Fluid_Flag);
   lbm::PSM_UBB ubb(blocks, pdfFieldGPUID, wallVelocity, real_t(0.0), real_t(0.0));
   ubb.fillFromFlagField< FlagField_T >(blocks, flagFieldID, Velocity_Flag, Fluid_Flag);

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

   // setup of the LBM communication for synchronizing the pdf field between neighboring blocks
   gpu::communication::UniformGPUScheme< Stencil_T > com(blocks, 0, false);
   com.addPackInfo(make_shared< PackInfo_T >(pdfFieldGPUID));
   auto communication = std::function< void() >([&]() { com.communicate(nullptr); });

   // create the timeloop
   SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);

   pystencils::PSMSweep PSMSweep(particleAndVolumeFractionSoA.BsFieldID, particleAndVolumeFractionSoA.BFieldID,
                                 particleAndVolumeFractionSoA.particleForcesFieldID,
                                 particleAndVolumeFractionSoA.particleVelocitiesFieldID, pdfFieldGPUID, real_t(0.0),
                                 real_t(0.0), real_t(0.0), lbm::collision_model::omegaFromViscosity(viscosity));

   timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");

   pystencils::PSM_MacroGetter getterSweep(densityFieldID, pdfFieldID, velFieldID, real_t(0.0), real_t(0.0),
                                           real_t(0.0));
   if (vtkIOFreq != uint_t(0))
   {
      // spheres
      auto particleVtkOutput = make_shared< mesa_pd::vtk::ParticleVtkOutput >(ps);
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleInteractionRadius >("radius");
      particleVtkOutput->addOutput< mesa_pd::data::SelectParticleLinearVelocity >("velocity");
      auto particleVtkWriter =
         vtk::createVTKOutput_PointData(particleVtkOutput, "Particles", vtkIOFreq, baseFolderVTK, "simulation_step");
      timeloop.addFuncBeforeTimeStep(vtk::writeFiles(particleVtkWriter), "VTK (sphere data)");

      // pdf field
      auto pdfFieldVTK =
         vtk::createVTKOutput_BlockData(blocks, "fluid_field", vtkIOFreq, uint_t(0), false, baseFolderVTK);

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
                  << Sweep(deviceSyncWrapper(ubb.getSweep()), "Boundary Handling (UBB)");
   timeloop.add() << Sweep(deviceSyncWrapper(noSlip.getSweep()), "Boundary Handling (NoSlip)");

   // stream + collide LBM step
   addPSMSweepsToTimeloop(timeloop, psmSweepCollection, PSMSweep);

   // add force evaluation and logging
   real_t normalizationFactor =
      math::pi / real_t(8) * densityFluid * shearRate * shearRate * wallDistance * wallDistance * diameter * diameter;
   std::string loggingFileName(baseFolderLogging + "/LoggingForcesNearPlane_");
   std::string executableName = argv[0];
   size_t lastSlash           = executableName.find_last_of("/\\");
   if (lastSlash != std::string::npos) { loggingFileName += executableName.substr(lastSlash + 1); }
   else { loggingFileName += executableName; }
   loggingFileName += "_D" + std::to_string(uint_c(diameter));
   loggingFileName += "_Re" + std::to_string(uint_c(ReynoldsNumberShear));
   loggingFileName += "_WD" + std::to_string(uint_c(normalizedWallDistance * real_t(1000)));
   loggingFileName += ".txt";
   WALBERLA_LOG_INFO_ON_ROOT(" - writing logging file " << loggingFileName);
   SpherePropertyLogger< ParticleAccessor_T > logger(accessor, sphereUid, loggingFileName, fileIO, normalizationFactor,
                                                     normalizationFactor, physicalTimeScale);

   // compute reference values from literature

   const real_t normalizedGapSize = normalizedWallDistance - real_t(0.5);

   // drag correlation for the drag coefficient
   const real_t standardDragCorrelation =
      real_t(24) / ReynoldsNumberShear *
      (real_t(1) + real_t(0.15) * std::pow(ReynoldsNumberShear, real_t(0.687))); // Schiller-Naumann correlation
   const real_t dragCorrelationWithGapSizeStokes =
      real_t(24) / ReynoldsNumberShear *
      (real_t(1) + real_t(0.138) * std::exp(real_t(-2) * normalizedGapSize) +
       real_t(9) / (real_t(16) * (real_t(1) + real_t(2) * normalizedGapSize))); // Goldman et al. (1967)
   const real_t alphaDragS = real_t(0.15) - real_t(0.046) *
                                               (real_t(1) - real_t(0.16) * normalizedGapSize * normalizedGapSize) *
                                               std::exp(-real_t(0.7) * normalizedGapSize);
   const real_t betaDragS = real_t(0.687) + real_t(0.066) *
                                               (real_t(1) - real_t(0.76) * normalizedGapSize * normalizedGapSize) *
                                               std::exp(-std::pow(normalizedGapSize, real_t(0.9)));
   const real_t dragCorrelationZeng =
      dragCorrelationWithGapSizeStokes *
      (real_t(1) + alphaDragS * std::pow(ReynoldsNumberShear, betaDragS)); // Zeng et al. (2009) - Eqs. (13) and (14)

   // lift correlations for the lift coefficient
   const real_t liftCorrelationZeroGapStokes = real_t(5.87); // Leighton, Acrivos (1985)
   const real_t liftCorrelationZeroGap =
      real_t(3.663) / std::pow(ReynoldsNumberShear * ReynoldsNumberShear + real_t(0.1173),
                               real_t(0.22)); //  Zeng et al. (2009) - Eq. (19)
   const real_t alphaLiftS = -std::exp(-real_t(0.3) + real_t(0.025) * ReynoldsNumberShear);
   const real_t betaLiftS  = real_t(0.8) + real_t(0.01) * ReynoldsNumberShear;
   const real_t lambdaLiftS =
      (real_t(1) - std::exp(-normalizedGapSize)) * std::pow(ReynoldsNumberShear / real_t(250), real_t(5) / real_t(2));
   const real_t liftCorrelationZeng =
      liftCorrelationZeroGap *
      std::exp(-real_t(0.5) * normalizedGapSize * std::pow(ReynoldsNumberShear / real_t(250), real_t(4) / real_t(3))) *
      (std::exp(alphaLiftS * std::pow(normalizedGapSize, betaLiftS)) -
       lambdaLiftS); // Zeng et al. (2009) - Eqs. (28) and (29)

   ////////////////////////
   // EXECUTE SIMULATION //
   ////////////////////////

   WcTimingPool timeloopTiming;

   const uint_t checkingFrequency = uint_c(physicalCheckingFrequency * physicalTimeScale);

   WALBERLA_LOG_INFO_ON_ROOT("Starting simulation with at maximum of " << timesteps << " time steps");
   WALBERLA_LOG_INFO_ON_ROOT("Convergence checking frequency = "
                             << checkingFrequency << " (physically = " << physicalCheckingFrequency
                             << ") until relative difference of " << relativeChangeConvergenceEps << " is reached.");

   real_t maxDragCurrentCheckingPeriod = -math::Limits< real_t >::inf();
   real_t minDragCurrentCheckingPeriod = math::Limits< real_t >::inf();
   real_t maxLiftCurrentCheckingPeriod = -math::Limits< real_t >::inf();
   real_t minLiftCurrentCheckingPeriod = math::Limits< real_t >::inf();
   uint_t timestep                     = 0;

   // time loop
   while (true)
   {
      // perform a single simulation step
      timeloop.singleStep(timeloopTiming);

      // sync hydrodynamic force to local particle
      reduceProperty.operator()< mesa_pd::HydrodynamicForceTorqueNotification >(*ps);

      // average force
      if (timestep == 0)
      {
         lbm_mesapd_coupling::InitializeHydrodynamicForceTorqueForAveragingKernel
            initializeHydrodynamicForceTorqueForAveragingKernel;
         ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor,
                             initializeHydrodynamicForceTorqueForAveragingKernel, *accessor);
      }
      ps->forEachParticle(false, mesa_pd::kernel::SelectLocal(), *accessor, averageHydrodynamicForceTorque, *accessor);

      // evaluation
      timeloopTiming["Logging"].start();
      logger(timestep);
      timeloopTiming["Logging"].end();

      // reset after logging here
      ps->forEachParticle(false, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor);

      // check for termination
      real_t curDrag = logger.getDragCoefficient();
      real_t curLift = logger.getLiftCoefficient();

      if (std::isinf(curDrag) || std::isnan(curDrag))
         WALBERLA_ABORT("Found invalid drag value " << curDrag << " in time step " << timestep);

      maxDragCurrentCheckingPeriod = std::max(maxDragCurrentCheckingPeriod, curDrag);
      minDragCurrentCheckingPeriod = std::min(minDragCurrentCheckingPeriod, curDrag);
      maxLiftCurrentCheckingPeriod = std::max(maxLiftCurrentCheckingPeriod, curLift);
      minLiftCurrentCheckingPeriod = std::min(minLiftCurrentCheckingPeriod, curLift);

      real_t dragDiffCurrentCheckingPeriod = std::fabs(maxDragCurrentCheckingPeriod - minDragCurrentCheckingPeriod) /
                                             std::fabs(maxDragCurrentCheckingPeriod);
      real_t liftDiffCurrentCheckingPeriod = std::fabs(maxLiftCurrentCheckingPeriod - minLiftCurrentCheckingPeriod) /
                                             std::fabs(maxLiftCurrentCheckingPeriod);

      // continuous output during simulation
      if (timestep % (checkingFrequency * uint_t(10)) == 0)
      {
         WALBERLA_LOG_INFO_ON_ROOT("Drag: current C_D = " << curDrag);
         WALBERLA_LOG_INFO_ON_ROOT(" - standard C_D = " << standardDragCorrelation);
         WALBERLA_LOG_INFO_ON_ROOT(" - C_D ( Stokes fit ) = " << dragCorrelationWithGapSizeStokes);
         WALBERLA_LOG_INFO_ON_ROOT(" - C_D ( Zeng ) = " << dragCorrelationZeng);

         WALBERLA_LOG_INFO_ON_ROOT("Lift: current C_L = " << curLift);
         WALBERLA_LOG_INFO_ON_ROOT(" - C_L ( Stokes, zero gap ) = " << liftCorrelationZeroGapStokes);
         WALBERLA_LOG_INFO_ON_ROOT(" - C_L ( zero gap ) = " << liftCorrelationZeroGap);
         WALBERLA_LOG_INFO_ON_ROOT(" - C_L ( Zeng ) = " << liftCorrelationZeng);

         WALBERLA_LOG_INFO_ON_ROOT("Drag difference [(max-min)/max] = " << dragDiffCurrentCheckingPeriod
                                                                        << ", lift difference = "
                                                                        << liftDiffCurrentCheckingPeriod);
      }

      // check for convergence ( = difference between min and max values in current checking period is below limit)
      if (timestep % checkingFrequency == 0 && timestep > 0)
      {
         if (dragDiffCurrentCheckingPeriod < relativeChangeConvergenceEps &&
             liftDiffCurrentCheckingPeriod < relativeChangeConvergenceEps)
         {
            WALBERLA_LOG_INFO_ON_ROOT("Forces converged with an eps of " << relativeChangeConvergenceEps);
            WALBERLA_LOG_INFO_ON_ROOT(" - drag min = " << minDragCurrentCheckingPeriod
                                                       << " , max = " << maxDragCurrentCheckingPeriod);
            WALBERLA_LOG_INFO_ON_ROOT(" - lift min = " << minLiftCurrentCheckingPeriod
                                                       << " , max = " << maxLiftCurrentCheckingPeriod);
            break;
         }

         // reset min and max values for new checking period
         maxDragCurrentCheckingPeriod = -math::Limits< real_t >::inf();
         minDragCurrentCheckingPeriod = math::Limits< real_t >::inf();
         maxLiftCurrentCheckingPeriod = -math::Limits< real_t >::inf();
         minLiftCurrentCheckingPeriod = math::Limits< real_t >::inf();
      }
      ++timestep;
   }

   timeloopTiming.logResultOnRoot();

   std::string resultFileName(baseFolderLogging + "/ResultForcesNearPlane");
   resultFileName += "_D" + std::to_string(uint_c(diameter));
   resultFileName += "_Re" + std::to_string(uint_c(ReynoldsNumberShear));
   resultFileName += "_WD" + std::to_string(uint_c(normalizedWallDistance * real_t(1000)));
   resultFileName += ".txt";

   WALBERLA_LOG_INFO_ON_ROOT(" - writing final result to file " << resultFileName);
   logFinalResult(resultFileName, ReynoldsNumberShear, normalizedWallDistance, diameter, domainSize[0], domainSize[1],
                  domainSize[2], logger.getDragCoefficient(), dragCorrelationZeng, logger.getLiftCoefficient(),
                  liftCorrelationZeng, timestep, real_c(timestep) / physicalTimeScale);

   return EXIT_SUCCESS;
}

} // namespace forces_on_sphere_near_plane

int main(int argc, char** argv) { forces_on_sphere_near_plane::main(argc, argv); }
