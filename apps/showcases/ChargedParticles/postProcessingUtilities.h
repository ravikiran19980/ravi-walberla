//
// Created by avnss on 7/13/2023.
//

#ifndef WALBERLA_POSTPROCESSINGUTILITIES_H
#define WALBERLA_POSTPROCESSINGUTILITIES_H

#pragma once
#include <vector>
#include <math.h>
#include "core/DataTypes.h"
#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm/field/AddToStorage.h"
//#include <filesystem>
#include <core/mpi/MPITextFile.h>

namespace walberla {

// class to compute the volume fractions average at each unique height along with xy spatial plane
template< typename BlockStorage_T, typename FlagField_T >
class computeSolidVolumeFraction
{
 public:
   computeSolidVolumeFraction(const shared_ptr< BlockStorage_T >& blocks, const BlockDataID& flagFieldID,
                              const BlockDataID& particleAndVolumeFractionFieldID,
                              const math::GenericAABB< real_t > simulationDomain)
      : blocks_(blocks), flagFieldID_(flagFieldID), particleAndVolumeFractionFieldID_(particleAndVolumeFractionFieldID),
        simulationDomain_(simulationDomain)

   {}

   void operator()(uint_t currentTimeStep)
   {
      for (auto blockIt = blocks_->begin(); blockIt != blocks_->end(); ++blockIt)
      {
         lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T* particleAndVolumeFractionField =
            blockIt->template getData< lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T >(
               particleAndVolumeFractionFieldID_);

         const FlagField_T* const flagField = blockIt->getData< const FlagField_T >(flagFieldID_);

         CellInterval xyz = flagField->xyzSize();
         Cell globalCell;
         z_SolidVolumeFraction_ = std::vector< real_t >(uint_c(simulationDomain_.zSize()), real_t(0));

         for (auto cell = xyz.begin(); cell != xyz.end(); ++cell)
         {
            blocks_->transformBlockLocalToGlobalCell(globalCell, *blockIt, *cell);
            for (auto& e : particleAndVolumeFractionField->get(cell->x(), cell->y(), cell->z()))
            {
               z_SolidVolumeFraction_[uint_c(globalCell.z())] += e.second;
            }
         }
      }
      mpi::allReduceInplace(z_SolidVolumeFraction_, mpi::SUM);
      for (uint_t i = 0; i < uint_c(simulationDomain_.zSize()); ++i)
      {
         z_SolidVolumeFraction_[i] /=
            simulationDomain_.xSize() * simulationDomain_.ySize(); // average for a unique height
      }

      std::string filename = "SolidVolumeFraction_" + std::to_string(currentTimeStep) + ".txt";
      printToSeparateFile(filename);
   }

 private:
   shared_ptr< StructuredBlockStorage > blocks_;
   const BlockDataID flagFieldID_;
   const BlockDataID particleAndVolumeFractionFieldID_;
   math::GenericAABB< real_t > simulationDomain_;
   std::vector< real_t > z_SolidVolumeFraction_;

   void printToSeparateFile(std::string filename)
   {
      WALBERLA_ROOT_SECTION()
      {
         const std::string directoryName = "FractionData";

         // Create the directory if it doesn't exist
         if (!std::filesystem::exists(directoryName)) { std::filesystem::create_directory(directoryName); }
         const std::string filePath = directoryName + "/" + filename;

         // Open the file for writing
         std::ofstream outputFile(filePath);
         if (!outputFile.is_open())
         {
            WALBERLA_LOG_INFO("Error: Unable to open file " << filename << " for writing.");
            return;
         }

         // Write the contents of z_SolidVolumeFraction_ to the file
         outputFile << "z"
                    << ","
                    << "values" << '\n';
         for (uint_t i = 0; i < uint_c(simulationDomain_.zSize()); ++i)
         {
            outputFile << i << "," << z_SolidVolumeFraction_[i] << '\n';
         }

         // Close the file
         outputFile.close();
      }
   }
};

// class to compute and write the particle x,y,z velocities and the unique ids into a file

template<typename ParticleAccessor_T >
void computeParticleProperties(const shared_ptr< ParticleAccessor_T >& ac,
                               uint_t currentTimeStep)
{
   std::string filename            = "ParticleProperties_" + std::to_string(currentTimeStep) + ".txt";
   const std::string directoryName = "ParticleData";

   // Create the directory if it doesn't exist
   if (!std::filesystem::exists(directoryName)) { std::filesystem::create_directory(directoryName); }
   const std::string filePath = directoryName + "/" + filename;

   // Open the file for writing using std::ostringstream
   std::ostringstream outputFile;

   WALBERLA_ROOT_SECTION()
   {

      outputFile << "Uid"
                 << ","
                 << "Velocity_X"
                 << ","
                 << "Velocity_Y"
                 << ","
                 << "Velocity_Z"
                 << ","
                 << "PositionX"
                 << ","
                 << "PositionY"
                 << ","
                 << "PositionZ" << '\n';
   }

   for (uint_t i = 0; i < ac->size(); ++i)
   {
      if (isSet(ac->getFlags(i), walberla::mesa_pd::data::particle_flags::GHOST)) continue;
      if (isSet(ac->getFlags(i), walberla::mesa_pd::data::particle_flags::GLOBAL)) continue;
      const size_t uid = (ac->getUid(i));
      real_t velocityX = ac->getLinearVelocity(i)[0];
      real_t velocityY = ac->getLinearVelocity(i)[1];
      real_t velocityZ = ac->getLinearVelocity(i)[2];
      real_t positionX = ac->getPosition(i)[0];
      real_t positionY = ac->getPosition(i)[1];
      real_t positionZ = ac->getPosition(i)[2];

      outputFile << uid << "," << velocityX << "," << velocityY << "," << velocityZ << "," << positionX << ","
                 << positionY << "," << positionZ << '\n';
   }
   walberla::mpi::writeMPITextFile(filePath, outputFile.str());
}


// class to compute and write the particle x,y,z velocities and the unique ids into a file

template<typename ParticleAccessor_T >
void computeParticleStresses(const shared_ptr< ParticleAccessor_T >& ac,
                             std::vector< real_t >& hydroForceGlobal, std::vector< real_t >& collisionForceGlobal,
                             std::vector< real_t >& binCount, Vector3<real_t> gravitationForce)
{
   real_t diameter = 20;


   for (uint_t i = 0; i < ac->size(); ++i)
   {
      if (isSet(ac->getFlags(i), walberla::mesa_pd::data::particle_flags::GHOST)) continue;
      if (isSet(ac->getFlags(i), walberla::mesa_pd::data::particle_flags::GLOBAL)) continue;

      real_t hydroLubricationForce = (ac->getHydrodynamicForce(i) + gravitationForce).length();   // F'h = Fh - Vp(rhof-rhop)*g
      real_t collisionForce        = (ac->getForce(i) - (ac->getHydrodynamicForce(i) + gravitationForce)).length(); // Total_force - (F'h)

      uint_t bin_index = uint_c((ac->getPosition(i)[2]) / (diameter/2));
      binCount[bin_index] +=1;
      hydroForceGlobal[bin_index] += hydroLubricationForce;
      collisionForceGlobal[bin_index] += collisionForce;

   }

}

void writeStressesToFile(std::vector<real_t>& hydroStress, std::vector<real_t>& collisionStress){


   WALBERLA_ROOT_SECTION()
   {
      std::string filename            = "ParticleStresses.txt";
      const std::string directoryName = ".";

      // Create the directory if it doesn't exist
      if (!std::filesystem::exists(directoryName)) { std::filesystem::create_directory(directoryName); }
      const std::string filePath = directoryName + "/" + filename;

      // Open the file for writing using std::ostringstream
      std::ofstream outputFile(filePath);

      outputFile << "Uid"
                 << ","
                 << "Collision_force"
                 << ","
                 << "Hydro_lub_force" << '\n';

      for (size_t i = 0; i < hydroStress.size(); i++) {
         outputFile << i << "," << collisionStress[i] << "," << hydroStress[i] << '\n';
      }
      outputFile.close();
   }
}

}




#endif //WALBERLA_POSTPROCESSINGUTILITIES_H
