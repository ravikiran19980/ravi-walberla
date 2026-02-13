//
// Created by dy94rovu on 2/9/26.
//


#pragma once

#include "core/math/all.h"
#include "core/DataTypes.h"
#include "GeneralInfoHeader.h"

namespace MaterialTransport
{
using namespace walberla;

class WallNormalHeatFlux
{
 public:
   WallNormalHeatFlux(uint_t Ny, real_t dy, real_t alpha_f, real_t alpha_p,
                      const std::string& wallNormalHeatFluxFileName = "meanFlux.txt")
      : Ny_(Ny), dy_(dy), alpha_f_(alpha_f), alpha_p_(alpha_p)
   {
      WALBERLA_ROOT_SECTION()
      {
         outFile_.open(wallNormalHeatFluxFileName, std::ios::out);
         outFile_ << "timeStep meanWallNormaHeatFlux" << std::endl;
      }
   }

   ~WallNormalHeatFlux()
   {
      WALBERLA_ROOT_SECTION() { outFile_.close(); };
   };

   real_t computeMeanWallNormalFlux(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& tempFieldID,
                                    const BlockDataID& BFieldID)
   {
      double qbot = 0.0, qtop = 0.0;
      double nbot = 0.0, ntop = 0.0;

      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block = *blockIt;
         auto Tfield = block.getData< DensityField_concentration_T >(tempFieldID);
         auto Bfield = block.getData< GhostLayerField< real_t, 1 > >(BFieldID);

         WALBERLA_FOR_ALL_CELLS_XYZ(
            Tfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));

            const uint_t j = uint_c(cell.z()); const real_t B = Bfield->get(x, y, z);

            if (j == 0) {
               const real_t dTdy =
                  (-3 * Tfield->get(x, y, z) + 4 * Tfield->get(x, y, z + 1) - Tfield->get(x, y, z + 2)) / (2 * dy_);

               qbot += ((1 - B) * alpha_f_ + B * alpha_p_) * dTdy;
               nbot += 1.0;
            } else if (j == Ny_ - 1) {
               const real_t dTdy =
                  (3 * Tfield->get(x, y, z) - 4 * Tfield->get(x, y, z - 1) + Tfield->get(x, y, z - 2)) / (2 * dy_);

               qtop += ((1 - B) * alpha_f_ + B * alpha_p_) * dTdy;
               ntop += 1.0;
            })
      }

      mpi::allReduceInplace(qbot, mpi::SUM);
      mpi::allReduceInplace(qtop, mpi::SUM);
      mpi::allReduceInplace(nbot, mpi::SUM);
      mpi::allReduceInplace(ntop, mpi::SUM);

      return 0.5 * (qbot / nbot + qtop / ntop);
   }

   void RunningMeanHeatFluxOutput(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& tempFieldID,
                                  const BlockDataID& BFieldID, uint_t timeStep, uint_t outputFrequency)
   {
      const real_t currentflux = computeMeanWallNormalFlux(blocks, tempFieldID, BFieldID);
      heatFlux_                = currentflux;
      // update running mean
      ++sampleCount_;
      runningMeanFlux_ += (currentflux - runningMeanFlux_) / real_t(sampleCount_);

      if (timeStep % outputFrequency == 0)
      {
         WALBERLA_ROOT_SECTION() { outFile_ << timeStep << " " << runningMeanFlux_ << std::endl; }
      }
   }

   void checkForConvergence(real_t tolerance, uint_t timeStep, uint_t timeBlockSize)
   {
      if (timeStep % timeBlockSize == 0 && timeStep > 0)
      {
         const real_t relDiff_percentage =
            std::abs(runningMeanFlux_ - previousMeanFlux_) * 100 / std::max(std::abs(previousMeanFlux_), real_t(1e-14));

         previousMeanFlux_ = runningMeanFlux_;

         if (relDiff_percentage < tolerance)
            ++convergenceCounter_;
         else
            convergenceCounter_ = 0;
         WALBERLA_LOG_INFO_ON_ROOT("rel perentage difference from  " <<  timeStep - timeBlockSize  << "to  " << timeStep << " is " << relDiff_percentage);
         WALBERLA_LOG_INFO_ON_ROOT("convergence counter is  " << convergenceCounter_);
         resetRunnningMeanFlux();
      }
   }

   bool convergenceStatus(uint_t requiredSampleBlocks = 3)
   {
      if (convergenceCounter_ >= requiredSampleBlocks) { return true; }
      else
      {
         return false;
      }
   }

   void resetRunnningMeanFlux(){
      runningMeanFlux_ = real_t(0);
      sampleCount_ = uint_t(0);
   }


 private:
   uint_t Ny_;
   real_t dy_, alpha_f_, alpha_p_;

   uint_t sampleCount_ = 0;
   real_t runningMeanFlux_ = real_t(0);
   real_t heatFlux_ = real_t(0);
   real_t previousMeanFlux_ = real_t(0);
   uint_t convergenceCounter_ = 0;
   std::ofstream outFile_;
};


class HeatFluxBudgets
{

 public:

   HeatFluxBudgets(const uint_t Nz, const real_t dz, const real_t alpha_f, const real_t alpha_p, uint_t &timeStep)
      : Nz_(Nz), dz_(dz), alpha_f_(alpha_f), alpha_p_(alpha_p), Tparticle_plane_(Nz_, 0), Vparticle_plane_(Nz_, 0),
        Tfluid_plane_(Nz_, 0), Vfluid_plane_(Nz_, 0), TVparticle_plane_(Nz_, 0), TVfluid_plane_(Nz_, 0),
        dTparticle_plane_(Nz_, 0), dTfluid_plane_(Nz_, 0), phi_p_(Nz_, 0), phi_f_(Nz_, 0), cell_count_plane_(Nz_, 0)

   {}

   // sample one simulation step (call after macros/temperature are updated & halo exchanged)
   void operator()(const shared_ptr< StructuredBlockStorage >& blocks,
                     const BlockDataID& velFieldFluidID,
                     const BlockDataID& tempFieldID,
                     const BlockDataID& BFieldID
                     )
   {
      resetPlanes_();

      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block = *blockIt;
         auto velF   = block.getData< VelocityField_fluid_T >(velFieldFluidID);
         auto Tfield = block.getData< DensityField_concentration_T >(tempFieldID);
         auto Bfield = block.getData< GhostLayerField< real_t, 1 > >(BFieldID);

         WALBERLA_FOR_ALL_CELLS_XYZ(

            Tfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));

            const uint_t j = uint_c(cell.z()); // global value is stored in j
            const real_t B = Bfield->get(x, y, z); const real_t temperature = Tfield->get(x, y, z);
            const real_t velocity_z = velF->get(x, y, z, 2); Tparticle_plane_[j] += (B) *temperature;
            Tfluid_plane_[j] += (1 - B) * temperature;

            TVparticle_plane_[j] += (B) *temperature * (B) *velocity_z;
            Tfluid_plane_[j] += (1 - B) * temperature * (1 - B) * velocity_z;

            phi_p_[j] += B; phi_f_[j] += (1 - B);

            cell_count_plane_[j] += 1.0;

            // for the conductive part of particles and fluid
            real_t dTdy_p = 0.0;
            real_t dTdy_f = 0.0;
            if (j > 0 && j < Nz_ - 1) {
               // for the INNER mpi blocks of the domain but the FIRST or the LAST cells use one sided FD
               // for the INNER mpi blocks of the domain and all middle cells use central FD

               if (z == 0)
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z + 1);
                  const real_t T2_p = B * Tfield->get(x, y, z + 2);
                  dTdy_p            = (-3.0 * T0_p + 4.0 * T1_p - T2_p) / (2.0 * dz_);
                  dTdy_p            = alpha_p_ * dTdy_p;
                  dTparticle_plane_[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z + 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z + 2);
                  dTdy_f            = (-3.0 * T0_f + 4.0 * T1_f - T2_f) / (2.0 * dz_);
                  dTdy_f            = alpha_f_ * dTdy_f;
                  dTfluid_plane_[j] += dTdy_f;
               }
               else if (z == cell_idx_c(blocks->getNumberOfZCells(block) - 1))
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z - 1);
                  const real_t T2_p = B * Tfield->get(x, y, z - 2);
                  dTdy_p            = (3.0 * T0_p - 4.0 * T1_p + T2_p) / (2.0 * dz_);
                  dTdy_p            = alpha_p_ * dTdy_p;
                  dTparticle_plane_[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z - 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z - 2);
                  dTdy_f            = (3.0 * T0_f - 4.0 * T1_f + T2_f) / (2.0 * dz_);
                  dTdy_f            = alpha_f_ * dTdy_f;
                  dTfluid_plane_[j] += dTdy_f;
               }
               else
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z - 1);
                  const real_t T1_p = B * Tfield->get(x, y, z + 1);
                  dTdy_p            = (T1_p - T0_p) / (2.0 * dz_);
                  dTdy_p            = alpha_p_ * dTdy_p;
                  dTparticle_plane_[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = B * Tfield->get(x, y, z - 1);
                  const real_t T1_f = B * Tfield->get(x, y, z + 1);
                  dTdy_f            = (T1_f - T0_f) / (2.0 * dz_);
                  dTdy_f            = alpha_f_ * dTdy_f;
                  dTfluid_plane_[j] += dTdy_f;
               }
            } else {
               // wall flux (one-sided derivative), average over x,z on the two extreme planes

               if (j == 0)
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z + 1);
                  const real_t T2_p = B * Tfield->get(x, y, z + 2);
                  dTdy_p            = (-3.0 * T0_p + 4.0 * T1_p - T2_p) / (2.0 * dz_);
                  dTdy_p            = alpha_p_ * dTdy_p;
                  dTparticle_plane_[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z + 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z + 2);
                  dTdy_f            = (-3.0 * T0_f + 4.0 * T1_f - T2_f) / (2.0 * dz_);
                  dTdy_f            = alpha_f_ * dTdy_f;
                  dTfluid_plane_[j] += dTdy_f;
               }
               if (j == Nz_ - 1)
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z - 1);
                  const real_t T2_p = B * Tfield->get(x, y, z - 2);
                  dTdy_p            = (3.0 * T0_p - 4.0 * T1_p + T2_p) / (2.0 * dz_);
                  dTdy_p            = alpha_p_ * dTdy_p;
                  dTparticle_plane_[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z - 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z - 2);
                  dTdy_f            = (3.0 * T0_f - 4.0 * T1_f + T2_f) / (2.0 * dz_);
                  dTdy_f            = alpha_f_ * dTdy_f;
                  dTfluid_plane_[j] += dTdy_f;
               }
            }

         )
      }
      // reductions needed here for all the quantities

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace(Tparticle_plane_, mpi::SUM);
         mpi::allReduceInplace(Vparticle_plane_, mpi::SUM);

         mpi::allReduceInplace(TVparticle_plane_, mpi::SUM);
         mpi::allReduceInplace(TVfluid_plane_, mpi::SUM);

         mpi::allReduceInplace(dTparticle_plane_, mpi::SUM);
         mpi::allReduceInplace(dTfluid_plane_, mpi::SUM);

         mpi::allReduceInplace(phi_p_, mpi::SUM);
         mpi::allReduceInplace(phi_f_, mpi::SUM);

         mpi::allReduceInplace(cell_count_plane_, mpi::SUM);
      }

      // computing the plane averages here

      for (size_t k = 0; k < cell_count_plane_.size(); ++k)
      {
         if (cell_count_plane_[k] > 0)
         {
            Tparticle_plane_[k] /= cell_count_plane_[k];
            Vparticle_plane_[k] /= cell_count_plane_[k];
            TVparticle_plane_[k] /= cell_count_plane_[k];
            TVfluid_plane_[k] /= cell_count_plane_[k];
            dTparticle_plane_[k] /= cell_count_plane_[k];
            dTfluid_plane_[k] /= cell_count_plane_[k];
            phi_p_[k] /= cell_count_plane_[k];
            phi_f_[k] /= cell_count_plane_[k];
         }
      }
   }




   void writePlaneAverages_(uint_t timeStep) const
   {
      if (mpi::MPIManager::instance()->rank() == 0)
      {
         std::ofstream file;
         file.open("plane_averages_t" + std::to_string(timeStep) + ".dat");

         file << "plane number"
              << "Tparticle Vparticle "
              << "TVparticle TVfluid "
              << "dTparticle dTfluid "
              << "phi_p phi_f cell_count\n";

         file << std::scientific << std::setprecision(8);

         for (size_t k = 0; k < cell_count_plane_.size(); ++k)
         {
            file << k << " " << Tparticle_plane_[k] << " " << Vparticle_plane_[k] << " " << TVparticle_plane_[k] << " "
                 << TVfluid_plane_[k] << " " << dTparticle_plane_[k] << " " << dTfluid_plane_[k] << " " << phi_p_[k] << " "
                 << phi_f_[k] << " " << cell_count_plane_[k] << "\n";
         }

         file.close();
      }
   }

 private:
   uint_t Nz_;
   real_t dz_;

   // constants
   real_t alpha_f_, alpha_p_;                                 // thermal diffusivities (LB units)
   std::vector< double > Tparticle_plane_, Vparticle_plane_; // particle temperature and velocity averages
   std::vector< double > Tfluid_plane_, Vfluid_plane_;       // fluid temperature and velocity averages
   std::vector< double > TVparticle_plane_, TVfluid_plane_;  // particle temperature velocity product averages
   std::vector< double > dTparticle_plane_, dTfluid_plane_;  // fluid temperature velocity product averages
   std::vector< double > phi_p_, phi_f_;                     // fraction fields of particle and fluid
   std::vector< double > cell_count_plane_;


   void resetPlanes_()
   {
      auto resetToZero = [](auto & v)
      {
         std::fill(v.begin(), v.end(), 0.0);
      };

      resetToZero(Tparticle_plane_);
      resetToZero(Vparticle_plane_);
      resetToZero(Tfluid_plane_);
      resetToZero(Vfluid_plane_);
      resetToZero(TVparticle_plane_);
      resetToZero(TVfluid_plane_);
      resetToZero(dTparticle_plane_);
      resetToZero(dTfluid_plane_);
      resetToZero(phi_p_);
      resetToZero(phi_f_);
   }

};



}
