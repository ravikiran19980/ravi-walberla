//
// Created by dy94rovu on 2/9/26.
//

#pragma once

#include "core/DataTypes.h"
#include "core/math/all.h"
#include "GeneralInfoHeader.h"

using DensityField_concentration_T = DensityField_temperature_T;
namespace MaterialTransport
{
using namespace walberla;



class MeanPlaneAverager
{
 public:
   MeanPlaneAverager(uint_t Nz)
      : Nz_(Nz), velocity_plane_(Nz_, 0.0), temperature_plane_(Nz_, 0.0), numPlaneCells_(Nz_, 0), velocityFluid_time_(Nz, 0.0),
        temperatureFluid_time_(Nz, 0.0), velocityParticle_time_(Nz, 0.0), temperatureParticle_time_(Nz, 0.0),
        timeCount_(0)

   {}

   void operator()(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& velFieldFluidID,
                   const BlockDataID& tempFieldID, const BlockDataID& BFieldID)
   {
      resetPlanes_();

      // --- spatial averaging ---
      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block = *blockIt;
         auto velF   = block.getData< VelocityField_fluid_T >(velFieldFluidID);
         auto Tfield = block.getData< DensityField_concentration_T >(tempFieldID);
         auto Bfield = block.getData< GhostLayerField< real_t, 1 > >(BFieldID);

         WALBERLA_FOR_ALL_CELLS_XYZ(
            Tfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));
            const uint_t j = uint_c(cell.z());

            velocity_plane_[j] +=   velF->get(x, y, z, 2);
            temperature_plane_[j] +=  Tfield->get(x, y, z);

            numPlaneCells_[j] += 1;)
      }

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace(velocity_plane_, mpi::SUM);
         mpi::allReduceInplace(temperature_plane_, mpi::SUM);
         mpi::allReduceInplace(numPlaneCells_, mpi::SUM);
      }

      for (uint_t k = 0; k < Nz_; ++k)
      {
         if (numPlaneCells_[k] > 0)
         {
            velocity_plane_[k] /= numPlaneCells_[k];
            temperature_plane_[k] /= numPlaneCells_[k];

         }
      }

      // ---- running time average ----
      timeCount_++;

      for (uint_t k = 0; k < Nz_; ++k)
      {
         velocityFluid_time_[k] += (velocity_plane_[k] - velocityFluid_time_[k]) / real_c(timeCount_);
         temperatureFluid_time_[k] += (temperature_plane_[k] - temperatureFluid_time_[k]) / real_c(timeCount_);

      }
   }

   const std::vector< real_t >& getVelocitySliceAvg() const { return velocityFluid_time_; }
   const std::vector< real_t >& getTemperatureSliceAvg() const { return temperatureFluid_time_; }


   const uint_t getTimeCounter() { return timeCount_; }

 private:
   uint_t Nz_;

   std::vector< real_t > velocity_plane_;
   std::vector< real_t > temperature_plane_;

   std::vector< uint_t > numPlaneCells_;

   uint_t timeCount_;
   std::vector< real_t > velocityFluid_time_;
   std::vector< real_t > temperatureFluid_time_;
   std::vector< real_t > velocityParticle_time_;
   std::vector< real_t > temperatureParticle_time_;

   void resetPlanes_()
   {
      auto resetToZero = [](auto& v) { std::fill(v.begin(), v.end(), 0.0); };

      resetToZero(velocity_plane_);
      resetToZero(temperature_plane_);


      resetToZero(numPlaneCells_);
   }
};

class HeatFluxBudgets
{
 public:
   HeatFluxBudgets(const uint_t Nz, const real_t dz, const real_t alpha_f, const real_t alpha_p,const uint_t averagingTimeBlock,
                   MeanPlaneAverager& meanPlaneAverager)
      : Nz_(Nz), dz_(dz), alpha_f_(alpha_f), alpha_p_(alpha_p), averagingTimeBlock_(averagingTimeBlock),
        fluctuation_plane_(Nz_, 0), dT_plane_(Nz_, 0), phi_p_plane(Nz_, 0),
        phi_f_plane(Nz_, 0), cell_count_plane_(Nz_, 0),

        fluctuation_time_(Nz_, 0),dT_time_(Nz_, 0), phi_p_time(Nz_, 0), phi_f_time(Nz_, 0),Qtotal_(Nz_,0),

        meanPlaneAverager_(meanPlaneAverager)
   {
      WALBERLA_ROOT_SECTION()
      {
         outFile_.open("output/heatFluxBudget.txt", std::ios::out);
         outFile_ << "height  fluidConvectiveFlux   particleConvectiveFlux  fluidConductiveFlux  particleConductiveFlux  fluidFraction  particleFraction  totalFlux" << std::endl;
      }
   }

   ~HeatFluxBudgets()
   {
      WALBERLA_ROOT_SECTION() { outFile_.close(); };
   }

   // sample one simulation step (call after macros/temperature are updated & halo exchanged)
   void operator()(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& velFieldFluidID,
                   const BlockDataID& tempFieldID, const BlockDataID& BFieldID)
   {
      resetFluxPlanes_();

      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block = *blockIt;
         auto velF   = block.getData< VelocityField_fluid_T >(velFieldFluidID);
         auto Tfield = block.getData< DensityField_concentration_T >(tempFieldID);
         auto Bfield = block.getData< GhostLayerField< real_t, 1 > >(BFieldID);

         // get the plane averages of velocity and temperature:

         auto velocity_fluid_avg       = meanPlaneAverager_.getVelocitySliceAvg();
         auto temperature_fluid_avg    = meanPlaneAverager_.getTemperatureSliceAvg();

         WALBERLA_FOR_ALL_CELLS_XYZ(

            Tfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));

            const uint_t j = uint_c(cell.z()); // global value is stored in j
            const real_t B = Bfield->get(x, y, z); const real_t temperature = Tfield->get(x, y, z);
            const real_t velocity_z                                         = velF->get(x, y, z, 2);



            const real_t T_dash =  temperature - temperature_fluid_avg[j];
            const real_t V_dash =   velocity_z - velocity_fluid_avg[j];

            fluctuation_plane_[j] += -T_dash * V_dash;

            phi_p_plane[j] += B; phi_f_plane[j] += (1 - B); cell_count_plane_[j] += 1.0;

            // for the conductive part of particles and fluid
            real_t dTdz_p = 0.0;
            real_t dTdz_f = 0.0;
               // for the INNER mpi blocks of the domain but the FIRST or the LAST cells use one sided FD
               // for the INNER mpi blocks of the domain and all middle cells use central FD
            if (j > 0 && j < Nz_ - 1) {
               if (z == 0)
               {
                  // conductive flux
                  const real_t T0_p = Tfield->get(x, y, z);
                  const real_t T1_p = Tfield->get(x, y, z + 1);
                  const real_t T2_p = Tfield->get(x, y, z + 2);
                  dTdz_p            = (-3.0 * T0_p + 4.0 * T1_p - T2_p) / (2.0 * dz_);
                  dTdz_p            = alpha_p_ * dTdz_p;
                  dT_plane_[j] += dTdz_p;
               }
               else if (z == cell_idx_c(blocks->getNumberOfZCells(block) - 1))
               {
                  //conductive flux
                  const real_t T0_p = Tfield->get(x, y, z);
                  const real_t T1_p = Tfield->get(x, y, z - 1);
                  const real_t T2_p = Tfield->get(x, y, z - 2);
                  dTdz_p            = (3.0 * T0_p - 4.0 * T1_p + T2_p) / (2.0 * dz_);
                  dTdz_p            = alpha_p_ * dTdz_p;
                  dT_plane_[j] += dTdz_p;

               }
               else
               {
                  // conductive flux
                  const real_t T0_p = Tfield->get(x, y, z - 1);
                  const real_t T1_p = Tfield->get(x, y, z + 1);
                  dTdz_p            = (T1_p - T0_p) / (2.0 * dz_);
                  dTdz_p            = alpha_p_ * dTdz_p;
                  dT_plane_[j] += dTdz_p;
               }
            } else {
               // wall flux (one-sided derivative), average over x,z on the two extreme planes

               if (j == 0)
               {
                  // conductive flux
                  const real_t T0_p = Tfield->get(x, y, z);
                  const real_t T1_p = Tfield->get(x, y, z + 1);
                  const real_t T2_p = Tfield->get(x, y, z + 2);
                  dTdz_p            = (-3.0 * T0_p + 4.0 * T1_p - T2_p) / (2.0 * dz_);
                  dTdz_p            = alpha_p_ * dTdz_p;
                  dT_plane_[j] += dTdz_p;

               }
               if (j == Nz_ - 1)
               {
                  // conductive flux
                  const real_t T0_p = Tfield->get(x, y, z);
                  const real_t T1_p = Tfield->get(x, y, z - 1);
                  const real_t T2_p = Tfield->get(x, y, z - 2);
                  dTdz_p            = (3.0 * T0_p - 4.0 * T1_p + T2_p) / (2.0 * dz_);
                  dTdz_p            = alpha_p_ * dTdz_p;
                  dT_plane_[j] += dTdz_p;
               }
            }

         )
      }
      // reductions needed here for all the quantities

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace(fluctuation_plane_, mpi::SUM);
         mpi::allReduceInplace(dT_plane_, mpi::SUM);
         mpi::allReduceInplace(phi_p_plane, mpi::SUM);
         mpi::allReduceInplace(phi_f_plane, mpi::SUM);
         mpi::allReduceInplace(cell_count_plane_, mpi::SUM);
      }

      // computing the plane averages here

      for (size_t k = 0; k < cell_count_plane_.size(); ++k)
      {
         if (cell_count_plane_[k] > 0)
         {
            fluctuation_plane_[k] /= cell_count_plane_[k];
            dT_plane_[k] /= cell_count_plane_[k];
            phi_p_plane[k] /= cell_count_plane_[k];
            phi_f_plane[k] /= cell_count_plane_[k];
         }
      }

      // time averaging of the x-y plane or spatial averages averages
      timeSamples_++;
      for (uint_t k = 0; k < Nz_; ++k)
      {

         fluctuation_time_[k] += (fluctuation_plane_[k] - fluctuation_time_[k]) / timeSamples_;
         dT_time_[k] += (dT_plane_[k] - dT_time_[k]) / timeSamples_;
         phi_p_time[k] += (phi_p_plane[k] - phi_p_time[k]) / timeSamples_;
         phi_f_time[k] += (phi_f_plane[k] - phi_f_time[k]) / timeSamples_;
      }

      if (timeSamples_ == averagingTimeBlock_)
      {
         for (uint_t k = 0; k < Nz_; ++k)
         {
            Qtotal_[k] =
               fluctuation_time_[k] + dT_time_[k];
            WALBERLA_ROOT_SECTION()
            {
               outFile_ << k << " " << fluctuation_time_[k]*phi_f_time[k] << " " << fluctuation_time_[k]*phi_p_time[k] << " "
                        << dT_time_[k]*phi_f_time[k] << " " << dT_time_[k]*phi_p_time[k] << " " << phi_f_time[k] << " "
                        << phi_p_time[k] << " " << Qtotal_[k] << std::endl;
            }
         }
      }

   }

   real_t getTimeSampleCount() const { return real_c(timeSamples_); }

 private:
   uint_t Nz_;
   real_t dz_;
   uint_t timeSamples_ = 0;
   std::ofstream outFile_;
   const uint_t averagingTimeBlock_;

   // constants
   real_t alpha_f_, alpha_p_;

   std::vector< double > fluctuation_plane_;
   std::vector< double > dT_plane_ ;
   std::vector< double > phi_p_plane, phi_f_plane;
   std::vector< double > cell_count_plane_;

   std::vector< double > fluctuation_time_;
   std::vector< double > dT_time_;
   std::vector< double > phi_p_time, phi_f_time;

   std::vector< double > Qtotal_;

   MeanPlaneAverager& meanPlaneAverager_;

   void resetFluxPlanes_()
   {
      auto resetToZero = [](auto& v) { std::fill(v.begin(), v.end(), 0.0); };

      resetToZero(fluctuation_plane_);
      resetToZero(dT_plane_);
      resetToZero(phi_p_plane);
      resetToZero(phi_f_plane);

      resetToZero(cell_count_plane_);
   }
};


class WallStatistics
{
 public:
   WallStatistics(uint_t Nz, real_t dz, real_t kinematicViscosity, uint_t outputFrequency,
                     const std::string& nusseltFileName = "output/wallStatistics.txt")
      : Nz_(Nz), dz_(dz), outputFrequency_(outputFrequency), kinematicViscosity_(kinematicViscosity)
   {
      WALBERLA_ROOT_SECTION()
      {
         outFile_.open(nusseltFileName, std::ios::out);
         outFile_ << "timeStep Nu_bottom Nu_top tauW_bottom tauW_top" << std::endl;
      }
   }

   ~WallStatistics()
   {
      WALBERLA_ROOT_SECTION() { outFile_.close(); };
   };

   void operator()(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& tempFieldID,
                   const BlockDataID& velFieldID, uint_t timeStep, uint_t outputFrequency, const real_t T_wall_top,
                   const real_t T_wall_bottom, const real_t tolerance)
   {


      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block = *blockIt;
         auto Tfield = block.getData< DensityField_concentration_T >(tempFieldID);
         auto velF   = block.getData< VelocityField_fluid_T >(velFieldID);

         WALBERLA_FOR_ALL_CELLS_XYZ(
            Tfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));

            const uint_t j = uint_c(cell.z());

            // Bottom wall (j = 0): dT/dz = (-8*T0 + 9*T1 - T2) / (3*dz)  since wall situated at half distance from center of bottom cell
            if (j == 0) {
               const real_t T0   = T_wall_bottom;
               const real_t T1   = Tfield->get(x, y, z);
               const real_t T2   = Tfield->get(x, y, z + 1);
               const real_t dTdz = real_c(Nz_)*(-8.0 * T0 + 9.0 * T1 - T2) / (3.0 * dz_);
               nuBottom_ += dTdz;

               const real_t U0   = real_t(0);
               const real_t U1   = velF->get(x, y, z, 0);
               const real_t U2   = velF->get(x, y, z + 1, 0);
               const real_t dUdz = real_c(Nz_)*(-8.0 * U0 + 9.0 * U1 - U2) / (3.0 * dz_);
               tauBottom_ +=  dUdz;
               countBottom_++;
            }
            // Top wall (j = Nz - 1): dT/dz = (8*T0 - 9*T1 + T2) / (3*dz) since wall situated at half distance from center of top cell
            else if (j == Nz_ - 1) {
               const real_t T0   = T_wall_top;
               const real_t T1   = Tfield->get(x, y, z);
               const real_t T2   = Tfield->get(x, y, z - 1);
               const real_t dTdz = real_c(Nz_)*(8.0 * T0 - 9.0 * T1 + T2) / (3.0 * dz_);
               nuTop_ += dTdz;

               const real_t U0   = real_t(0);
               const real_t U1   = velF->get(x, y, z, 0);
               const real_t U2   = velF->get(x, y, z - 1, 0);
               const real_t dUdz = real_c(Nz_)*(8.0 * U0 - 9.0 * U1 + U2) / (3.0 * dz_);
               tauTop_ +=  dUdz;
               countTop_++;
            })
      }

      // MPI reduction
      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace(nuBottom_, mpi::SUM);
         mpi::allReduceInplace(nuTop_, mpi::SUM);
         mpi::allReduceInplace(tauBottom_, mpi::SUM);
         mpi::allReduceInplace(tauTop_, mpi::SUM);
         mpi::allReduceInplace(countBottom_, mpi::SUM);
         mpi::allReduceInplace(countTop_, mpi::SUM);
      }

      // Average over cells at wall
      if (countBottom_ > 0) nuBottom_ /= real_c(countBottom_);
      if (countTop_ > 0) nuTop_ /= real_c(countTop_);
      if (countBottom_ > 0) tauBottom_ /= real_c(countBottom_);
      if (countTop_ > 0) tauTop_ /= real_c(countTop_);

      if (timeStep % outputFrequency == 0)
      {
         WALBERLA_ROOT_SECTION()
         {
            outFile_ << timeStep << " " << nuBottom_ << " " << nuTop_ << " "
                     << tauBottom_ << " " << tauTop_ << std::endl;
         }
      }

      if (!wallstatistics_convergence_)
      {
         wallStatisticsConvergence(tolerance, timeStep);

      }
      resetWallValues();
   }

   bool getWallStatisticsConvergence() { return wallstatistics_convergence_; }
   real_t getWallShearStress() { return tauBottomOld_; }


 private:

   void resetWallValues() {
      nuBottom_ = 0.0;
      nuTop_ = 0.0;
      tauBottomOld_ = tauBottom_;
      tauBottom_ = 0.0;
      tauTop_ = 0.0;
      countBottom_ = 0;
      countTop_ = 0;
   }

   void wallStatisticsConvergence(real_t tolerance, uint_t timeStep)
   {
      // WALBERLA_LOG_INFO_ON_ROOT("time block size is " << timeBlockSize << "  "<< timeStep%timeBlockSize);
      if (timeStep % outputFrequency_ == 0 && timeStep > 0)
      {
         const real_t currentNu = nuBottom_;
         const real_t relDiff_percentage_nu =
            std::abs(currentNu - oldNu_) * 100 / std::max(std::abs(oldNu_), real_t(1e-14));

         oldNu_ = currentNu;

         const real_t currentTau = tauBottom_;
         const real_t relDiff_percentage_tau =
            std::abs(currentTau - oldTau_) * 100 / std::max(std::abs(oldTau_), real_t(1e-14));

         oldTau_ = currentTau;

         if (relDiff_percentage_nu < tolerance   && relDiff_percentage_tau < tolerance) { ++convergenceCounter_; }
         else
         {
            convergenceCounter_ = 0;
         }

         if (convergenceCounter_ == 20)
         {
            wallstatistics_convergence_ = true;
            WALBERLA_LOG_INFO_ON_ROOT("converged wall stats");
         }
      }
   }



   uint_t Nz_;
   real_t dz_;
   real_t kinematicViscosity_;
   std::ofstream outFile_;
   real_t nuTop_;
   real_t nuBottom_;
   real_t tauTop_;
   real_t tauBottom_;
   real_t tauBottomOld_;
   uint_t countTop_ = 0;
   uint_t countBottom_ = 0;

   uint_t outputFrequency_ = 0;
   real_t oldNu_ = 0.0;
   real_t oldTau_ = 0.0;
   uint_t convergenceCounter_ = 0;
   bool wallstatistics_convergence_ = false;
};








} // namespace MaterialTransport
