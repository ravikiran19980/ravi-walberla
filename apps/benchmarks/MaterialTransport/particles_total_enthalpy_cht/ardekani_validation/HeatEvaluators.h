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

   HeatFluxBudgets(const uint_t Nz, const real_t dz, const real_t alpha_f, const real_t alpha_p)
      : Nz_(Nz), dz_(dz), alpha_f_(alpha_f), alpha_p_(alpha_p), particleFluctuation_plane_(Nz_, 0),
        fluidFluctuation_plane_(Nz_, 0), dTparticle_plane_(Nz_, 0), dTfluid_plane_(Nz_, 0), phi_p_(Nz_, 0),
        phi_f_(Nz_, 0), cell_count_plane_(Nz_, 0),

        particleFluctuation_time_(Nz_, 0), fluidFluctuation_time_(Nz_, 0), dTparticle_time_(Nz_, 0),
        dTfluid_time_(Nz_, 0), phi_p_time(Nz_, 0), phi_f_time(Nz_, 0)

   {}

   // sample one simulation step (call after macros/temperature are updated & halo exchanged)
   void operator()(const shared_ptr< StructuredBlockStorage >& blocks,
                     const BlockDataID& velFieldFluidID,
                     const BlockDataID& tempFieldID,
                     const BlockDataID& BFieldID
                     )
   {
      resetFluxPlanes_();

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
            const real_t velocity_z = velF->get(x, y, z, 2);

            const real_t Tp_dash = (B) *temperature - temperature_particle_plane_avg_[j];
            const real_t Vp_dash = B*velocity_z - velocity_particle_plane_avg_[j];

            const real_t Tf_dash = (1-B) *temperature - temperature_fluid_plane_avg_[j];
            const real_t Vf_dash = (1-B) *velocity_z - velocity_fluid_plane_avg_[j];

            particleFluctuation_plane_[j] += Tp_dash*Vp_dash;
            fluidFluctuation_plane_[j]    += Tf_dash*Vf_dash;

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
         mpi::allReduceInplace(particleFluctuation_plane_, mpi::SUM);
         mpi::allReduceInplace(fluidFluctuation_plane_, mpi::SUM);


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
            particleFluctuation_plane_[k] /= cell_count_plane_[k];
            fluidFluctuation_plane_[k] /= cell_count_plane_[k];

            dTparticle_plane_[k] /= cell_count_plane_[k];
            dTfluid_plane_[k] /= cell_count_plane_[k];

            phi_p_[k] /= cell_count_plane_[k];
            phi_f_[k] /= cell_count_plane_[k];
         }
      }

      // time averaging of the x-y plane or spatial averages averages
      timeSamples_ ++;
      for (uint_t k = 0; k < Nz_; ++k)
      {
         particleFluctuation_time_[k] += (particleFluctuation_plane_[k] - particleFluctuation_time_[k]) / timeSamples_;
         fluidFluctuation_time_[k] += (fluidFluctuation_plane_[k] - fluidFluctuation_time_[k]) / timeSamples_;

         dTparticle_time_[k] += (dTparticle_plane_[k] - dTparticle_time_[k]) / timeSamples_;
         dTfluid_time_[k] += (dTfluid_plane_[k] - dTfluid_time_[k]) / timeSamples_;

         phi_p_time[k] += (phi_p_[k] - phi_p_time[k]) / timeSamples_;
         phi_f_time[k] += (phi_f_[k] - phi_f_time[k]) / timeSamples_;

      }
   }




 private:
   uint_t Nz_;
   real_t dz_;
   uint_t timeSamples_ = 0;

   // constants
   real_t alpha_f_, alpha_p_;                                   // thermal diffusivities (LB units)
   std::vector< double > particleFluctuation_plane_ ;          // particle temperature and velocity averages
   std::vector< double > fluidFluctuation_plane_;             // fluid temperature and velocity averages
   std::vector< double > dTparticle_plane_, dTfluid_plane_;  // fluid temperature velocity product averages
   std::vector< double > phi_p_, phi_f_;                     // fraction fields of particle and fluid
   std::vector< double > cell_count_plane_;

   std::vector<double>    particleFluctuation_time_;
   std::vector< double > fluidFluctuation_time_;
   std::vector< double > dTparticle_time_, dTfluid_time_;
   std::vector< double > phi_p_time, phi_f_time;



   std::vector<real_t>  velocity_fluid_plane_avg_;
   std::vector<real_t>  temperature_fluid_plane_avg_;

   std::vector<real_t>  velocity_particle_plane_avg_;
   std::vector<real_t>  temperature_particle_plane_avg_;

   void resetFluxPlanes_()
   {
      auto resetToZero = [](auto & v)
      {
         std::fill(v.begin(), v.end(), 0.0);
      };

      resetToZero(particleFluctuation_plane_);
      resetToZero(fluidFluctuation_plane_);

      resetToZero(dTparticle_plane_);
      resetToZero(dTfluid_plane_);

      resetToZero(phi_p_);
      resetToZero(phi_f_);
   }

   void setPlaneAverages(const std::vector<real_t>& velFluid,const std::vector<real_t>& velParticle,
                     const std::vector<real_t>& tempFluid, const std::vector<real_t>& tempParticle)
   {
      velocity_fluid_plane_avg_ = velFluid;
      velocity_particle_plane_avg_ = velParticle;

      temperature_fluid_plane_avg_ = tempFluid;
      temperature_particle_plane_avg_ = tempParticle;
   }

};


class MeanPlaneAverager
{
public:
    MeanPlaneAverager(uint_t Nz)
     : Nz_(Nz), velocityFluid_plane_(Nz_, 0.0), temperatureFluid_plane_(Nz_, 0.0), velocityParticle_plane_(Nz_, 0.0),
       temperatureParticle_plane_(Nz_, 0.0), numPlaneCells_(Nz_, 0), velocityFluid_time_(Nz, 0.0),
       temperatureFluid_time_(Nz, 0.0), velocityParticle_time_(Nz, 0.0), temperatureParticle_time_(Nz, 0.0),
       timeCount_(0)

    {}

    void operator()(const shared_ptr< StructuredBlockStorage >& blocks,
                    const BlockDataID& velFieldFluidID,
                    const BlockDataID& tempFieldID,
                    const BlockDataID& BFieldID)
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
                Tfield,
                Cell cell;
                blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x,y,z));
                const uint_t j = uint_c(cell.z());

                velocityFluid_plane_[j]    += (1 - Bfield->get(x,y,z)) * velF->get(x,y,z,2);
                temperatureFluid_plane_[j] += (1 - Bfield->get(x,y,z)) * Tfield->get(x,y,z);

                velocityParticle_plane_[j]    += (Bfield->get(x,y,z)) * velF->get(x,y,z,2);
                temperatureParticle_plane_[j] += (Bfield->get(x,y,z)) * Tfield->get(x,y,z);
                numPlaneCells_[j]   += 1;
            )
        }

        WALBERLA_MPI_SECTION()
        {
           mpi::allReduceInplace(velocityFluid_plane_, mpi::SUM);
           mpi::allReduceInplace(temperatureFluid_plane_, mpi::SUM);

           mpi::allReduceInplace(velocityParticle_plane_, mpi::SUM);
           mpi::allReduceInplace(temperatureParticle_plane_, mpi::SUM);
           mpi::allReduceInplace(numPlaneCells_, mpi::SUM);
        }

        for (uint_t k = 0; k < Nz_; ++k)
        {
           if (numPlaneCells_[k] > 0)
           {
              velocityFluid_plane_[k] /= numPlaneCells_[k];
              temperatureFluid_plane_[k] /= numPlaneCells_[k];

              velocityParticle_plane_[k] /= numPlaneCells_[k];
              temperatureParticle_plane_[k] /= numPlaneCells_[k];
           }
        }

        // ---- running time average ----
        timeCount_++;

        for (uint_t k = 0; k < Nz_; ++k)
        {
           velocityFluid_time_[k] += (velocityFluid_plane_[k] - velocityFluid_time_[k]) / timeCount_;
           temperatureFluid_time_[k] += (temperatureFluid_plane_[k] - temperatureFluid_time_[k]) / timeCount_;

           velocityParticle_time_[k] += (velocityParticle_plane_[k] - velocityParticle_time_[k]) / timeCount_;
           temperatureParticle_time_[k] += (temperatureParticle_plane_[k] - temperatureParticle_time_[k]) / timeCount_;
        }
    }

    const std::vector< real_t >& getFluidSliceVelocityAvg() const { return velocityFluid_time_; }
    const std::vector< real_t >& getFluidSliceTemperatureAvg() const { return temperatureFluid_time_; }

    const std::vector< real_t >& getParticleSliceVelocityAvg() const { return velocityParticle_time_; }
    const std::vector< real_t >& getParticleSliceTemperatureAvg() const { return temperatureParticle_time_; }

private:
    uint_t Nz_;

   std::vector< real_t > velocityFluid_plane_;
   std::vector< real_t > temperatureFluid_plane_;
   std::vector< real_t > velocityParticle_plane_;
   std::vector< real_t > temperatureParticle_plane_;
   std::vector< uint_t > numPlaneCells_;


    uint_t timeCount_;
    std::vector<real_t> velocityFluid_time_;
    std::vector<real_t> temperatureFluid_time_;
   std::vector<real_t> velocityParticle_time_;
   std::vector<real_t> temperatureParticle_time_;

   void resetPlanes_()
   {
      auto resetToZero = [](auto & v)
      {
         std::fill(v.begin(), v.end(), 0.0);
      };

      resetToZero(velocityFluid_plane_);
      resetToZero(temperatureFluid_plane_);

      resetToZero(velocityParticle_plane_);
      resetToZero(temperatureParticle_plane_);

   }
};





}
