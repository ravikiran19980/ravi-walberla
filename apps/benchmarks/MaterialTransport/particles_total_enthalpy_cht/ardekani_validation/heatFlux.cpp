//
// Created by dy94rovu on 9/26/25.
//
#include "core/math/all.h"
#include "core/DataTypes.h"
#include "GeneralInfoHeader.h"

namespace walberla
{
struct HeatFluxAverager
{
   // geometry
   uint_t Nx, Ny, Nz;
   real_t dy;

   // constants
   real_t alpha_f, alpha_p; // thermal diffusivities (LB units)

   // running time-average (plane-wise)

   std::vector< double > Tbar, Vybar;                    // scratch each step
   uint_t nSamples = 0;

   // local (per-step) accumulators
   std::vector< double > sum_T, sum_Vy, cell_count_plane, sum_fluct_prod, avg_fluct_prod;  // used for the convective part
   std::vector< double > dtdy;   // used for the diffusive part with derivatives
   std::vector< double > count_final;
   std::vector<double> Qfluctuation;
   std::vector<double> Qderivative;
   std::vector<double> Qtotal;


   HeatFluxAverager(uint_t Nx_, uint_t Ny_, uint_t Nz_, real_t dy_, real_t alpha_f_, real_t alpha_p_)
      : Nx(Nx_), Ny(Ny_), Nz(Nz_), dy(dy_), alpha_f(alpha_f_), alpha_p(alpha_p_),  Tbar(Ny_, 0),
        Vybar(Ny_, 0), sum_T(Ny_, 0), sum_Vy(Ny_, 0), cell_count_plane(Ny_, 0),
        sum_fluct_prod(Ny_,0), avg_fluct_prod(Ny_,0), dtdy(Ny_,0), count_final(Ny_,0)
        , Qfluctuation(Ny_,0), Qderivative(Ny_,0), Qtotal(Ny_,0)

   {}

   // helper: running mean update
   static inline void runningMean(double& avg, double val, uint_t n) { avg += (val - avg) / double(n); }

   // helper: running mean for vectors (elementwise)
   static inline std::vector<double> runningMeanVec(std::vector<double> &avgVec,
                                     const std::vector<double> &valVec,
                                     uint_t n) {
      const size_t N = avgVec.size();
      for (size_t i = 0; i < N; ++i) {
         avgVec[i] += (valVec[i] - avgVec[i]) / double(n);
      }
      return avgVec;
   }

   // compute one-sided dT/dy (2nd order) at wall planes
   static inline double oneSidedBottom(double T0, double T1, double T2, double dy)
   {
      return (-3.0 * T0 + 4.0 * T1 - T2) / (2.0 * dy);
   }
   static inline double oneSidedTop(double Tm, double Tm1, double Tm2, double dy)
   {
      return (3.0 * Tm - 4.0 * Tm1 + Tm2) / (2.0 * dy);
   }

   // sample one simulation step (call after macros/temperature are updated & halo exchanged)
   void sampleStep(const shared_ptr< StructuredBlockStorage >& blocks,
                   const BlockDataID& velFieldFluidID, // VelocityField_fluid_T  (vy_f)
                   const BlockDataID& tempFieldID,     // DensityField_concentration_T  (your T field)
                   const BlockDataID& BFieldID,        // overlap/solid fraction B
                   uint_t timeStep
   )
   {
      // --- pass 1: plane means of composite Vy and T
      std::fill(sum_T.begin(), sum_T.end(), 0.0);
      std::fill(sum_Vy.begin(), sum_Vy.end(), 0.0);
      std::fill(cell_count_plane.begin(), cell_count_plane.end(), 0.0);
      std::fill(sum_fluct_prod.begin(), sum_fluct_prod.end(), 0.0);
      std::fill(avg_fluct_prod.begin(), avg_fluct_prod.end(), 0.0);
      std::fill(dtdy.begin(), dtdy.end(), 0.0);
      std::fill(count_final.begin(), count_final.end(), 0.0);

      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block = *blockIt;
         auto velF   = block.getData< VelocityField_fluid_T >(velFieldFluidID);
         auto Tfield = block.getData< DensityField_concentration_T >(tempFieldID);
         auto Bfield = block.getData< GhostLayerField< real_t, 1 > >(BFieldID);

         WALBERLA_FOR_ALL_CELLS_XYZ(

            Tfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));

            const uint_t j = uint_c(cell.z()); const real_t B = Bfield->get(x, y, z);
            const real_t vyf                                  = velF->get(x, y, z, 2);
            //WALBERLA_LOG_INFO_ON_ROOT("vy is  " << vyf);
            sum_T[j] += Tfield->get(x, y, z); sum_Vy[j] += vyf; cell_count_plane[j] += 1.0;)
      }
      // reductions needed here for the sums of T and vy and cell_plane_count
      mpi::allReduceInplace(sum_T, mpi::SUM);
      mpi::allReduceInplace(sum_Vy, mpi::SUM);
      mpi::allReduceInplace(cell_count_plane, mpi::SUM);
      // compute fluctuations for each time in each height "y":

      for (uint_t j = 0; j < Ny; ++j)
      {
         Tbar[j]  = (cell_count_plane[j] > 0.0) ? (sum_T[j] / cell_count_plane[j]) : 0.0;
         Vybar[j] = (cell_count_plane[j] > 0.0) ? (sum_Vy[j] / cell_count_plane[j]) : 0.0;
      }

      double wallFluxBot_sum = 0.0, wallFluxTop_sum = 0.0;

      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block = *blockIt;
         auto velF   = block.getData< VelocityField_fluid_T >(velFieldFluidID);
         auto Tfield = block.getData< DensityField_concentration_T >(tempFieldID);
         auto Bfield = block.getData< GhostLayerField< real_t, 1 > >(BFieldID);

         WALBERLA_FOR_ALL_CELLS_XYZ(
            Tfield, Cell cell; blocks->transformBlockLocalToGlobalCell(cell, block, Cell(x, y, z));
            const uint_t j = uint_c(cell.z()); const real_t B = Bfield->get(x, y, z);
            count_final[j] += 1;

            // fluctuations part
            const real_t vy = velF->get(x, y, z, 2); const real_t T = Tfield->get(x, y, z);
            const real_t T_fluctuation = T - Tbar[j]; const real_t vel_fluctuation = vy - Vybar[j];
            //WALBERLA_LOG_INFO_ON_ROOT("vel fluctuation is  " << Vybar[j] << "temp fluctuation is  " << T_fluctuation);
            real_t fluctuation_product = -vel_fluctuation*T_fluctuation;
            sum_fluct_prod[j] += fluctuation_product;

            // dT/dy central (use one-sided later at walls)
            real_t dTdy = 0.0;
            if (j > 0 && j < Ny - 1) {

               // grab neighbors using local y +/- 1 – assumes one ghost layer present

               if(z ==0 )
               {
                  const real_t T0 = Tfield->get(x,y,z);
                  const real_t T1 = Tfield->get(x,y,z+1);
                  const real_t T2 = Tfield->get(x,y,z+2);
                  dTdy = (-3.0*T0 + 4.0*T1 - T2) / (2.0*dy);
                  dTdy  = (1 - B) * alpha_f * dTdy + B * alpha_p * dTdy;
                  dtdy[j] += dTdy;
               }
               else if(z == cell_idx_c(blocks->getNumberOfZCells(block) -1))
               {
                  const real_t T0 = Tfield->get(x,y,z);
                  const real_t Tm1= Tfield->get(x,y,z-1);
                  const real_t Tm2= Tfield->get(x,y,z-2);
                  dTdy = (3.0*T0 - 4.0*Tm1 + Tm2) / (2.0*dy);
                  dTdy  = (1 - B) * alpha_f * dTdy + B * alpha_p * dTdy;
                  dtdy[j] += dTdy;
               }
               else
               {
                  const real_t T0 = Tfield->get(x, y, z-1);
                  const real_t T1 = Tfield->get(x, y, z+1);
                  dTdy            = (T1 - T0) / (2.0 * dy);
                  dTdy            = (1 - B) * alpha_f * dTdy + B * alpha_p * dTdy;
                  dtdy[j] += dTdy;
               }
            }
            else {

               // wall flux (one-sided derivative), average over x,z on the two extreme planes
               //WALBERLA_LOG_INFO_ON_ROOT("entered else part of dtdy loop");
               if (j == 0)
               {
                  //WALBERLA_LOG_INFO_ON_ROOT("entered j==0  dtdy loop");
                  const real_t T0        = Tfield->get(x, y, z);
                  const real_t T1        = Tfield->get(x, y, z+1);
                  const real_t T2        = Tfield->get(x, y, z+2);
                  const real_t dTdy_wall = oneSidedBottom(T0, T1, T2, dy);
                  wallFluxBot_sum = (1 - B) * alpha_f * dTdy_wall + B * alpha_p * dTdy_wall;
                  dtdy[j] += wallFluxBot_sum;
               }
               if (j == Ny - 1)
               {
                  //WALBERLA_LOG_INFO_ON_ROOT("entered j == Ny-1 dtdy loop");
                  const real_t Tm        = Tfield->get(x, y, z);
                  const real_t Tm1       = Tfield->get(x, y, z-1);
                  const real_t Tm2       = Tfield->get(x, y, z-2);
                  const real_t dTdy_wall = oneSidedTop(Tm, Tm1, Tm2, dy);
                  wallFluxTop_sum = (1 - B) * alpha_f * dTdy_wall + B * alpha_p * dTdy_wall;
                  dtdy[j] += wallFluxTop_sum;
               }
            })
      }

      // MPI reduce pass-2

      mpi::allReduceInplace(sum_fluct_prod, mpi::SUM);
      mpi::allReduceInplace(dtdy, mpi::SUM);
      mpi::allReduceInplace(count_final, mpi::SUM);

      // plane-averaged instantaneous and time running averages
      ++nSamples;

      // instantaneous per-plane fluxes
      std::vector<double> q_fluc_inst(Ny, 0.0);
      std::vector<double> q_cond_inst(Ny, 0.0);
      std::vector<double> q_total_inst(Ny, 0.0);

      for (uint_t j = 0; j < Ny; ++j) {
         if (count_final[j] > 0.0) {
            q_fluc_inst[j] = sum_fluct_prod[j] / count_final[j];
            q_cond_inst[j] = dtdy[j] / count_final[j];
            q_total_inst[j] = q_fluc_inst[j] + q_cond_inst[j];
         }
      }

      runningMeanVec(Qderivative, q_cond_inst, nSamples);
      runningMeanVec(Qfluctuation, q_fluc_inst, nSamples);
      runningMeanVec(Qtotal, q_total_inst, nSamples);

      if(timeStep % 10000 == 0) {

         for(int i =0; i < Ny; i++){

            WALBERLA_LOG_INFO_ON_ROOT(" q dtdy " << i << " is " << Qderivative[i]
                                                 << "  q fluctuation " << i << " is " << Qfluctuation[i]
                                                 << "  q totl "<< i << " is  "  << Qtotal[i]);

         }
      }
   }
};
}
