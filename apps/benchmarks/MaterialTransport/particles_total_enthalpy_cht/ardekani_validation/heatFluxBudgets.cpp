//
// Created by ravi-kiran on 30.01.26.
//
//
// Created by dy94rovu on 9/26/25.
//
#include "core/DataTypes.h"
#include "core/math/all.h"

#include "GeneralInfoHeader.h"

namespace walberla
{
struct HeatFluxBudgets
{
   // geometry
   uint_t Nx, Ny, Nz;
   real_t dz;

   // constants
   real_t alpha_f, alpha_p; // thermal diffusivities (LB units)

   std::vector< double > Tparticle_plane, Vparticle_plane; // particle temperature and velocity averages
   std::vector< double > Tfluid_plane, Vfluid_plane;       // fluid temperature and velocity averages
   std::vector< double > TVparticle_plane, TVfluid_plane;  // particle temperature velocity product averages
   std::vector< double > dTparticle_plane, dTfluid_plane;  // fluid temperature velocity product averages
   std::vector< double > phi_p, phi_f;                     // fraction fields of particle and fluid
   std::vector< double > cell_count_plane;
   uint_t nSamples = 0;

   HeatFluxBudgets(uint_t Nz_, real_t dz_, real_t alpha_f_, real_t alpha_p_)
      : Nz(Nz_), dz(dz_), alpha_f(alpha_f_), alpha_p(alpha_p_), Tparticle_plane(Nz_, 0), Vparticle_plane(Nz_, 0),
        Tfluid_plane(Nz_, 0), Vfluid_plane(Nz_, 0), TVparticle_plane(Nz_, 0), TVfluid_plane(Nz_, 0),
        dTparticle_plane(Nz_, 0), dTfluid_plane(Nz_, 0), phi_p(Nz_, 0), phi_f(Nz_, 0), cell_count_plane(Nz_, 0)

   {}

   // helper: running mean update
   static inline void runningMean(double& avg, double val, uint_t n) { avg += (val - avg) / double(n); }

   // helper: running mean for vectors (elementwise)
   static inline std::vector< double > runningMeanVec(std::vector< double >& avgVec,
                                                      const std::vector< double >& valVec, uint_t n)
   {
      const size_t N = avgVec.size();
      for (size_t i = 0; i < N; ++i)
      {
         avgVec[i] += (valVec[i] - avgVec[i]) / double(n);
      }
      return avgVec;
   }

   // compute one-sided dT/dy (2nd order) at wall planes
   static inline double oneSidedBottom(double T0, double T1, double T2, double dy)
   { return (-3.0 * T0 + 4.0 * T1 - T2) / (2.0 * dy); }
   static inline double oneSidedTop(double Tm, double Tm1, double Tm2, double dy)
   { return (3.0 * Tm - 4.0 * Tm1 + Tm2) / (2.0 * dy); }

   // sample one simulation step (call after macros/temperature are updated & halo exchanged)
   void planeAverage(const shared_ptr< StructuredBlockStorage >& blocks,
                     const BlockDataID& velFieldFluidID, // VelocityField_fluid_T  (vy_f)
                     const BlockDataID& tempFieldID,     // DensityField_concentration_T  (your T field)
                     const BlockDataID& BFieldID,        // overlap/solid fraction B
                     uint_t timeStep)
   {
      // --- pass 1: plane means of composite Vy and T
      std::fill(Tparticle_plane.begin(), Tparticle_plane.end(), 0.0);
      std::fill(Vparticle_plane.begin(), Vparticle_plane.end(), 0.0);

      std::fill(Tfluid_plane.begin(), Tfluid_plane.end(), 0.0);
      std::fill(Vfluid_plane.begin(), Vfluid_plane.end(), 0.0);

      std::fill(TVparticle_plane.begin(), TVparticle_plane.end(), 0.0);
      std::fill(TVfluid_plane.begin(), TVfluid_plane.end(), 0.0);

      std::fill(dTparticle_plane.begin(), dTparticle_plane.end(), 0.0);
      std::fill(dTfluid_plane.begin(), dTfluid_plane.end(), 0.0);

      std::fill(phi_p.begin(), phi_p.end(), 0.0);
      std::fill(phi_f.begin(), phi_f.end(), 0.0);

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
            const real_t velocity_z = velF->get(x, y, z, 2); Tparticle_plane[j] += (B) *temperature;
            Tfluid_plane[j] += (1 - B) * temperature;

            TVparticle_plane[j] += (B) *temperature * (B) *velocity_z;
            Tfluid_plane[j] += (1 - B) * temperature * (1 - B) * velocity_z;

            phi_p[j] += B; phi_f[j] += (1 - B);

            cell_count_plane[j] += 1.0;

            // for the conductive part of particles and fluid
            real_t dTdy_p = 0.0;
            real_t dTdy_f = 0.0; if (j > 0 && j < Ny - 1) {
               // for the INNER mpi blocks of the domain but the FIRST or the LAST cells use one sided FD
               // for the INNER mpi blocks of the domain and all middle cells use central FD

               if (z == 0)
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z + 1);
                  const real_t T2_p = B * Tfield->get(x, y, z + 2);
                  dTdy_p            = (-3.0 * T0_p + 4.0 * T1_p - T2_p) / (2.0 * dz);
                  dTdy_p            = alpha_p * dTdy_p;
                  dTparticle_plane[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z + 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z + 2);
                  dTdy_f            = (-3.0 * T0_f + 4.0 * T1_f - T2_f) / (2.0 * dz);
                  dTdy_f            = alpha_f * dTdy_f;
                  dTfluid_plane[j] += dTdy_f;
               }
               else if (z == cell_idx_c(blocks->getNumberOfZCells(block) - 1))
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z - 1);
                  const real_t T2_p = B * Tfield->get(x, y, z - 2);
                  dTdy_p            = (3.0 * T0_p - 4.0 * T1_p + T2_p) / (2.0 * dz);
                  dTdy_p            = alpha_p * dTdy_p;
                  dTparticle_plane[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z - 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z - 2);
                  dTdy_f            = (3.0 * T0_f - 4.0 * T1_f + T2_f) / (2.0 * dz);
                  dTdy_f            = alpha_f * dTdy_f;
                  dTfluid_plane[j] += dTdy_f;
               }
               else
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z - 1);
                  const real_t T1_p = B * Tfield->get(x, y, z + 1);
                  dTdy_p            = (T1_p - T0_p) / (2.0 * dz);
                  dTdy_p            = alpha_p * dTdy_p;
                  dTparticle_plane[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = B * Tfield->get(x, y, z - 1);
                  const real_t T1_f = B * Tfield->get(x, y, z + 1);
                  dTdy_f            = (T1_f - T0_f) / (2.0 * dz);
                  dTdy_f            = alpha_f * dTdy_f;
                  dTfluid_plane[j] += dTdy_f;
               }
            } else {
               // wall flux (one-sided derivative), average over x,z on the two extreme planes

               if (j == 0)
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z + 1);
                  const real_t T2_p = B * Tfield->get(x, y, z + 2);
                  dTdy_p            = (-3.0 * T0_p + 4.0 * T1_p - T2_p) / (2.0 * dz);
                  dTdy_p            = alpha_p * dTdy_p;
                  dTparticle_plane[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z + 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z + 2);
                  dTdy_f            = (-3.0 * T0_f + 4.0 * T1_f - T2_f) / (2.0 * dz);
                  dTdy_f            = alpha_f * dTdy_f;
                  dTfluid_plane[j] += dTdy_f;
               }
               if (j == Ny - 1)
               {
                  // particles conductive flux
                  const real_t T0_p = B * Tfield->get(x, y, z);
                  const real_t T1_p = B * Tfield->get(x, y, z - 1);
                  const real_t T2_p = B * Tfield->get(x, y, z - 2);
                  dTdy_p            = (3.0 * T0_p - 4.0 * T1_p + T2_p) / (2.0 * dz);
                  dTdy_p            = alpha_p * dTdy_p;
                  dTparticle_plane[j] += dTdy_p;

                  // fluid conductive flux
                  const real_t T0_f = (1 - B) * Tfield->get(x, y, z);
                  const real_t T1_f = (1 - B) * Tfield->get(x, y, z - 1);
                  const real_t T2_f = (1 - B) * Tfield->get(x, y, z - 2);
                  dTdy_f            = (3.0 * T0_f - 4.0 * T1_f + T2_f) / (2.0 * dz);
                  dTdy_f            = alpha_f * dTdy_f;
                  dTfluid_plane[j] += dTdy_f;
               }
            }

         )
      }
      // reductions needed here for all the quantities

      mpi::allReduceInplace(Tparticle_plane, mpi::SUM);
      mpi::allReduceInplace(Vparticle_plane, mpi::SUM);

      mpi::allReduceInplace(TVparticle_plane, mpi::SUM);
      mpi::allReduceInplace(TVfluid_plane, mpi::SUM);

      mpi::allReduceInplace(dTparticle_plane, mpi::SUM);
      mpi::allReduceInplace(dTfluid_plane, mpi::SUM);

      mpi::allReduceInplace(phi_p, mpi::SUM);
      mpi::allReduceInplace(phi_f, mpi::SUM);

      mpi::allReduceInplace(cell_count_plane, mpi::SUM);

      // computing the plane averages here

      for (size_t k = 0; k < cell_count_plane.size(); ++k)
      {
         if (cell_count_plane[k] > 0)
         {
            Tparticle_plane[k] /= cell_count_plane[k];
            Vparticle_plane[k] /= cell_count_plane[k];
            TVparticle_plane[k] /= cell_count_plane[k];
            TVfluid_plane[k] /= cell_count_plane[k];
            dTparticle_plane[k] /= cell_count_plane[k];
            dTfluid_plane[k] /= cell_count_plane[k];
            phi_p[k] /= cell_count_plane[k];
            phi_f[k] /= cell_count_plane[k];
         }
      }
   }

   void writePlaneAverages(uint_t timeStep) const
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

         for (size_t k = 0; k < cell_count_plane.size(); ++k)
         {
            file << k << " " << Tparticle_plane[k] << " " << Vparticle_plane[k] << " " << TVparticle_plane[k] << " "
                 << TVfluid_plane[k] << " " << dTparticle_plane[k] << " " << dTfluid_plane[k] << " " << phi_p[k] << " "
                 << phi_f[k] << " " << cell_count_plane[k] << "\n";
         }

         file.close();
      }
   }
};
} // namespace walberla
