#pragma once

#include <array>

#include "HeatEvaluators.h"
#include "gpu/ErrorChecking.h"
#include "gpu/FieldAccessor.h"
#include "gpu/FieldIndexing.h"
#include "gpu/GPUField.h"

namespace MaterialTransport
{
using namespace walberla;

namespace detail
{
__global__ void wallNormalHeatFluxKernel(gpu::FieldAccessor< real_t > temperatureField,
                                         gpu::FieldAccessor< real_t > bField, uint_t xSize, uint_t ySize, uint_t zSize,
                                         uint_t globalZOffset, uint_t globalNz, real_t dz, real_t alphaFluid,
                                         real_t alphaParticle, double* qbot, double* qtop, double* nbot, double* ntop)
{
   const uint3 bIdx = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
   const uint3 tIdx = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

   const uint_t linearIdx = temperatureField.getLinearIndex(bIdx, tIdx, gridDim, blockDim);
   const uint_t numCells  = xSize * ySize * zSize;
   if (linearIdx >= numCells) { return; }

   const uint_t z = linearIdx / (xSize * ySize);
   const uint_t j = globalZOffset + z;

   if (j != 0 && j != globalNz - 1) { return; }

   temperatureField.set(bIdx, tIdx);
   bField.set(bIdx, tIdx);

   const real_t B      = bField.get();
   const real_t alpha  = (real_t(1) - B) * alphaFluid + B * alphaParticle;
   const real_t twoDz  = real_t(2) * dz;
   const real_t center = temperatureField.get();

   if (j == 0)
   {
      const real_t dTdz = (-real_t(3) * center + real_t(4) * temperatureField.getNeighbor(0, 0, 1) -
                           temperatureField.getNeighbor(0, 0, 2)) /
                          twoDz;
      atomicAdd(qbot, double(alpha * dTdz));
      atomicAdd(nbot, double(1));
   }
   else
   {
      const real_t dTdz = (real_t(3) * center - real_t(4) * temperatureField.getNeighbor(0, 0, -1) +
                           temperatureField.getNeighbor(0, 0, -2)) /
                          twoDz;
      atomicAdd(qtop, double(alpha * dTdz));
      atomicAdd(ntop, double(1));
   }
}

// MeanPlaneAverager::operator()
__global__ void meanPlaneAveragerKernel(gpu::FieldAccessor< real_t > velocityField,
                                        gpu::FieldAccessor< real_t > temperatureField, uint_t xSize, uint_t ySize,
                                        uint_t zSize, uint_t globalZOffset, uint_t globalNz, double* velocityPlane,
                                        double* temperaturePlane, double* numPlaneCells)
{
   const uint3 bIdx = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
   const uint3 tIdx = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

   const uint_t linearIdx = temperatureField.getLinearIndex(bIdx, tIdx, gridDim, blockDim);
   const uint_t numCells  = xSize * ySize * zSize;
   if (linearIdx >= numCells) { return; }

   const uint_t z = linearIdx / (xSize * ySize);
   const uint_t j = globalZOffset + z;
   if (j >= globalNz) { return; }

   velocityField.set(bIdx, tIdx);
   temperatureField.set(bIdx, tIdx);

   const real_t velocityZ   = velocityField.get(2);
   const real_t temperature = temperatureField.get();

   atomicAdd(velocityPlane + j, double(velocityZ));
   atomicAdd(temperaturePlane + j, double(temperature));
   atomicAdd(numPlaneCells + j, double(1));
}

// HeatFluxBudgets::operator()
__global__ void heatFluxBudgetsKernel(gpu::FieldAccessor< real_t > velocityField, gpu::FieldAccessor< real_t > temperatureField,
                                      gpu::FieldAccessor< real_t > bField, uint_t xSize, uint_t ySize, uint_t zSize,
                                      uint_t globalZOffset, uint_t globalNz, real_t dz, real_t alphaFluid,
                                      real_t alphaParticle, const double* velocityMean, const double* temperatureMean,
                                      double* particleFluctuation, double* fluidFluctuation, double* dTparticle,
                                      double* dTfluid, double* phiP, double* phiF, double* cellCount)
{
   const uint3 bIdx = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
   const uint3 tIdx = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

   const uint_t linearIdx = temperatureField.getLinearIndex(bIdx, tIdx, gridDim, blockDim);
   const uint_t numCells  = xSize * ySize * zSize;
   if (linearIdx >= numCells) { return; }

   const uint_t z = linearIdx / (xSize * ySize);
   const uint_t j = globalZOffset + z;
   if (j >= globalNz) { return; }

   velocityField.set(bIdx, tIdx);
   temperatureField.set(bIdx, tIdx);
   bField.set(bIdx, tIdx);

   const real_t B           = bField.get();
   const real_t temperature = temperatureField.get();
   const real_t velocityZ   = velocityField.get(2);
   const real_t T_dash      = temperature - real_t(temperatureMean[j]);
   const real_t V_dash      = velocityZ - real_t(velocityMean[j]);

   atomicAdd(particleFluctuation + j, double(-B * T_dash * V_dash));
   atomicAdd(fluidFluctuation + j, double(-(real_t(1) - B) * T_dash * V_dash));
   atomicAdd(phiP + j, double(B));
   atomicAdd(phiF + j, double(real_t(1) - B));
   atomicAdd(cellCount + j, double(1));

   real_t dTdzP = real_t(0);
   real_t dTdzF = real_t(0);

   if (j > 0 && j < globalNz - 1)
   {
      if (z == 0)
      {
         const real_t T0p = bField.get() * temperatureField.get();
         const real_t T1p = bField.getNeighbor(0, 0, 1) * temperatureField.getNeighbor(0, 0, 1);
         const real_t T2p = bField.getNeighbor(0, 0, 2) * temperatureField.getNeighbor(0, 0, 2);
         dTdzP            = alphaParticle * (-real_t(3) * T0p + real_t(4) * T1p - T2p) / (real_t(2) * dz);

         const real_t B0f = real_t(1) - bField.get();
         const real_t B1f = real_t(1) - bField.getNeighbor(0, 0, 1);
         const real_t B2f = real_t(1) - bField.getNeighbor(0, 0, 2);
         const real_t T0f = B0f * temperatureField.get();
         const real_t T1f = B1f * temperatureField.getNeighbor(0, 0, 1);
         const real_t T2f = B2f * temperatureField.getNeighbor(0, 0, 2);
         dTdzF            = alphaFluid * (-real_t(3) * T0f + real_t(4) * T1f - T2f) / (real_t(2) * dz);
      }
      else if (z == zSize - 1)
      {
         const real_t T0p = bField.get() * temperatureField.get();
         const real_t T1p = bField.getNeighbor(0, 0, -1) * temperatureField.getNeighbor(0, 0, -1);
         const real_t T2p = bField.getNeighbor(0, 0, -2) * temperatureField.getNeighbor(0, 0, -2);
         dTdzP            = alphaParticle * (real_t(3) * T0p - real_t(4) * T1p + T2p) / (real_t(2) * dz);

         const real_t B0f = real_t(1) - bField.get();
         const real_t B1f = real_t(1) - bField.getNeighbor(0, 0, -1);
         const real_t B2f = real_t(1) - bField.getNeighbor(0, 0, -2);
         const real_t T0f = B0f * temperatureField.get();
         const real_t T1f = B1f * temperatureField.getNeighbor(0, 0, -1);
         const real_t T2f = B2f * temperatureField.getNeighbor(0, 0, -2);
         dTdzF            = alphaFluid * (real_t(3) * T0f - real_t(4) * T1f + T2f) / (real_t(2) * dz);
      }
      else
      {
         const real_t T0p = bField.getNeighbor(0, 0, -1) * temperatureField.getNeighbor(0, 0, -1);
         const real_t T1p = bField.getNeighbor(0, 0, 1) * temperatureField.getNeighbor(0, 0, 1);
         dTdzP            = alphaParticle * (T1p - T0p) / (real_t(2) * dz);

         const real_t T0f =
            (real_t(1) - bField.getNeighbor(0, 0, -1)) * temperatureField.getNeighbor(0, 0, -1);
         const real_t T1f = (real_t(1) - bField.getNeighbor(0, 0, 1)) * temperatureField.getNeighbor(0, 0, 1);
         dTdzF            = alphaFluid * (T1f - T0f) / (real_t(2) * dz);
      }
   }
   else
   {
      if (j == 0)
      {
         const real_t T0p = bField.get() * temperatureField.get();
         const real_t T1p = bField.getNeighbor(0, 0, 1) * temperatureField.getNeighbor(0, 0, 1);
         const real_t T2p = bField.getNeighbor(0, 0, 2) * temperatureField.getNeighbor(0, 0, 2);
         dTdzP            = alphaParticle * (-real_t(3) * T0p + real_t(4) * T1p - T2p) / (real_t(2) * dz);

         const real_t T0f = (real_t(1) - bField.get()) * temperatureField.get();
         const real_t T1f =
            (real_t(1) - bField.getNeighbor(0, 0, 1)) * temperatureField.getNeighbor(0, 0, 1);
         const real_t T2f =
            (real_t(1) - bField.getNeighbor(0, 0, 2)) * temperatureField.getNeighbor(0, 0, 2);
         dTdzF = alphaFluid * (-real_t(3) * T0f + real_t(4) * T1f - T2f) / (real_t(2) * dz);
      }
      else if (j == globalNz - 1)
      {
         const real_t T0p = bField.get() * temperatureField.get();
         const real_t T1p = bField.getNeighbor(0, 0, -1) * temperatureField.getNeighbor(0, 0, -1);
         const real_t T2p = bField.getNeighbor(0, 0, -2) * temperatureField.getNeighbor(0, 0, -2);
         dTdzP            = alphaParticle * (real_t(3) * T0p - real_t(4) * T1p + T2p) / (real_t(2) * dz);

         const real_t T0f = (real_t(1) - bField.get()) * temperatureField.get();
         const real_t T1f =
            (real_t(1) - bField.getNeighbor(0, 0, -1)) * temperatureField.getNeighbor(0, 0, -1);
         const real_t T2f =
            (real_t(1) - bField.getNeighbor(0, 0, -2)) * temperatureField.getNeighbor(0, 0, -2);
         dTdzF = alphaFluid * (real_t(3) * T0f - real_t(4) * T1f + T2f) / (real_t(2) * dz);
      }
   }

   atomicAdd(dTparticle + j, double(dTdzP));
   atomicAdd(dTfluid + j, double(dTdzF));
}

// WallNusseltNumber::operator()
__global__ void wallNusseltKernel(gpu::FieldAccessor< real_t > temperatureField, uint_t xSize, uint_t ySize,
                                  uint_t zSize, uint_t globalZOffset, uint_t globalNz, real_t dz, real_t TwallTop,
                                  real_t TwallBottom, double* nuBottom, double* nuTop, double* countBottom,
                                  double* countTop)
{
   const uint3 bIdx = make_uint3(blockIdx.x, blockIdx.y, blockIdx.z);
   const uint3 tIdx = make_uint3(threadIdx.x, threadIdx.y, threadIdx.z);

   const uint_t linearIdx = temperatureField.getLinearIndex(bIdx, tIdx, gridDim, blockDim);
   const uint_t numCells  = xSize * ySize * zSize;
   if (linearIdx >= numCells) { return; }

   const uint_t z = linearIdx / (xSize * ySize);
   const uint_t j = globalZOffset + z;
   if (j != 0 && j != globalNz - 1) { return; }

   temperatureField.set(bIdx, tIdx);

   if (j == 0)
   {
      const real_t T1   = temperatureField.getNeighbor(0, 0, 1);
      const real_t T2   = temperatureField.getNeighbor(0, 0, 2);
      const real_t dTdz = (-real_t(8) * TwallBottom + real_t(9) * T1 - T2) / (real_t(3) * dz);
      atomicAdd(nuBottom, double(dTdz));
      atomicAdd(countBottom, double(1));
   }
   else
   {
      const real_t T1   = temperatureField.getNeighbor(0, 0, -1);
      const real_t T2   = temperatureField.getNeighbor(0, 0, -2);
      const real_t dTdz = (real_t(8) * TwallTop - real_t(9) * T1 + T2) / (real_t(3) * dz);
      atomicAdd(nuTop, double(dTdz));
      atomicAdd(countTop, double(1));
   }
}
} // namespace detail

class WallNormalHeatFluxGPU
{
 public:
   WallNormalHeatFluxGPU(uint_t Nz, real_t dz, real_t alpha_f, real_t alpha_p,
                         const std::string& = "meanFlux.txt")
   : Nz_(Nz), dz_(dz), alpha_f_(alpha_f), alpha_p_(alpha_p)
   {}

   WallNormalHeatFluxGPU(uint_t Nz, real_t dz, real_t alpha_f, real_t alpha_p, const BlockDataID&, const BlockDataID&,
                         const std::string& = "meanFlux.txt")
      : Nz_(Nz), dz_(dz), alpha_f_(alpha_f), alpha_p_(alpha_p)
   {}

   real_t computeMeanWallNormalFlux(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& tempFieldID,
                                    const BlockDataID& BFieldID) const
   {

      double* dSums = nullptr;
      WALBERLA_GPU_CHECK(gpuMalloc(reinterpret_cast< void** >(&dSums), 4 * sizeof(double)));
      WALBERLA_GPU_CHECK(gpuMemset(dSums, 0, 4 * sizeof(double)));

      for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
      {
         auto& block  = *blockIt;
         auto TField  = block.getData< gpu::GPUField< real_t > >(tempFieldID);
         auto BField  = block.getData< gpu::GPUField< real_t > >(BFieldID);
         auto TLayout = gpu::FieldIndexing< real_t >::xyz(*TField);
         auto BLayout = gpu::FieldIndexing< real_t >::xyz(*BField);

         Cell firstCell;
         blocks->transformBlockLocalToGlobalCell(firstCell, block, Cell(0, 0, 0));
         const uint_t globalZOffset = uint_c(firstCell.z());

         detail::wallNormalHeatFluxKernel<<<TLayout.gridDim(), TLayout.blockDim()>>>(
            TLayout.gpuAccess(), BLayout.gpuAccess(), TField->xSize(), TField->ySize(), TField->zSize(), globalZOffset,
            Nz_, dz_, alpha_f_, alpha_p_, dSums + 0, dSums + 1, dSums + 2, dSums + 3);
         WALBERLA_GPU_CHECK_LAST_ERROR();
      }

      WALBERLA_GPU_CHECK(gpuDeviceSynchronize());

      std::array< double, 4 > hostSums{ 0.0, 0.0, 0.0, 0.0 };
      WALBERLA_GPU_CHECK(gpuMemcpy(hostSums.data(), dSums, 4 * sizeof(double), gpuMemcpyDeviceToHost));
      WALBERLA_GPU_CHECK(gpuFree(dSums));

      double qbot = hostSums[0];
      double qtop = hostSums[1];
      double nbot = hostSums[2];
      double ntop = hostSums[3];

      WALBERLA_MPI_SECTION()
      {
         mpi::allReduceInplace(qbot, mpi::SUM);
         mpi::allReduceInplace(qtop, mpi::SUM);
         mpi::allReduceInplace(nbot, mpi::SUM);
         mpi::allReduceInplace(ntop, mpi::SUM);
      }

      WALBERLA_ASSERT_GREATER(nbot, 0.0);
      WALBERLA_ASSERT_GREATER(ntop, 0.0);

      return real_c(0.5 * (qbot / nbot + qtop / ntop));

   }

 private:
   uint_t Nz_;
   real_t dz_;
   real_t alpha_f_;
   real_t alpha_p_;
};

} // namespace MaterialTransport
