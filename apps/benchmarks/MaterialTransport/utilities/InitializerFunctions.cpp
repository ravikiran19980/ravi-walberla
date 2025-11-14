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
//! \file InitializerFunctions.h
//! \author Ravi Ayyala Somayajula <ravi.k.ayyala@fau.de>
//
//======================================================================================================================

#include "core/Environment.h"
#include "core/all.h"
#include "core/logging/Initialization.h"
#include "core/math/Constants.h"

#include "field/FlagField.h"
#include "field/communication/PackInfo.h"
#include "field/vtk/VTKWriter.h"

// #include "python_coupling/DictWrapper.h"
#include "GeneralInfoHeader.h"
#include "InitializerFunctions.h"
#pragma once

namespace walberla
{

void initConcentrationField(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID& ConcentrationFieldID,
                            const math::AABB& domainAABB, Vector3< uint_t > domainSize, bool performanceBenchmark,const real_t Tref)
{
   //const real_t radius = real_c(domainSize[1] / 10);
   for (auto& block : *blocks)
   {
      auto ConcentrationField = block.getData< DensityField_concentration_T >(ConcentrationFieldID);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(ConcentrationField, {
         Cell globalCell;
         const auto cellAABB = blocks->getBlockLocalCellAABB(block, Cell(x, y, z));
         auto cellCenter     = cellAABB.center();
         const real_t posX   = x; // cellCenter[0];
         const real_t posY   = y; // cellCenter[1];
         const real_t posZ   = z; // cellCenter[2];

         if(performanceBenchmark){
            ConcentrationField->get(x, y, z) = real_t(0.2);
         }
         else{
            ConcentrationField->get(x, y, z) = real_t(Tref);
         }

      }) // WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ
   }

} // initConcentrationField


void initConcentrationFieldCoutte(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID& ConcentrationFieldID,
                                  BlockDataID& BFieldID,Vector3< uint_t > domainSize )
{

   for (auto& block : *blocks)
   {
      Block& b                = dynamic_cast< Block& >(block);
      uint_t level            = b.getLevel();
      auto ConcentrationField = block.getData< DensityField_concentration_T >(ConcentrationFieldID);
      auto Bfield = block.getData< GhostLayerField< real_t, 1 > >(BFieldID);
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(ConcentrationField, {
         Cell globalCell;
         blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x,y,z));
         Vector3< real_t > position = blocks->getCellCenter(globalCell, level);
         const real_t posX   = position[0]; // cellCenter[0];
         const real_t posY   = position[1]; // cellCenter[1];
         const real_t posZ   = position[2]; // cellCenter[2];

         ConcentrationField->get(x, y, z) = (1 - Bfield->get(x,y,z)) * (0.5 - posZ/(domainSize[2])) + Bfield->get(x,y,z)*((0.5 - posZ/(domainSize[2])));

      }) // WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ
   }

} // initConcentrationField


void initFluidField(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID& FluidFieldID,
                    const Vector3< real_t > uInflow,Vector3< uint_t > domainSize)
{
   for (auto& block : *blocks)
   {

      auto FluidVelocityField = block.getData< VelocityField_fluid_T >(FluidFieldID);

      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(FluidVelocityField, {
         Cell globalCell;
         const auto cellAABB                 = blocks->getBlockLocalCellAABB(block, Cell(x, y, z));
         FluidVelocityField->get(x, y, z, 0) = uInflow[0];
         FluidVelocityField->get(x, y, z, 1) = uInflow[1];
         if(domainSize[2] != 1)
         {
            FluidVelocityField->get(x, y, z, 2) = uInflow[2];
         }
      }) // WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ
   }
}

void initFluidFieldCoutte(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID& FluidFieldID, const real_t h,
                    const Vector3< real_t > uInflow)
{
   for (auto& block : *blocks)
   {
      Block& b                = dynamic_cast< Block& >(block);
      uint_t level            = b.getLevel();
      auto FluidVelocityField = block.getData< VelocityField_fluid_T >(FluidFieldID);

      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(FluidVelocityField, {
         Cell globalCell;
         blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x, y, z));
         Vector3< real_t > position          = blocks->getCellCenter(globalCell, level);
         FluidVelocityField->get(x, y, z, 0) = (uInflow[0] / (2 * h)) * (position[2] - h);
         FluidVelocityField->get(x, y, z, 1) = uInflow[1];
         FluidVelocityField->get(x, y, z, 2) = uInflow[2];
      }) // WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ
   }
}

void initFluidFieldPoiseuille(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID& FluidFieldID,
                    const Vector3< real_t > uInflow,Vector3< uint_t > domainSize)
{
   for (auto& block : *blocks)
   {
      auto FluidVelocityField = block.getData< VelocityField_fluid_T >(FluidFieldID);

      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(FluidVelocityField, {
         Cell globalCell;
         const auto cellAABB                 = blocks->getBlockLocalCellAABB(block, Cell(x, y, z));
         FluidVelocityField->get(x, y, z, 0) = uInflow[0]*(1 - std::pow((y/real_c(domainSize[1])),2));
         FluidVelocityField->get(x, y, z, 1) = 0;

      }) // WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ
   }
}


std::vector< real_t > computeErrorL2(const shared_ptr< StructuredBlockStorage >& blocks,
                                     BlockDataID& NumericalSolFieldID, BlockDataID& AnalyticalSolFieldID,
                                     BlockDataID& ErrorFieldID, const math::AABB& domainAABB)
{
   real_t Linf{ 0.0 };
   real_t L1{ 0.0 };
   real_t L2{ 0.0 };
   real_t analytical_squared{ 0.0 };
   uint_t cells{ 0 };
   std::vector< real_t > Errors{ 0, 0, 0 };

   for (auto block = blocks->begin(); block != blocks->end(); ++block)
   {
      auto numericalSolutionField  = block->getData< DensityField_concentration_T >(NumericalSolFieldID);
      auto analyticalSolutionField = block->getData< DensityField_concentration_T >(AnalyticalSolFieldID);
      auto ErrorField              = block->getData< DensityField_concentration_T >(ErrorFieldID);
      Block& b                     = dynamic_cast< Block& >(*block);
      uint_t level                 = b.getLevel();
      CellInterval xyz             = analyticalSolutionField->xyzSize();

      for (auto cellIt = xyz.begin(); cellIt != xyz.end(); ++cellIt)
      {
         real_t currErr = (numericalSolutionField->get(*cellIt) - analyticalSolutionField->get(*cellIt));

         // potentialErrorField ->get(x, y, z) = real_c(fabs(currErr));
         ErrorField->get(*cellIt) = fabs(currErr);
         L1 += std::abs(currErr);
         L2 += currErr * currErr;
         analytical_squared += analyticalSolutionField->get(*cellIt) * analyticalSolutionField->get(*cellIt);
         Linf = std::max(Linf, std::abs(currErr));
         cells += 1;
      }
   }
   mpi::allReduceInplace(L1, mpi::SUM);
   mpi::allReduceInplace(L2, mpi::SUM);
   mpi::allReduceInplace(analytical_squared, mpi::SUM);
   mpi::allReduceInplace(Linf, mpi::MAX);
   mpi::allReduceInplace(cells, mpi::SUM);
   Errors[0] = Linf;
   Errors[1] = L1 / cells;
   Errors[2] = std::sqrt(L2 / analytical_squared);
   return Errors;
}

 real_t computeResidual(const shared_ptr< StructuredBlockStorage >& blocks,
                                     BlockDataID& oldtemperatureFieldID, BlockDataID& currenttemperatureFieldID
                                     )
{
   real_t Linf{ 0.0 };
   real_t L1{ 0.0 };
   real_t L2{ 0.0 };
   uint_t residual{ 0 };
   uint_t cells{ 0 };
   real_t maxresidual{0};

   for (auto block = blocks->begin(); block != blocks->end(); ++block)
   {
      auto oldtemperatureField  = block->getData< DensityField_concentration_T >(oldtemperatureFieldID);
      auto currenttemperatureField = block->getData< DensityField_concentration_T >(currenttemperatureFieldID);
      Block& b                     = dynamic_cast< Block& >(*block);
      uint_t level                 = b.getLevel();
      CellInterval xyz             = currenttemperatureField->xyzSize();

      for (auto cellIt = xyz.begin(); cellIt != xyz.end(); ++cellIt)
      {
         real_t currErr = (currenttemperatureField->get(*cellIt) - oldtemperatureField->get(*cellIt));
         
         L1 += std::abs(currErr);
         L2 += currErr * currErr;
         Linf = std::max(Linf, std::abs(currErr));
         cells += 1;
         // after computing the residuals, assign old temperature with current temperature
         oldtemperatureField->get(*cellIt) = currenttemperatureField->get(*cellIt);
      }
   }
   mpi::allReduceInplace(L1, mpi::SUM);
   mpi::allReduceInplace(L2, mpi::SUM);
   mpi::allReduceInplace(Linf, mpi::MAX);
   mpi::allReduceInplace(cells, mpi::SUM);
   //Errors[0] = Linf;
   //Errors[1] = L1 / cells;
   //Errors[2] = std::sqrt(L2 / analytical_squared);
   maxresidual = Linf;

   return maxresidual;
}



} // namespace walberla