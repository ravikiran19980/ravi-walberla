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
//! \file InitializerFunctions.cpp
//! \author Markus Holzer <markus.holzer@fau.de>
//
//======================================================================================================================
#include "core/Environment.h"
#include "core/logging/Initialization.h"
#include "core/math/Constants.h"

#include "field/FlagField.h"
#include "field/communication/PackInfo.h"
#include "field/vtk/VTKWriter.h"

namespace walberla
{
using PhaseField_T = GhostLayerField< real_t, 1 >;

void initPhaseField_bubble(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID phaseFieldID,
                           const real_t R                         = 10,
                           const Vector3< real_t > bubbleMidPoint = Vector3< real_t >(0.0, 0.0, 0.0),
                           const real_t W                         = 5)
{
   for (auto& block : *blocks)
   {
      auto phaseField = block.getData< PhaseField_T >(phaseFieldID);
      // clang-format off
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(phaseField, Cell globalCell;
         blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x, y, z));
         real_t Ri = std::sqrt((real_c(globalCell[0]) - bubbleMidPoint[0]) * (real_c(globalCell[0]) - bubbleMidPoint[0]) +
                               (real_c(globalCell[1]) - bubbleMidPoint[1]) * (real_c(globalCell[1]) - bubbleMidPoint[1]) +
                               (real_c(globalCell[2]) - bubbleMidPoint[2]) * (real_c(globalCell[2]) - bubbleMidPoint[2]));
         phaseField->get(x, y, z) = 0.5_r + 0.5_r * std::tanh(2.0_r * (Ri - R) / W);
      )
      // clang-format on
   }
}

void initPhaseField_RTI(const shared_ptr< StructuredBlockStorage >& blocks, BlockDataID phaseFieldID,
                        const real_t W = 5)
{
   const real_t X              = real_c(blocks->getDomainCellBB().xMax());
   const real_t halfY          = real_c(blocks->getDomainCellBB().yMax()) / 2.0_r;
   const real_t perturbation   = 0.05_r;

   for (auto& block : *blocks)
   {
      auto phaseField = block.getData< PhaseField_T >(phaseFieldID);
      // clang-format off
      WALBERLA_FOR_ALL_CELLS_INCLUDING_GHOST_LAYER_XYZ(phaseField, Cell globalCell;
         blocks->transformBlockLocalToGlobalCell(globalCell, block, Cell(x, y, z));
         real_t tmp = perturbation * X * (std::cos((2.0_r * math::pi * real_c(globalCell[0])) / X)
                                        + std::cos((2.0_r * math::pi * real_c(globalCell[2])) / X));
         phaseField->get(x, y, z) = 0.5_r + 0.5_r * std::tanh(((real_t(globalCell[1]) - halfY) - tmp) / (W / 2.0_r));
      )
      // clang-format on
   }
}
} // namespace walberla
