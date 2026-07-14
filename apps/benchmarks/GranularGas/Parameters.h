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
//! \file Parameters.h
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//
//======================================================================================================================

//======================================================================================================================
//
//  THIS FILE IS GENERATED - PLEASE CHANGE THE TEMPLATE !!!
//
//======================================================================================================================

#pragma once

#include <core/config/Config.h>
#include <mesa_pd/data/DataTypes.h>

#include <string>

namespace walberla {
namespace mesa_pd {

struct Parameters
{
   std::string sorting = "none";
   Vec3 normal = Vec3(1.0_r, 1.0_r, 1.0_r);
   real_t spacing = 1.0_r;
   Vec3 shift = Vec3(0.1_r, 0.1_r, 0.1_r);
   real_t radius = 0.5_r;
   bool bBarrier = false;
   bool storeNodeTimings = false;
   bool checkSimulation = false;
   int64_t numOuterIterations = 10;
   int64_t initialRefinementLevel = 0;
   int64_t simulationSteps = 10;
   real_t dt = 0.01_r;
   int64_t visSpacing = 1000;
   std::string vtk_out = "vtk_out";
   std::string sqlFile = "benchmark.sqlite";
   bool recalculateBlockLevelsInRefresh = false;
   bool alwaysRebalanceInRefresh = true;
   bool reevaluateMinTargetLevelsAfterForcedRefinement = false;
   bool allowRefreshChangingDepth = false;
   bool allowMultipleRefreshCycles = false;
   bool checkForEarlyOutInRefresh = true;
   bool checkForLateOutInRefresh = true;
   uint_t regridMin = uint_c(100);
   uint_t regridMax = uint_c(1000);
   int maxBlocksPerProcess = int_c(1000);
   real_t baseWeight = 10.0_r;
   real_t metisipc2redist = 1000.0_r;
   std::string LBAlgorithm = "Hilbert";
   std::string metisAlgorithm = "PART_GEOM_KWAY";
   std::string metisWeightsToUse = "BOTH_WEIGHTS";
   std::string metisEdgeSource = "EDGES_FROM_EDGE_WEIGHTS";
};

void loadFromConfig(Parameters& params,
                    const Config::BlockHandle& cfg);

void saveToSQL(const Parameters& params,
               std::map< std::string, walberla::int64_t >& integerProperties,
               std::map< std::string, double >&            realProperties,
               std::map< std::string, std::string >&       stringProperties );

} //namespace mesa_pd
} //namespace walberla