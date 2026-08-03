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
//! \file D2Q9.h
//! \ingroup lbm
//! \author Florian Schornbaum <florian.schornbaum@fau.de>
//
//======================================================================================================================

#pragma once

#include "LatticeModelBase.h"
#include "stencil/D2Q9.h"

#include <type_traits>


namespace walberla {
namespace lbm {



template< typename CollisionModel_T, bool Compressible = false, typename ForceModel_T = force_model::None, int EquilibriumAccuracyOrder = 2 >
class D2Q9 : public LatticeModelBase< CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder >
{
public:

   static_assert( ! std::is_same_v< CollisionModel_T, collision_model::D3Q19MRT >, "D3Q19MRT only works with D3Q19!" );

   using CollisionModel = typename LatticeModelBase<CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder>::CollisionModel;
   using ForceModel = typename LatticeModelBase<CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder>::ForceModel;

   using Stencil = stencil::D2Q9;
   using CommunicationStencil = stencil::D2Q9;

   static const char * NAME;

   static const real_t w_0;
   static const real_t w_1;
   static const real_t w_2;
   static const real_t w[9];    // Stencil::Size !
   static const real_t wInv[9]; // Stencil::Size !

   D2Q9( const CollisionModel_T & cm, const ForceModel_T & fm ) :
      LatticeModelBase< CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder >( cm, fm ) {}

   // available only if the force model == force_model::None
   D2Q9( const CollisionModel_T & cm ) :
      LatticeModelBase< CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder >( cm, force_model::None() )
   {
      static_assert( std::is_same_v< ForceModel_T, force_model::None >, "This constructor is only available if the force model is equal to force_model::None!" );
   }

   ~D2Q9() override = default;

protected:

   void config( IBlock & /*block*/, StructuredBlockStorage & /*sbs*/ ) override {}
};

template< typename CM, bool C, typename FM, int EAO > const char*  D2Q9<CM,C,FM,EAO>::NAME = "D2Q9";

template< typename CM, bool C, typename FM, int EAO > const real_t D2Q9<CM,C,FM,EAO>::w_0 = 4.0_r / 9.0_r;
template< typename CM, bool C, typename FM, int EAO > const real_t D2Q9<CM,C,FM,EAO>::w_1 = 1.0_r / 9.0_r;
template< typename CM, bool C, typename FM, int EAO > const real_t D2Q9<CM,C,FM,EAO>::w_2 = 1.0_r / 36.0_r;

// must match with the static array 'dir' in stencil::D2Q9
template< typename CM, bool C, typename FM, int EAO > const real_t D2Q9<CM,C,FM,EAO>::w[9] = { 4.0_r / 9.0_r,   // C
                                                                                               1.0_r / 9.0_r,   // N
                                                                                               1.0_r / 9.0_r,   // S
                                                                                               1.0_r / 9.0_r,   // W
                                                                                               1.0_r / 9.0_r,   // E
                                                                                               1.0_r / 36.0_r,   // NW
                                                                                               1.0_r / 36.0_r,   // NE
                                                                                               1.0_r / 36.0_r,   // SW
                                                                                               1.0_r / 36.0_r }; // SE

// must match with the static array 'dir' in stencil::D2Q9
template< typename CM, bool C, typename FM, int EAO > const real_t D2Q9<CM,C,FM,EAO>::wInv[9] = { 9.0_r / 4.0_r,   // C
                                                                                                  9.0_r,                 // N
                                                                                                  9.0_r,                 // S
                                                                                                  9.0_r,                 // W
                                                                                                  9.0_r,                 // E
                                                                                                  36.0_r,                 // NW
                                                                                                  36.0_r,                 // NE
                                                                                                  36.0_r,                 // SW
                                                                                                  36.0_r };               // SE

} // namespace lbm
} // namespace walberla
