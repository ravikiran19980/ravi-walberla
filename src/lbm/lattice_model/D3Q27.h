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
//! \file D3Q27.h
//! \ingroup lbm
//! \author Florian Schornbaum <florian.schornbaum@fau.de>
//
//======================================================================================================================

#pragma once

#include "LatticeModelBase.h"
#include "stencil/D3Q27.h"

#include <type_traits>


namespace walberla {
namespace lbm {



template< typename CollisionModel_T, bool Compressible = false, typename ForceModel_T = force_model::None, int EquilibriumAccuracyOrder = 2 >
class D3Q27 : public LatticeModelBase< CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder >
{
public:

   static_assert( ! std::is_same_v< CollisionModel_T, collision_model::D3Q19MRT >, "D3Q19MRT only works with D3Q19!" );

   using CollisionModel = typename LatticeModelBase<CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder>::CollisionModel;
   using ForceModel = typename LatticeModelBase<CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder>::ForceModel;

   using Stencil = stencil::D3Q27;
   using CommunicationStencil = stencil::D3Q27;

   static const char * NAME;

   static const real_t w_0;
   static const real_t w_1;
   static const real_t w_2;
   static const real_t w_3;
   static const real_t w[27];    // Stencil::Size !
   static const real_t wInv[27]; // Stencil::Size !

   D3Q27( const CollisionModel_T & cm, const ForceModel_T & fm  ) :
      LatticeModelBase< CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder >( cm, fm ) {}

   // available only if the force model == force_model::None
   D3Q27( const CollisionModel_T & cm ) :
      LatticeModelBase< CollisionModel_T, Compressible, ForceModel_T, EquilibriumAccuracyOrder >( cm, force_model::None() )
   {
      static_assert( std::is_same_v< ForceModel_T, force_model::None >, "This constructor is only available if the force model is equal to force_model::None!" );
   }

   ~D3Q27() override = default;

protected:

   void config( IBlock & /*block*/, StructuredBlockStorage & /*sbs*/ ) override {}
};

template< typename CM, bool C, typename FM, int EAO > const char*  D3Q27<CM,C,FM,EAO>::NAME = "D3Q27";

template< typename CM, bool C, typename FM, int EAO > const real_t D3Q27<CM,C,FM,EAO>::w_0 = 8.0_r / 27.0_r;
template< typename CM, bool C, typename FM, int EAO > const real_t D3Q27<CM,C,FM,EAO>::w_1 = 2.0_r / 27.0_r;
template< typename CM, bool C, typename FM, int EAO > const real_t D3Q27<CM,C,FM,EAO>::w_2 = 1.0_r / 54.0_r;
template< typename CM, bool C, typename FM, int EAO > const real_t D3Q27<CM,C,FM,EAO>::w_3 = 1.0_r / 216.0_r;

// must match with the static array 'dir' in stencil::D3Q27
template< typename CM, bool C, typename FM, int EAO > const real_t D3Q27<CM,C,FM,EAO>::w[27] = { 8.0_r / 27.0_r,   // C
                                                                                                 2.0_r / 27.0_r,   // N
                                                                                                 2.0_r / 27.0_r,   // S
                                                                                                 2.0_r / 27.0_r,   // W
                                                                                                 2.0_r / 27.0_r,   // E
                                                                                                 2.0_r / 27.0_r,   // T
                                                                                                 2.0_r / 27.0_r,   // B
                                                                                                 1.0_r / 54.0_r,   // NW
                                                                                                 1.0_r / 54.0_r,   // NE
                                                                                                 1.0_r / 54.0_r,   // SW
                                                                                                 1.0_r / 54.0_r,   // SE
                                                                                                 1.0_r / 54.0_r,   // TN
                                                                                                 1.0_r / 54.0_r,   // TS
                                                                                                 1.0_r / 54.0_r,   // TW
                                                                                                 1.0_r / 54.0_r,   // TE
                                                                                                 1.0_r / 54.0_r,   // BN
                                                                                                 1.0_r / 54.0_r,   // BS
                                                                                                 1.0_r / 54.0_r,   // BW
                                                                                                 1.0_r / 54.0_r,   // BE
                                                                                                 1.0_r / 216.0_r,   // TNE
                                                                                                 1.0_r / 216.0_r,   // TNW
                                                                                                 1.0_r / 216.0_r,   // TSE
                                                                                                 1.0_r / 216.0_r,   // TSW
                                                                                                 1.0_r / 216.0_r,   // BNE
                                                                                                 1.0_r / 216.0_r,   // BNW
                                                                                                 1.0_r / 216.0_r,   // BSE
                                                                                                 1.0_r / 216.0_r }; // BSW

// must match with the static array 'dir' in stencil::D3Q27
template< typename CM, bool C, typename FM, int EAO > const real_t D3Q27<CM,C,FM,EAO>::wInv[27] = { 27.0_r / 8.0_r,   // C
                                                                                                    27.0_r / 2.0_r,   // N
                                                                                                    27.0_r / 2.0_r,   // S
                                                                                                    27.0_r / 2.0_r,   // W
                                                                                                    27.0_r / 2.0_r,   // E
                                                                                                    27.0_r / 2.0_r,   // T
                                                                                                    27.0_r / 2.0_r,   // B
                                                                                                    54.0_r,                 // NW
                                                                                                    54.0_r,                 // NE
                                                                                                    54.0_r,                 // SW
                                                                                                    54.0_r,                 // SE
                                                                                                    54.0_r,                 // TN
                                                                                                    54.0_r,                 // TS
                                                                                                    54.0_r,                 // TW
                                                                                                    54.0_r,                 // TE
                                                                                                    54.0_r,                 // BN
                                                                                                    54.0_r,                 // BS
                                                                                                    54.0_r,                 // BW
                                                                                                    54.0_r,                 // BE
                                                                                                    216.0_r,                 // TNE
                                                                                                    216.0_r,                 // TNW
                                                                                                    216.0_r,                 // TSE
                                                                                                    216.0_r,                 // TSW
                                                                                                    216.0_r,                 // BNE
                                                                                                    216.0_r,                 // BNW
                                                                                                    216.0_r,                 // BSE
                                                                                                    216.0_r };               // BSW

} // namespace lbm
} // namespace walberla
