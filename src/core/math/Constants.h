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
//! \file Constants.h
//! \author Klaus Iglberger
//! \author Sebastian Eibl <sebastian.eibl@fau.de>
//! \author Christoph Schwarzmeier <christoph.schwarzmeier@fau.de>
//! \brief Header file for mathematical constants
//
//======================================================================================================================

#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cmath>
#include <core/DataTypes.h>

namespace walberla
{
namespace math
{
constexpr real_t e                  = 2.7182818284590452354_r;  /* e */
constexpr real_t log2_e             = 1.4426950408889634074_r;  /* log_2 e */
constexpr real_t log10_e            = 0.43429448190325182765_r; /* log_10 e */
constexpr real_t ln_two             = 0.69314718055994530942_r; /* log_e 2 */
constexpr real_t ln_ten             = 2.30258509299404568402_r; /* log_e 10 */
constexpr real_t pi                 = 3.14159265358979323846_r; /* pi */
constexpr real_t half_pi            = 1.57079632679489661923_r; /* pi/2 */
constexpr real_t fourth_pi          = 0.78539816339744830962_r; /* pi/4 */
constexpr real_t one_div_pi         = 0.31830988618379067154_r; /* 1/pi */
constexpr real_t two_div_pi         = 0.63661977236758134308_r; /* 2/pi */
constexpr real_t two_div_root_pi    = 1.12837916709551257390_r; /* 2/sqrt(pi) */
constexpr real_t root_two           = 1.41421356237309504880_r; /* sqrt(2) */
constexpr real_t one_div_root_two   = 0.70710678118654752440_r; /* 1/sqrt(2) */
constexpr real_t root_three         = 1.73205080757_r; /* sqrt(3) */
constexpr real_t one_div_root_three = 0.57735026919_r; /* 1/sqrt(3) */

} // namespace math
} // namespace walberla