//
// Created by dy94rovu on 10/14/25.
//

#include "core/math/all.h"
#include "core/DataTypes.h"

namespace walberla
{
std::vector< math::Vector3< real_t > > generatePositionsSimple(const math::AABB& box, uint_t targetN, real_t minDist,real_t boundarymargin,
                                                               std::mt19937& gen, int maxAttemptsPerPoint = 50000)
{
   const auto mn = box.min();
   const auto mx = box.max();

   // Optional: keep a margin so particles don't sit on the walls.
   const real_t margin = real_t(boundarymargin); // set to particleDiameter*0.5 if you want wall clearance

   std::uniform_real_distribution< real_t > rx(mn[0] + margin, mx[0] - margin);
   std::uniform_real_distribution< real_t > ry(mn[1] + margin, mx[1] - margin);
   std::uniform_real_distribution< real_t > rz(mn[2] + margin, mx[2] - margin);

   const real_t minDist2 = minDist * minDist;

   std::vector< math::Vector3< real_t > > pts;
   pts.reserve(targetN);

   while (pts.size() < targetN)
   {
      bool placed = false;
      for (int attempt = 0; attempt < maxAttemptsPerPoint && !placed; ++attempt)
      {
         math::Vector3< real_t > c(rx(gen), ry(gen), rz(gen));

         // Check distance to all previously accepted points
         bool distancesok = true;
         for (const auto& q : pts)
         {
            real_t dx = c[0] - q[0], dy = c[1] - q[1], dz = c[2] - q[2];
            if (dx * dx + dy * dy + dz * dz < minDist2)
            {
               distancesok = false;
               break;
            }
         }

         if (distancesok)
         {
            pts.push_back(c);
            placed = true;
         }
      }

      if (!placed)
      {
         // Domain likely too tight for this spacing and count
         WALBERLA_ABORT("Failed to place");
         break;
      }
   }

   return pts;
}
}