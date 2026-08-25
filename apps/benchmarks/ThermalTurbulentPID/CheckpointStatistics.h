#pragma once

#include "core/DataTypes.h"
#include "core/logging/all.h"
#include "core/mpi/Broadcast.h"
#include "lbm_mesapd_coupling/DataTypesCodegen.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <system_error>
#include <vector>

namespace MaterialTransport
{


inline void renameCheckpointFile(const std::string& oldName, const std::string& newName)
{
   std::error_code ec;
   std::filesystem::rename(oldName, newName, ec);
   if (ec)
   {
      WALBERLA_LOG_WARNING_ON_ROOT("Could not rename file " << oldName << " to " << newName << " with error code "
                                                            << ec.message());
   }
}

template < typename DomainSize_T, typename VelocityProfiles_T
#ifdef run_with_temperature
         , typename TemperatureProfiles_T
#endif
         >
inline void writeCheckpointStatistics(const DomainSize_T& domainSize, const uint_t currentTimeStep,
                                      const uint_t startTimeStep,
                                      const uint_t checkPointingFrequency,
                                      const std::string& checkpointingFileName,
                                      const uint_t nTurnovers,
                                      const real_t turnOverPeriod,
                                      const VelocityProfiles_T& planeAveragedProfiles_velocity
#ifdef run_with_temperature
                                      , const TemperatureProfiles_T& planeAveragedProfiles_temperature
#endif
)
{
   if (currentTimeStep >=  uint_c(nTurnovers * turnOverPeriod)){
      if (checkPointingFrequency > uint_t(0) && (currentTimeStep % checkPointingFrequency == 0) &&
          currentTimeStep != startTimeStep)
      {
         WALBERLA_ROOT_SECTION()
         {
            std::ofstream checkpointTimeStatsVelocityOS(checkpointingFileName + "_timeStatistics_Velocity_tmp.txt",
                                                        std::ios::out | std::ios::trunc);
            WALBERLA_CHECK(checkpointTimeStatsVelocityOS.is_open(), "Could not open checkpoint config tmp file "
                                                                       << checkpointingFileName +
                                                                             "_timeStatistics_Velocity_tmp.txt"
                                                                       << " for writing");

            checkpointTimeStatsVelocityOS << std::fixed << std::setprecision(6);
            auto printRowVelocity = [&](auto&&... args) {
               ((checkpointTimeStatsVelocityOS << std::setw(12) << args), ...);
               checkpointTimeStatsVelocityOS << "\n";
            };

            printRowVelocity("Uxt_f", "Uyt_f", "Uzt_f", "UUt_f", "UVt_f", "UWt_f", "VUt_f", "VVt_f", "VWt_f",
                             "WUt_f", "WVt_f", "WWt_f", "Uxt_p", "Uyt_p", "Uzt_p", "UUt_p", "UVt_p", "UWt_p",
                             "VUt_p", "VVt_p", "VWt_p", "WUt_p", "WVt_p", "WWt_p");


            for (uint_t idx = 0; idx < domainSize[codegen::wall_axis]; ++idx)
            {
               for (uint_t i = 0; i < codegen::vectorSize; ++i)
               {
                  checkpointTimeStatsVelocityOS << std::setw(12)
                                                << planeAveragedProfiles_velocity->getFluidAVGProfile()[idx * codegen::vectorSize + i];
               }

               for (uint_t i = 0; i < codegen::tensorSize; ++i)
               {
                  checkpointTimeStatsVelocityOS << std::setw(12)
                                                << planeAveragedProfiles_velocity->getFluidAVGSquaredProfile()[idx * codegen::tensorSize + i];
               }

               for (uint_t i = 0; i < codegen::vectorSize; ++i)
               {
                  checkpointTimeStatsVelocityOS << std::setw(12)
                                                << planeAveragedProfiles_velocity->getParticleAVGProfile()[idx * codegen::vectorSize + i];
               }

               for (uint_t i = 0; i < codegen::tensorSize; ++i)
               {
                  checkpointTimeStatsVelocityOS << std::setw(12)
                                                << planeAveragedProfiles_velocity->getParticleAVGSquaredProfile()[idx * codegen::tensorSize + i];
               }

               checkpointTimeStatsVelocityOS << "\n";
            }
            checkpointTimeStatsVelocityOS.flush();
            checkpointTimeStatsVelocityOS.close();

#ifdef run_with_temperature
            std::ofstream checkpointTimeStatsTemperatureOS(checkpointingFileName + "_timeStatistics_Temperature_tmp.txt",
                                                           std::ios::out | std::ios::trunc);
            WALBERLA_CHECK(checkpointTimeStatsTemperatureOS.is_open(), "Could not open checkpoint config tmp file "
                                                                          << checkpointingFileName +
                                                                                "_timeStatistics_Temperature_tmp.txt"
                                                                          << " for writing");

            checkpointTimeStatsTemperatureOS << std::fixed << std::setprecision(6);
            auto printRowTemperature = [&](auto&&... args) {
               ((checkpointTimeStatsTemperatureOS << std::setw(12) << args), ...);
               checkpointTimeStatsTemperatureOS << "\n";
            };

            printRowTemperature("Tt_f", "TTt_f", "Tt_p", "TTt_p");

            for (uint_t idx = 0; idx < domainSize[codegen::wall_axis]; ++idx)
            {
               for (uint_t i = 0; i < codegen::scalarSize; ++i)
               {
                  checkpointTimeStatsTemperatureOS << std::setw(12)
                                                   << planeAveragedProfiles_temperature->getFluidAVGProfile()[idx * codegen::scalarSize + i];
                  checkpointTimeStatsTemperatureOS << std::setw(12)
                                                   << planeAveragedProfiles_temperature->getFluidAVGSquaredProfile()[idx * codegen::scalarSize + i];
                  checkpointTimeStatsTemperatureOS << std::setw(12)
                                                   << planeAveragedProfiles_temperature->getParticleAVGProfile()[idx * codegen::scalarSize + i];
                  checkpointTimeStatsTemperatureOS << std::setw(12)
                                                   << planeAveragedProfiles_temperature->getParticleAVGSquaredProfile()[idx * codegen::scalarSize + i];
                  checkpointTimeStatsTemperatureOS << "\n";
               }
            }
            checkpointTimeStatsTemperatureOS.flush();
            checkpointTimeStatsTemperatureOS.close();
#endif

            std::ofstream checkpointAvgCounterOS(checkpointingFileName + "_averagingCounter_tmp.txt",
                                                 std::ios::out | std::ios::trunc);
            WALBERLA_CHECK(checkpointAvgCounterOS.is_open(), "Could not open checkpoint config tmp file "
                                                                << checkpointingFileName + "_averagingCounter_tmp.txt"
                                                                << " for writing");
            checkpointAvgCounterOS << planeAveragedProfiles_velocity->getTimeStepCount() << "\n";
            checkpointAvgCounterOS.flush();
            checkpointAvgCounterOS.close();

            renameCheckpointFile(checkpointingFileName + "_timeStatistics_Velocity_tmp.txt",
                                 checkpointingFileName + "_timeStatistics_Velocity.txt");
#ifdef run_with_temperature
            renameCheckpointFile(checkpointingFileName + "_timeStatistics_Temperature_tmp.txt",
                                 checkpointingFileName + "_timeStatistics_Temperature.txt");
#endif
            renameCheckpointFile(checkpointingFileName + "_averagingCounter_tmp.txt",
                                 checkpointingFileName + "_averagingCounter.txt");
         }
      }
   }
}

template < typename DomainSize_T >
inline void readCheckpointStatistics(const DomainSize_T& domainSize, const std::string& checkpointingFileName,
                                     std::vector< real_t >& timeAveragedFluidProfile_velocity,
                                     std::vector< real_t >& timeAveragedFluidSquaredProfile_velocity,
                                     std::vector< real_t >& timeAveragedParticleProfile_velocity,
                                     std::vector< real_t >& timeAveragedParticleSquaredProfile_velocity,
                                     uint_t& statsAveragingCounter
#ifdef run_with_temperature
                                     , std::vector< real_t >& timeAveragedFluidProfile_temperature,
                                     std::vector< real_t >& timeAveragedFluidSquaredProfile_temperature,
                                     std::vector< real_t >& timeAveragedParticleProfile_temperature,
                                     std::vector< real_t >& timeAveragedParticleSquaredProfile_temperature
#endif
)
{
   WALBERLA_ROOT_SECTION()
   {
      std::ifstream checkpointTimeStatsVelocityIS(checkpointingFileName + "_timeStatistics_Velocity.txt");
      if (!checkpointTimeStatsVelocityIS.is_open())
      {
         WALBERLA_LOG_WARNING_ON_ROOT("Simulation restart is requested but could not open checkpoint file "
                                      << checkpointingFileName + "_timeStatistics_Velocity.txt");
      }
      else
      {
         std::string ignoredLine;
         std::getline(checkpointTimeStatsVelocityIS, ignoredLine);
         WALBERLA_LOG_INFO_ON_ROOT("parsing the velocity time stats file now");

         for (uint_t idx = 0; idx < domainSize[codegen::wall_axis]; ++idx)
         {
            for (uint_t i = 0; i < codegen::vectorSize; ++i)
            {
               checkpointTimeStatsVelocityIS >> timeAveragedFluidProfile_velocity[idx * codegen::vectorSize + i];
            }

            for (uint_t i = 0; i < codegen::tensorSize; ++i)
            {
               checkpointTimeStatsVelocityIS >> timeAveragedFluidSquaredProfile_velocity[idx * codegen::tensorSize + i];
            }

            for (uint_t i = 0; i < codegen::vectorSize; ++i)
            {
               checkpointTimeStatsVelocityIS >> timeAveragedParticleProfile_velocity[idx * codegen::vectorSize + i];
            }

            for (uint_t i = 0; i < codegen::tensorSize; ++i)
            {
               checkpointTimeStatsVelocityIS >>
                  timeAveragedParticleSquaredProfile_velocity[idx * codegen::tensorSize + i];
            }
         }
         if (!checkpointTimeStatsVelocityIS.good() && !checkpointTimeStatsVelocityIS.eof())
         {
            WALBERLA_ABORT("Could not parse from checkpoint file "
                           << checkpointingFileName + "_timeStatistics_Velocity.txt");
         }
      }

#ifdef run_with_temperature
      std::ifstream checkpointTimeStatsTemperatureIS(checkpointingFileName + "_timeStatistics_Temperature.txt");
      if (!checkpointTimeStatsTemperatureIS.is_open())
      {
         WALBERLA_LOG_WARNING_ON_ROOT("Simulation restart is requested but could not open checkpoint file "
                                      << checkpointingFileName + "_timeStatistics_Temperature.txt");
      }
      else
      {
         std::string ignoredLine;
         std::getline(checkpointTimeStatsTemperatureIS, ignoredLine);
         WALBERLA_LOG_INFO_ON_ROOT("parsing the temperature time stats file now");

         for (uint_t idx = 0; idx < domainSize[codegen::wall_axis]; ++idx)
         {
            for (uint_t i = 0; i < codegen::scalarSize; ++i)
            {
               checkpointTimeStatsTemperatureIS >>
                  timeAveragedFluidProfile_temperature[idx * codegen::scalarSize + i];
               checkpointTimeStatsTemperatureIS >>
                  timeAveragedFluidSquaredProfile_temperature[idx * codegen::scalarSize + i];
               checkpointTimeStatsTemperatureIS >>
                  timeAveragedParticleProfile_temperature[idx * codegen::scalarSize + i];
               checkpointTimeStatsTemperatureIS >>
                  timeAveragedParticleSquaredProfile_temperature[idx * codegen::scalarSize + i];
            }
         }
         if (!checkpointTimeStatsTemperatureIS.good() && !checkpointTimeStatsTemperatureIS.eof())
         {
            WALBERLA_ABORT("Could not parse from checkpoint file "
                           << checkpointingFileName + "_timeStatistics_Temperature.txt");
         }
      }
#endif

      std::ifstream checkpointAvgCounterIS(checkpointingFileName + "_averagingCounter.txt");
      if (!checkpointAvgCounterIS.is_open())
      {
         WALBERLA_LOG_WARNING_ON_ROOT("Could not open checkpoint config file "
                                      << checkpointingFileName + "_averagingCounter.txt");
      }
      else
      {
         checkpointAvgCounterIS >> statsAveragingCounter;
      }
   }

   mpi::broadcastObject(timeAveragedFluidProfile_velocity);
   mpi::broadcastObject(timeAveragedFluidSquaredProfile_velocity);
   mpi::broadcastObject(timeAveragedParticleProfile_velocity);
   mpi::broadcastObject(timeAveragedParticleSquaredProfile_velocity);
   mpi::broadcastObject(statsAveragingCounter);

#ifdef run_with_temperature
   mpi::broadcastObject(timeAveragedFluidProfile_temperature);
   mpi::broadcastObject(timeAveragedFluidSquaredProfile_temperature);
   mpi::broadcastObject(timeAveragedParticleProfile_temperature);
   mpi::broadcastObject(timeAveragedParticleSquaredProfile_temperature);
#endif
}


} // namespace MaterialTransport
