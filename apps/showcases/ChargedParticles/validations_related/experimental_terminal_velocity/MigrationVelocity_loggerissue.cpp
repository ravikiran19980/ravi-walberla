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
//! \file ChargedParticles.cpp
//! \ingroup lbm_mesapd_coupling
//! \author Samuel Kemmler <samuel.kemmler@fau.de>
//! \author Christoph Rettinger <christoph.rettinger@fau.de>
//! \brief Modification of showcases/FluidizedBed/FluidizedBedMEM.cpp
//
//======================================================================================================================

#include "blockforest/Initialization.h"
#include "blockforest/communication/UniformBufferedScheme.h"

#include "boundary/all.h"

#include "core/DataTypes.h"
#include "core/Environment.h"
#include "core/debug/Debug.h"
#include "core/grid_generator/SCIterator.h"
#include "core/logging/all.h"
#include "core/math/all.h"
#include "core/mpi/Broadcast.h"
#include "core/timing/RemainingTimeLogger.h"
#include "core/waLBerlaBuildInfo.h"

#include "domain_decomposition/SharedSweep.h"

#include "field/AddToStorage.h"
#include "field/vtk/all.h"

#include "lbm/boundary/all.h"
#include "lbm/communication/PdfFieldPackInfo.h"
#include "lbm/field/AddToStorage.h"
#include "lbm/field/PdfField.h"
#include "lbm/lattice_model/D3Q19.h"
#include "lbm/vtk/all.h"

#include "lbm_mesapd_coupling/DataTypes.h"
#include "lbm_mesapd_coupling/mapping/ParticleMapping.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/PSMSweep.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/PSMUtility.h"
#include "lbm_mesapd_coupling/partially_saturated_cells_method/ParticleAndVolumeFractionMapping.h"
#include "lbm_mesapd_coupling/utility/AddForceOnParticlesKernel.h"
#include "lbm_mesapd_coupling/utility/AddHydrodynamicInteractionKernel.h"
#include "lbm_mesapd_coupling/utility/AverageHydrodynamicForceTorqueKernel.h"
#include "lbm_mesapd_coupling/utility/InitializeHydrodynamicForceTorqueForAveragingKernel.h"
#include "lbm_mesapd_coupling/utility/LubricationCorrectionKernel.h"
#include "lbm_mesapd_coupling/utility/ParticleSelector.h"
#include "lbm_mesapd_coupling/utility/ResetHydrodynamicForceTorqueKernel.h"

#include "mesa_pd/collision_detection/AnalyticContactDetection.h"
#include "mesa_pd/data/DataTypes.h"
#include "mesa_pd/data/ParticleAccessorWithShape.h"
#include "mesa_pd/data/ParticleStorage.h"
#include "mesa_pd/data/ShapeStorage.h"
#include "mesa_pd/data/shape/HalfSpace.h"
#include "mesa_pd/data/shape/Sphere.h"
#include "mesa_pd/domain/BlockForestDataHandling.h"
#include "mesa_pd/domain/BlockForestDomain.h"
#include "mesa_pd/kernel/DoubleCast.h"
#include "mesa_pd/kernel/LinearSpringDashpot.h"
#include "mesa_pd/kernel/SpringDashpot.h"
#include "mesa_pd/kernel/ParticleSelector.h"
#include "mesa_pd/kernel/VelocityVerlet.h"
#include "mesa_pd/mpi/ContactFilter.h"
#include "mesa_pd/mpi/ReduceContactHistory.h"
#include "mesa_pd/mpi/ReduceProperty.h"
#include "mesa_pd/mpi/SyncNextNeighbors.h"
#include "mesa_pd/mpi/notifications/ElectrostaticForceNotification.h"
#include "mesa_pd/mpi/notifications/ForceTorqueNotification.h"
#include "mesa_pd/mpi/notifications/HydrodynamicForceTorqueNotification.h"
#include "mesa_pd/vtk/ParticleVtkOutput.h"

#include "timeloop/SweepTimeloop.h"

#include "vtk/all.h"

#include "AddElectrostaticInteractionKernel.h"
#include "ChargeForce.h"
#include "PoissonSolver.h"
#include "ResetElectrostaticForceKernel.h"
#include "Utility.h"
#include "ErrorNorms.h"
#include "BoundaryValues.h"
#include "FunctionBoundary.h"

namespace charged_particles
{
///////////
// USING //
///////////

    using namespace walberla;
    using walberla::uint_t;
    using LatticeModel_T = lbm::D3Q19< lbm::collision_model::TRT >;

using Stencil_T  = LatticeModel_T::Stencil;
using PdfField_T = lbm::PdfField< LatticeModel_T >;

using flag_t      = walberla::uint8_t;
using FlagField_T = FlagField< flag_t >;

using ScalarField_T = GhostLayerField< real_t, 1 >;
using VectorField_T = GhostLayerField< real_t, 3 >;

const uint_t FieldGhostLayers = 1;


///////////
// FLAGS //
///////////

    const FlagUID Fluid_Flag("fluid");
    const FlagUID NoSlip_Flag("no slip");
    const FlagUID FreeSlip_Flag("free slip");
    const FlagUID Inflow_Flag("inflow");
    const FlagUID Outflow_Flag("outflow");

/////////////////////////////////////
// BOUNDARY HANDLING CUSTOMIZATION //
/////////////////////////////////////


    template< typename ParticleAccessor_T>
    class MyBoundaryHandling {
    public:
        using NoSlip_T = lbm::NoSlip<LatticeModel_T, flag_t>;
        using Type = BoundaryHandling<FlagField_T, Stencil_T, NoSlip_T>;



        MyBoundaryHandling(const BlockDataID &flagFieldID, const BlockDataID &pdfFieldID,
                           const shared_ptr<ParticleAccessor_T> &ac)
                : flagFieldID_(flagFieldID), pdfFieldID_(pdfFieldID), ac_(ac){}


        Type *operator()(IBlock *const block, const StructuredBlockStorage *const storage) const {
            WALBERLA_ASSERT_NOT_NULLPTR(block);
            WALBERLA_ASSERT_NOT_NULLPTR(storage);

            auto *flagField = block->getData<FlagField_T>(flagFieldID_);
            auto *pdfField = block->getData<PdfField_T>(pdfFieldID_);

            const auto fluid =
                    flagField->flagExists(Fluid_Flag) ? flagField->getFlag(Fluid_Flag) : flagField->registerFlag(
                            Fluid_Flag);

            Type* handling =
                    new Type("moving obstacle boundary handling", flagField, fluid, NoSlip_T("NoSlip", NoSlip_Flag, pdfField));


            handling->fillWithDomain(FieldGhostLayers);

            return handling;
        }

    private:
        const BlockDataID flagFieldID_;
        const BlockDataID pdfFieldID_;

        shared_ptr<ParticleAccessor_T> ac_;



    };


    template< typename ParticleAccessor_T >
    class SpherePropertyLogger
    {
    public:
        SpherePropertyLogger(const shared_ptr< ParticleAccessor_T >& ac, walberla::id_t sphereUid,
                             const std::string& fileName, bool fileIO, real_t dx_SI, real_t dt_SI, real_t diameter,
                             real_t gravitationalForceMag)
                : ac_(ac), sphereUid_(sphereUid), fileName_(fileName), fileIO_(fileIO), dx_SI_(dx_SI), dt_SI_(dt_SI),
                  diameter_(diameter), gravitationalForceMag_(gravitationalForceMag), position_(real_t(0)),
                  maxVelocity_(real_t(0))
        {
            if (fileIO_)
            {
                WALBERLA_ROOT_SECTION()
                {
                    std::ofstream file;
                    file.open(fileName_.c_str());
                    file << "#\t t\t posX\t posY\t gapZ\t velX\t velY\t velZ\n";
                    file.close();
                }
            }
        }

        void operator()(const uint_t timestep)
        {
            Vector3< real_t > pos(real_t(0));
            Vector3< real_t > transVel(real_t(0));
            Vector3< real_t > hydForce(real_t(0));

            size_t idx = ac_->uidToIdx(sphereUid_);
            if (idx != ac_->getInvalidIdx())
            {
                if (!mesa_pd::data::particle_flags::isSet(ac_->getFlags(idx), mesa_pd::data::particle_flags::GHOST))
                {
                    pos      = ac_->getPosition(idx);
                    transVel = ac_->getLinearVelocity(idx);
                    hydForce = ac_->getHydrodynamicForce(idx);
                }
            }

            WALBERLA_MPI_SECTION()
            {
                mpi::allReduceInplace(pos, mpi::SUM);
                mpi::allReduceInplace(transVel, mpi::SUM);
                mpi::allReduceInplace(hydForce, mpi::SUM);
            }

            position_    = pos[2];
            maxVelocity_ = std::max(maxVelocity_, -transVel[2]);

            if (fileIO_) writeToFile(timestep, pos, transVel, hydForce);
        }

        real_t getPosition() const { return position_; }

        real_t getMaxVelocity() const { return maxVelocity_; }

    private:
        void writeToFile(const uint_t timestep, const Vector3< real_t >& position, const Vector3< real_t >& velocity,
                         const Vector3< real_t >& hydForce)
        {
            WALBERLA_ROOT_SECTION()
            {
                std::ofstream file;
                file.open(fileName_.c_str(), std::ofstream::app);

                auto scaledPosition     = position / diameter_;
                auto velocity_SI        = velocity * dx_SI_ / dt_SI_;
                auto normalizedHydForce = hydForce / gravitationalForceMag_;

                file << timestep << "\t" << real_c(timestep) * dt_SI_ << "\t"
                     << "\t" << scaledPosition[0] << "\t" << scaledPosition[1] << "\t" << scaledPosition[2] - real_t(0.5)
                     << "\t" << velocity_SI[0] << "\t" << velocity_SI[1] << "\t" << velocity_SI[2] << "\t"
                     << normalizedHydForce[0] << "\t" << normalizedHydForce[1] << "\t" << normalizedHydForce[2] << "\n";
                file.close();
            }
        }

        shared_ptr< ParticleAccessor_T > ac_;
        const walberla::id_t sphereUid_;
        std::string fileName_;
        bool fileIO_;
        real_t dx_SI_, dt_SI_, diameter_, gravitationalForceMag_;

        real_t position_;
        real_t maxVelocity_;
    };





//*******************************************************************************************************************

    void createPlaneSetup(const shared_ptr< mesa_pd::data::ParticleStorage >& ps,
                          const shared_ptr< mesa_pd::data::ShapeStorage >& ss, const math::AABB& simulationDomain)
    {
        // create bounding planes
        mesa_pd::data::Particle p0 = *ps->create(true);
        p0.setPosition(simulationDomain.minCorner());
        p0.setInteractionRadius(std::numeric_limits< real_t >::infinity());
        p0.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 0, 1)));
        p0.setOwner(mpi::MPIManager::instance()->rank());
        p0.setType(0);
        mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
        mesa_pd::data::particle_flags::set(p0.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

        mesa_pd::data::Particle p1 = *ps->create(true);
        p1.setPosition(simulationDomain.maxCorner());
        p1.setInteractionRadius(std::numeric_limits< real_t >::infinity());
        p1.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 0, -1)));
        p1.setOwner(mpi::MPIManager::instance()->rank());
        p1.setType(0);
        mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
        mesa_pd::data::particle_flags::set(p1.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

        mesa_pd::data::Particle p2 = *ps->create(true);
        p2.setPosition(simulationDomain.minCorner());
        p2.setInteractionRadius(std::numeric_limits< real_t >::infinity());
        p2.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(1, 0, 0)));
        p2.setOwner(mpi::MPIManager::instance()->rank());
        p2.setType(0);
        mesa_pd::data::particle_flags::set(p2.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
        mesa_pd::data::particle_flags::set(p2.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

        mesa_pd::data::Particle p3 = *ps->create(true);
        p3.setPosition(simulationDomain.maxCorner());
        p3.setInteractionRadius(std::numeric_limits< real_t >::infinity());
        p3.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(-1, 0, 0)));
        p3.setOwner(mpi::MPIManager::instance()->rank());
        p3.setType(0);
        mesa_pd::data::particle_flags::set(p3.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
        mesa_pd::data::particle_flags::set(p3.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

        mesa_pd::data::Particle p4 = *ps->create(true);
        p4.setPosition(simulationDomain.minCorner());
        p4.setInteractionRadius(std::numeric_limits< real_t >::infinity());
        p4.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, 1, 0)));
        p4.setOwner(mpi::MPIManager::instance()->rank());
        p4.setType(0);
        mesa_pd::data::particle_flags::set(p4.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
        mesa_pd::data::particle_flags::set(p4.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);

        mesa_pd::data::Particle p5 = *ps->create(true);
        p5.setPosition(simulationDomain.maxCorner());
        p5.setInteractionRadius(std::numeric_limits< real_t >::infinity());
        p5.setShapeID(ss->create< mesa_pd::data::HalfSpace >(Vector3< real_t >(0, -1, 0)));
        p5.setOwner(mpi::MPIManager::instance()->rank());
        p5.setType(0);
        mesa_pd::data::particle_flags::set(p5.getFlagsRef(), mesa_pd::data::particle_flags::INFINITE);
        mesa_pd::data::particle_flags::set(p5.getFlagsRef(), mesa_pd::data::particle_flags::FIXED);
    }

    struct ParticleInfo
    {
        real_t averageVelocity = 0_r;
        real_t maximumVelocity = 0_r;
        uint_t numParticles    = 0;
        real_t maximumHeight   = 0_r;
        real_t particleVolume  = 0_r;
        real_t heightOfMass    = 0_r;

        void allReduce()
        {
            walberla::mpi::allReduceInplace(numParticles, walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(averageVelocity, walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(maximumVelocity, walberla::mpi::MAX);
            walberla::mpi::allReduceInplace(maximumHeight, walberla::mpi::MAX);
            walberla::mpi::allReduceInplace(particleVolume, walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(heightOfMass, walberla::mpi::SUM);

            averageVelocity /= real_c(numParticles);
            heightOfMass /= particleVolume;
        }
    };

    std::ostream& operator<<(std::ostream& os, ParticleInfo const& m)
    {
        return os << "Particle Info: uAvg = " << m.averageVelocity << ", uMax = " << m.maximumVelocity
                  << ", numParticles = " << m.numParticles << ", zMax = " << m.maximumHeight << ", Vp = " << m.particleVolume
                  << ", zMass = " << m.heightOfMass;
    }

    template< typename Accessor_T >
    ParticleInfo evaluateParticleInfo(const Accessor_T& ac)
    {
        static_assert(std::is_base_of< mesa_pd::data::IAccessor, Accessor_T >::value, "Provide a valid accessor");

        ParticleInfo info;
        for (uint_t i = 0; i < ac.size(); ++i)
        {
            if (isSet(ac.getFlags(i), mesa_pd::data::particle_flags::GHOST)) continue;
            if (isSet(ac.getFlags(i), mesa_pd::data::particle_flags::GLOBAL)) continue;

            ++info.numParticles;
            real_t velMagnitude   = ac.getLinearVelocity(i).length();
            real_t particleVolume = ac.getShape(i)->getVolume();
            real_t height         = ac.getPosition(i)[2];
            info.averageVelocity += velMagnitude;
            info.maximumVelocity = std::max(info.maximumVelocity, velMagnitude);
            info.maximumHeight   = std::max(info.maximumHeight, height);
            info.particleVolume += particleVolume;
            info.heightOfMass += particleVolume * height;
        }

        info.allReduce();

        return info;
    }

    struct FluidInfo
    {
        uint_t numFluidCells   = 0;
        real_t averageVelocity = 0_r;
        real_t maximumVelocity = 0_r;
        real_t averageDensity  = 0_r;
        real_t maximumDensity  = 0_r;

        void allReduce()
        {
            walberla::mpi::allReduceInplace(numFluidCells, walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(averageVelocity, walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(maximumVelocity, walberla::mpi::MAX);
            ;
            walberla::mpi::allReduceInplace(averageDensity, walberla::mpi::SUM);
            walberla::mpi::allReduceInplace(maximumDensity, walberla::mpi::MAX);

            averageVelocity /= real_c(numFluidCells);
            averageDensity /= real_c(numFluidCells);
        }
    };

    std::ostream& operator<<(std::ostream& os, FluidInfo const& m)
    {
        return os << "Fluid Info: numFluidCells = " << m.numFluidCells << ", uAvg = " << m.averageVelocity
                  << ", uMax = " << m.maximumVelocity << ", densityAvg = " << m.averageDensity
                  << ", densityMax = " << m.maximumDensity;
    }

    template< typename BoundaryHandling_T >
    FluidInfo evaluateFluidInfo(const shared_ptr< StructuredBlockStorage >& blocks, const BlockDataID& pdfFieldID,
                                const BlockDataID& boundaryHandlingID)
    {
        FluidInfo info;

        for (auto blockIt = blocks->begin(); blockIt != blocks->end(); ++blockIt)
        {
            auto pdfField         = blockIt->getData< PdfField_T >(pdfFieldID);
            auto boundaryHandling = blockIt->getData< BoundaryHandling_T >(boundaryHandlingID);

            WALBERLA_FOR_ALL_CELLS_XYZ(
                    pdfField, if (!boundaryHandling->isDomain(x, y, z)) continue; ++info.numFluidCells;
                    Vector3< real_t > velocity(0_r); real_t density = pdfField->getDensityAndVelocity(velocity, x, y, z);
                    real_t velMagnitude                             = velocity.length(); info.averageVelocity += velMagnitude;
                    info.maximumVelocity = std::max(info.maximumVelocity, velMagnitude); info.averageDensity += density;
                    info.maximumDensity  = std::max(info.maximumDensity, density);)
        }
        info.allReduce();
        return info;
    }


//////////
// MAIN //
//////////

//*******************************************************************************************************************
/*!\brief Basic simulation of a fluidization setup
 *
 * Initially, the mono-sized sphere are created on a structured grid inside the domain.
 * The domain is either periodic or bounded by walls in the horizontal directions (x and y).
 * In z-direction, a constant inflow from below is provided
 * and a pressure boundary condition is set at the top, resembling an outflow boundary.
 *
 * The simulation is run for the given number of seconds (runtime).
 *
 * All parameters should be set via the input file.
 *
 * For the overall algorithm and the different model parameters, see
 * Rettinger, Rüde - An efficient four-way coupled lattice Boltzmann - discrete element method for
 * fully resolved simulations of particle-laden flows (2020, preprint: https://arxiv.org/abs/2003.01490)
 *
 */
//*******************************************************************************************************************
    int main(int argc, char** argv)
    {
        Environment env(argc, argv);

        auto cfgFile = env.config();
        if (!cfgFile) { WALBERLA_ABORT("Usage: " << argv[0] << " path-to-configuration-file \n"); }

        WALBERLA_LOG_INFO_ON_ROOT("waLBerla revision: " << std::string(WALBERLA_GIT_SHA1).substr(0, 8));
        WALBERLA_LOG_INFO_ON_ROOT(*cfgFile);

        // read all parameters from the config file

        Config::BlockHandle physicalSetup         = cfgFile->getBlock("PhysicalSetup");
        const real_t xSize_SI                     = physicalSetup.getParameter< real_t >("xSize");
        const real_t ySize_SI                     = physicalSetup.getParameter< real_t >("ySize");
        const real_t zSize_SI                     = physicalSetup.getParameter< real_t >("zSize");
        const bool periodicInX                    = physicalSetup.getParameter< bool >("periodicInX");
        const bool periodicInY                    = physicalSetup.getParameter< bool >("periodicInY");
        const real_t expectedSettlingVelocity_SI                   = physicalSetup.getParameter< real_t >("uInflow");
        const real_t gravitationalAcceleration_SI = physicalSetup.getParameter< real_t >("gravitationalAcceleration");
        const real_t dynamicViscosityFluid_SI     =   physicalSetup.getParameter< real_t >("DynamicViscosity");
        WALBERLA_LOG_INFO_ON_ROOT("dynamic viccs si" << " " << dynamicViscosityFluid_SI<< " " << gravitationalAcceleration_SI << " "<<expectedSettlingVelocity_SI);
        const real_t densityFluid_SI              = physicalSetup.getParameter< real_t >("densityFluid");
        const real_t particleDiameter_SI          = physicalSetup.getParameter< real_t >("particleDiameter");
        const real_t densityParticle_SI           = physicalSetup.getParameter< real_t >("densityParticle");
        const real_t dynamicFrictionCoefficient   = physicalSetup.getParameter< real_t >("dynamicFrictionCoefficient");
        const real_t coefficientOfRestitution     = physicalSetup.getParameter< real_t >("coefficientOfRestitution");
        const real_t particleGenerationSpacingx_SI = physicalSetup.getParameter< real_t >("particleGenerationSpacing_x");
        const real_t particleGenerationSpacingy_SI = physicalSetup.getParameter< real_t >("particleGenerationSpacing_y");
        const real_t particleGenerationSpacingz_SI = physicalSetup.getParameter< real_t >("particleGenerationSpacing_z");


        Config::BlockHandle numericalSetup     = cfgFile->getBlock("NumericalSetup");

        const real_t expectedSettlingVelocity                  = numericalSetup.getParameter< real_t >("uInflow");
        const uint_t numXBlocks                = numericalSetup.getParameter< uint_t >("numXBlocks");
        const uint_t numYBlocks                = numericalSetup.getParameter< uint_t >("numYBlocks");
        const uint_t numZBlocks                = numericalSetup.getParameter< uint_t >("numZBlocks");
        const bool useLubricationForces        = numericalSetup.getParameter< bool >("useLubricationForces");
        const uint_t numberOfParticleSubCycles = numericalSetup.getParameter< uint_t >("numberOfParticleSubCycles");


        // For boundaries in paramter file -> begin


        Config::BlockHandle IntegratorSchemes = cfgFile->getBlock("IntegratorSchemes");
        int on;
        on = IntegratorSchemes.getParameter<int>("ON");
        bool useIntegrator = true;
        if(on ==0){useIntegrator = false;}


        Config::BlockHandle Boundarytypes = cfgFile->getBlock("boundarytypes");
        std::string Northboundarytype     = Boundarytypes.getParameter< std::string >("North_type");
        std::string Southboundarytype     = Boundarytypes.getParameter< std::string >("South_type");
        std::string Westboundarytype      = Boundarytypes.getParameter< std::string >("West_type");
        std::string Eastboundarytype      = Boundarytypes.getParameter< std::string >("East_type");
        std::string Topboundarytype       = Boundarytypes.getParameter< std::string >("Top_type");
        std::string Bottomboundarytype    = Boundarytypes.getParameter< std::string >("Bottom_type");

        Config::BlockHandle ParticleCharges = cfgFile->getBlock("ParticleCharges");
        real_t maxCharge_SI           = ParticleCharges.getParameter< real_t >("maxCharge");
        real_t minCharge_SI           = ParticleCharges.getParameter< real_t >("minCharge");

        Config::BlockHandle Boundaryvalues  = cfgFile->getBlock("boundaryvalues");
        const real_t Northboundaryvalue_SI  = Boundaryvalues.getParameter< real_t >("North_value");
        const real_t Southboundaryvalue_SI  = Boundaryvalues.getParameter< real_t >("South_value");
        const real_t Westboundaryvalue_SI   = Boundaryvalues.getParameter< real_t >("West_value");
        const real_t Eastboundaryvalue_SI   = Boundaryvalues.getParameter< real_t >("East_value");
        const real_t Topboundaryvalue_SI    = Boundaryvalues.getParameter< real_t >("Top_value");
        const real_t Bottomboundaryvalue_SI = Boundaryvalues.getParameter< real_t >("Bottom_value");

        const real_t V = 1; // conversion factor for potential. Potentail in SI units = Potential in LBM units

        const real_t Northboundaryvalue  = V * Northboundaryvalue_SI;
        const real_t Southboundaryvalue  = V * Southboundaryvalue_SI;
        const real_t Westboundaryvalue   = V * Westboundaryvalue_SI;
        const real_t Eastboundaryvalue   = V * Eastboundaryvalue_SI;
        const real_t Bottomboundaryvalue = V * Bottomboundaryvalue_SI;
        const real_t Topboundaryvalue    = V * Topboundaryvalue_SI;


        BoundaryCondition north(stencil::Direction::N, Northboundarytype, Northboundaryvalue);
        BoundaryCondition south(stencil::Direction::S, Southboundarytype, Southboundaryvalue);
        BoundaryCondition west(stencil::Direction::W, Westboundarytype, Westboundaryvalue);
        BoundaryCondition east(stencil::Direction::E, Eastboundarytype, Eastboundaryvalue);
        BoundaryCondition top(stencil::Direction::T, Topboundarytype, Topboundaryvalue);
        BoundaryCondition bottom(stencil::Direction::B, Bottomboundarytype, Bottomboundaryvalue);

        std::vector< BoundaryCondition > boundaryconditions; // this will be passed into the poissonsolver as an argument
        boundaryconditions.push_back(north);
        boundaryconditions.push_back(south);
        boundaryconditions.push_back(west);
        boundaryconditions.push_back(east);
        boundaryconditions.push_back(top);
        boundaryconditions.push_back(bottom);

        // For boundaries in paramter file -> end

        // convert SI units to simulation (LBM) units and check setup

        uint_t numberOfCellsInHorizontalDirection = uint_t(135);

        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--resolution") == 0) {
                numberOfCellsInHorizontalDirection = uint_c(std::atof(argv[++i]));
                continue;
            }
        }

        const real_t kinematicViscosityFluid_SI = dynamicViscosityFluid_SI / densityFluid_SI;
        Vector3< real_t > domainSize_SI(real_t(100e-3), real_t(100e-3), real_t(160e-3));
        const real_t startingGapSize_SI = real_t(120e-3) + real_t(0.25) * particleDiameter_SI;

        WALBERLA_LOG_INFO_ON_ROOT("Setup (in SI units):");
        WALBERLA_LOG_INFO_ON_ROOT(" - domain size = " << domainSize_SI);
        WALBERLA_LOG_INFO_ON_ROOT(" - sphere: diameter = " << particleDiameter_SI << ", density = " << densityParticle_SI
                                          << ", starting gap size = " << startingGapSize_SI);
        WALBERLA_LOG_INFO_ON_ROOT(" - fluid: density = " << densityFluid_SI << ", dyn. visc = " << dynamicViscosityFluid_SI
                                                         << ", kin. visc = " << kinematicViscosityFluid_SI);
        WALBERLA_LOG_INFO_ON_ROOT(" - expected settling velocity = "
                                          << expectedSettlingVelocity_SI << " --> Re_p = "
                                          << expectedSettlingVelocity_SI * particleDiameter_SI / kinematicViscosityFluid_SI);



        //////////////////////////
        // NUMERICAL PARAMETERS //
        //////////////////////////

        const real_t dx_SI = domainSize_SI[0] / real_c(numberOfCellsInHorizontalDirection);
        const Vector3< uint_t > domainSize(uint_c(floor(domainSize_SI[0] / dx_SI + real_t(0.5))),
                                           uint_c(floor(domainSize_SI[1] / dx_SI + real_t(0.5))),
                                           uint_c(floor(domainSize_SI[2] / dx_SI + real_t(0.5))));
        const real_t diameter     = particleDiameter_SI / dx_SI;
        const real_t sphereVolume = math::pi / real_t(6) * diameter * diameter * diameter;

        const real_t dt_SI                    = expectedSettlingVelocity / expectedSettlingVelocity_SI * dx_SI;

        const real_t viscosity      = kinematicViscosityFluid_SI * dt_SI / (dx_SI * dx_SI);
        const real_t relaxationTime = real_t(1) / lbm::collision_model::omegaFromViscosity(viscosity);

        const real_t gravitationalAcceleration = gravitationalAcceleration_SI * dt_SI * dt_SI / dx_SI;

        const real_t densityFluid  = real_t(1);
        const real_t densitySphere = densityFluid * densityParticle_SI / densityFluid_SI;

        const real_t dx = real_t(1);

        const uint_t timesteps = uint_t(250000);

        const uint_t vtkSpacingParticles = 100;
        const uint_t vtkSpacingFluid     = 100;
        const uint_t infoSpacing         = 10;
        std::string vtkFolder = "vtk_out";



        // ravi implementation for vacuum permitivity :
        // vacuum permitivity -> (vacuum_permitivity = 8.8541878128 x 10-12 ) (εe = εr ε0)
        // dielectric permitivity = (relative_permitivity_medium)*(vacuum_permitivity)
        // relative_permitivity = 78 and vacuum_permitivity SI units ->  A2⋅s4⋅kg−1⋅m−3

        const real_t vacuum_permitivity_SI = 78 * 8.8541878128 * pow(10, -12);
        const real_t Ampere_unit           = ((densityFluid_SI) * (pow(dx_SI, 5))) / (pow(dt_SI, 3) * V);
        real_t vacuum_permitivity =
                ((densityFluid_SI * (pow(dx_SI, 3))) * (pow(dx_SI, 3))) / ((pow(dt_SI, 4)) * Ampere_unit*Ampere_unit);

        vacuum_permitivity = vacuum_permitivity * (vacuum_permitivity_SI);


        // now for the charge: read from the paramter file
        const real_t elementaryCharge = 1.60217663 * pow(10, -19); // 1 elementary charge = 1.60217663 * pow(10, -19) coloumbs
        maxCharge_SI = maxCharge_SI*elementaryCharge;
        minCharge_SI = minCharge_SI*elementaryCharge;
        const real_t maxCharge        = ((maxCharge_SI) * (V * dt_SI * dt_SI)) / (densityFluid_SI * pow(dx_SI, 5));
        const real_t minCharge        = ((minCharge_SI) * (V * dt_SI * dt_SI)) / (densityFluid_SI * pow(dx_SI, 5));


        //std::cout << "permitivity_LBM units:" <<" " << vacuum_permitivity;
        //std::cout << "charge in LBM units:" << " " << maxCharge << std::endl;

        // this is just for verification of the SI to LBM conversions is correctly done or not
        const real_t q_unit = (densityFluid_SI * pow(dx_SI, 5))/(V * dt_SI * dt_SI);
        const real_t epsilon_unit = ((pow(dt_SI, 4)) * Ampere_unit*Ampere_unit)/((densityFluid_SI * (pow(dx_SI, 3))) * (pow(dx_SI, 3)));
        const real_t potenial_unit = q_unit/(epsilon_unit*dx_SI);

        //std:: cout << "Potential_unit" << " " << potenial_unit << std::endl;
        //std::cout << "domain size" << " " << domainSize[0] << std::endl;
        //std::cout << "potential at boundary in  Lattice units"<< " " <<(1/(4*3.14*vacuum_permitivity))*(maxCharge/(domainSize[0]/2)) << std::endl;
        //std::cout << "potential at boundary in  SI units"<< " " << (1/(4*3.14*vacuum_permitivity*epsilon_unit))*((maxCharge*q_unit)/(domainSize[0]/2*dx_SI)) << std::endl;
        //std::cout << "potential at boundary in  SI units"<< " " << (1/(4*3.14*vacuum_permitivity_SI))*((maxCharge_SI)/(domainSize[0]/2*dx_SI)) << std::endl;

        // verification ends here

        WALBERLA_LOG_INFO_ON_ROOT(" - dx_SI = " << dx_SI << ", dt_SI = " << dt_SI);
        WALBERLA_LOG_INFO_ON_ROOT("Setup (in simulation, i.e. lattice, units):");
        WALBERLA_LOG_INFO_ON_ROOT(" - domain size = " << domainSize);
        WALBERLA_LOG_INFO_ON_ROOT(" - sphere: diameter LBM = " << diameter << ", density = " << densitySphere);
        WALBERLA_LOG_INFO_ON_ROOT(" - fluid: density = " << densityFluid << ", relaxation time (tau) = " << relaxationTime
                                                         << ", kin. visc = " << viscosity);
        WALBERLA_LOG_INFO_ON_ROOT(" - gravitational acceleration = " << gravitationalAcceleration);
        WALBERLA_LOG_INFO_ON_ROOT(" - expected settling velocity = " << expectedSettlingVelocity << " --> Re_p = "
                                                                     << expectedSettlingVelocity * diameter / viscosity);


        const real_t poissonsRatio         = real_t(0.22);
        const real_t kappa                 = real_t(2) * (real_t(1) - poissonsRatio) / (real_t(2) - poissonsRatio);
        const real_t particleCollisionTime = 4_r * diameter;



        ///////////////////////////
        // BLOCK STRUCTURE SETUP //
        ///////////////////////////

        const bool periodicInZ                     = false;
        Vector3< uint_t > numberOfBlocksPerDirection(uint_t(1), uint_t(1), uint_t(4));
        Vector3< uint_t > cellsPerBlockPerDirection(domainSize[0] / numberOfBlocksPerDirection[0],
                                                    domainSize[1] / numberOfBlocksPerDirection[1],
                                                    domainSize[2] / numberOfBlocksPerDirection[2]);

        for (uint_t i = 0; i < 3; ++i)
        {
            WALBERLA_CHECK_EQUAL(cellsPerBlockPerDirection[i] * numberOfBlocksPerDirection[i], domainSize[i],
                                 "Unmatching domain decomposition in direction " << i << "!");
        }

        auto blocks = blockforest::createUniformBlockGrid(numberOfBlocksPerDirection[0], numberOfBlocksPerDirection[1],
                                                          numberOfBlocksPerDirection[2], cellsPerBlockPerDirection[0],
                                                          cellsPerBlockPerDirection[1], cellsPerBlockPerDirection[2], dx, 0,
                                                          false, false, false, false, false, // periodicity
                                                          false);

        WALBERLA_LOG_INFO_ON_ROOT("Domain decomposition:");
        WALBERLA_LOG_INFO_ON_ROOT(" - blocks per direction = " << numberOfBlocksPerDirection);
        WALBERLA_LOG_INFO_ON_ROOT(" - cells per block = " << cellsPerBlockPerDirection);


        auto simulationDomain = blocks->getDomain();


        //////////////////
        // RPD COUPLING //
        //////////////////

        auto rpdDomain = std::make_shared< mesa_pd::domain::BlockForestDomain >(blocks->getBlockForestPointer());

        // init data structures
        auto ps                  = walberla::make_shared< mesa_pd::data::ParticleStorage >(1);
        auto ss                  = walberla::make_shared< mesa_pd::data::ShapeStorage >();
        using ParticleAccessor_T = mesa_pd::data::ParticleAccessorWithShape;
        auto accessor            = walberla::make_shared< ParticleAccessor_T >(ps, ss);

        // prevent particles from interfering with inflow and outflow by putting the bounding planes slightly in front
        const real_t planeOffsetFromInflow  = 0;//dx;
        const real_t planeOffsetFromOutflow = 0;//dx;
        createPlaneSetup(ps, ss, simulationDomain);

        // create sphere and store Uid
        Vector3< real_t > initialPosition(real_t(0.5) * real_c(domainSize[0]), real_t(0.5) * real_c(domainSize[1]),
                                          startingGapSize_SI / dx_SI + real_t(0.5) * diameter);
        auto sphereShape = ss->create< mesa_pd::data::Sphere >(diameter * real_t(0.5));
        ss->shapes[sphereShape]->updateMassAndInertia(densitySphere);

        walberla::id_t sphereUid = 0;
        if (rpdDomain->isContainedInProcessSubdomain(uint_c(mpi::MPIManager::instance()->rank()), initialPosition))
        {
            mesa_pd::data::Particle&& p = *ps->create();
            p.setPosition(initialPosition);
            p.setInteractionRadius(diameter * real_t(0.5));
            p.setOwner(mpi::MPIManager::instance()->rank());
            p.setShapeID(sphereShape);
            sphereUid = p.getUid();
        }
        mpi::allReduceInplace(sphereUid, mpi::SUM);


        ///////////////////////
        // ADD DATA TO BLOCKS //
        ////////////////////////

        LatticeModel_T latticeModel =
                LatticeModel_T(lbm::collision_model::TRT::constructWithMagicNumber(real_t(1) / relaxationTime));


        // add PDF field
        BlockDataID pdfFieldID = lbm::addPdfFieldToStorage< LatticeModel_T >(blocks, "pdf field", latticeModel, Vector3< real_t >(real_t(0)),
                                                                             densityFluid, uint_t(1), field::fzyx);

        // add flag field
        BlockDataID flagFieldID = field::addFlagFieldToStorage< FlagField_T >(blocks, "flag field");

        // add boundary handling
        using BoundaryHandling_T       = MyBoundaryHandling< ParticleAccessor_T >::Type;
        BlockDataID boundaryHandlingID = blocks->addStructuredBlockData< BoundaryHandling_T >(
                MyBoundaryHandling< ParticleAccessor_T >(flagFieldID, pdfFieldID, accessor), "boundary handling");

        // set up RPD functionality
        std::function< void(void) > syncCall = [ps, rpdDomain]() {
            const real_t overlap = real_t(1.5);
            mesa_pd::mpi::SyncNextNeighbors syncNextNeighborFunc;
            syncNextNeighborFunc(*ps, *rpdDomain, overlap);
        };


        syncCall();


        mesa_pd::kernel::VelocityVerletPreForceUpdate vvIntegratorPreForce(real_t(1) / real_t(numberOfParticleSubCycles));
        mesa_pd::kernel::VelocityVerletPostForceUpdate vvIntegratorPostForce(real_t(1) / real_t(numberOfParticleSubCycles));

        mesa_pd::kernel::SpringDashpot collisionResponse(1);
        mesa_pd::mpi::ReduceProperty reduceProperty;

        // set up coupling functionality
        lbm_mesapd_coupling::RegularParticlesSelector sphereSelector;
        Vector3< real_t > gravitationalForce(real_t(0), real_t(0),
                                             -(densitySphere - densityFluid) * gravitationalAcceleration * sphereVolume);
        lbm_mesapd_coupling::AddForceOnParticlesKernel addGravitationalForce(gravitationalForce);
        lbm_mesapd_coupling::AddHydrodynamicInteractionKernel addHydrodynamicInteraction;
        lbm_mesapd_coupling::ResetHydrodynamicForceTorqueKernel resetHydrodynamicForceTorque;
        lbm_mesapd_coupling::AverageHydrodynamicForceTorqueKernel averageHydrodynamicForceTorque;
        lbm_mesapd_coupling::LubricationCorrectionKernel lubricationCorrectionKernel(
                viscosity, [](real_t r) { return real_t(0.0016) * r; });

        lbm_mesapd_coupling::ParticleMappingKernel< BoundaryHandling_T > particleMappingKernel(blocks, boundaryHandlingID);


        ///////////////
        // TIME LOOP //
        ///////////////

        // map particles into the LBM simulation
        // note: planes are not mapped and are thus only visible to the particles, not to the fluid
        // instead, the respective boundary conditions for the fluid are explicitly set, see the boundary handling


        // map planes into the LBM simulation -> act as no-slip boundaries
        ps->forEachParticle(false, lbm_mesapd_coupling::GlobalParticlesSelector(), *accessor, particleMappingKernel,
                            *accessor, NoSlip_Flag);

        BlockDataID particleAndVolumeFractionFieldID =
                field::addToStorage< lbm_mesapd_coupling::psm::ParticleAndVolumeFractionField_T >(
                        blocks, "particle and volume fraction field",
                        std::vector< lbm_mesapd_coupling::psm::ParticleAndVolumeFraction_T >(), field::fzyx, 0);

        lbm_mesapd_coupling::psm::ParticleAndVolumeFractionMapping particleMapping(
                blocks, accessor, lbm_mesapd_coupling::RegularParticlesSelector(), particleAndVolumeFractionFieldID, 4);
        particleMapping();

        lbm_mesapd_coupling::psm::initializeDomainForPSM< LatticeModel_T, 1 >(*blocks, pdfFieldID,
                                                                              particleAndVolumeFractionFieldID, *accessor);


        std::function< void() > commFunction;
        blockforest::communication::UniformBufferedScheme< Stencil_T > scheme(blocks);
        scheme.addPackInfo(make_shared< lbm::PdfFieldPackInfo< LatticeModel_T > >(pdfFieldID));
        commFunction = scheme;

        // create the timeloop
        SweepTimeloop timeloop(blocks->getBlockStorage(), timesteps);
        auto bhSweep  = BoundaryHandling_T::getBlockSweep(boundaryHandlingID);
        auto lbmSweep = lbm_mesapd_coupling::psm::makePSMSweep< LatticeModel_T, FlagField_T, 1, 1 >(
                pdfFieldID, particleAndVolumeFractionFieldID, blocks, accessor, flagFieldID, Fluid_Flag);

        timeloop.addFuncBeforeTimeStep(RemainingTimeLogger(timeloop.getNrOfTimeSteps()), "Remaining Time Logger");
        timeloop.add() << BeforeFunction(commFunction, "LBM Communication") << Sweep(bhSweep, "Boundary Handling");

        // stream + collide LBM step
        timeloop.add() << Sweep(makeSharedSweep(lbmSweep), "cell-wise LB sweep");

        // evaluation functionality
        std::string baseFolder = "vtk_out_SettlingSphere";

        std::string loggingFileName(baseFolder + "/LoggingSettlingSphere_");
        uint_t fluidType = 1;
        loggingFileName += std::to_string(fluidType);
        bool fileIO            = false;
        loggingFileName += ".txt";
        if (fileIO) { WALBERLA_LOG_INFO_ON_ROOT(" - writing logging output to file \"" << loggingFileName << "\""); }
        SpherePropertyLogger< ParticleAccessor_T > logger(accessor, sphereUid, loggingFileName, fileIO, dx_SI, dt_SI,
                                                          diameter, -gravitationalForce[2]);


        ////////////////////////
        // EXECUTE SIMULATION //
        ////////////////////////

        WcTimingPool timeloopTiming;
        const bool useOpenMP = false;
        real_t terminationPosition = real_t(0.51) * diameter;
        //WALBERLA_LOG_INFO_ON_ROOT("termination pos:" << terminationPosition);


        // time loop
        for (uint_t timeStep = 0; timeStep < timesteps; ++timeStep)
        {


            // perform a single simulation step -> this contains LBM and setting of the hydrodynamic interactions
            timeloop.singleStep(timeloopTiming);
            timeloopTiming["RPD"].start();
            reduceProperty.operator()< mesa_pd::HydrodynamicForceTorqueNotification >(*ps);

            // TODO: wrap into subcycle loop if stability issues occur
            // TODO: add charge and electrostatic force variables to particles and add ElectrostaticForceNotification, see
            // python/mesa_pd.py  ---> Completed

            //walberla::charged_particles::computeChargeDensity(blocks, particleAndVolumeFractionFieldID, chargeDensityFieldID,
            //                                                  accessor,vacuum_permitivity);

            // Solve poisson equation to obtain electric potential

            //poissonSolver();

            // resetting electrostatic force

            //ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectAll(), *accessor, resetElectrostaticForce, *accessor);

            // Compute electrostatic force field from electric potential (using finite differences)
            //chargeForceUpdate();

            // TODO: reduce electrostatic force field components into corresponding particles, see
            // src/lbm_mesapd_coupling/partially_saturated_cells_method/PSMSweep.h for how to reduce a force field into the
            // corresponding particles  --> completed

            //reduceProperty.operator()< mesa_pd::ElectrostaticForceNotification >(*ps);
            // TODO: add ElectrostaticForceNotification for reduction between the blocks, see above  --> completed

            if (timeStep == 0)
            {
                lbm_mesapd_coupling::InitializeHydrodynamicForceTorqueForAveragingKernel
                        initializeHydrodynamicForceTorqueForAveragingKernel;
                ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor,
                                    initializeHydrodynamicForceTorqueForAveragingKernel, *accessor);
            }
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, averageHydrodynamicForceTorque,
                                *accessor);

            for (auto subCycle = uint_t(0); subCycle < numberOfParticleSubCycles; ++subCycle)
            {


                if(useIntegrator) {
                    ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPreForce, *accessor);
                    syncCall();
                }


                ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, addHydrodynamicInteraction,
                                    *accessor);
                ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, addGravitationalForce, *accessor);

                if (useLubricationForces)
                {
                    // lubrication correction
                    ps->forEachParticlePairHalf(
                            useOpenMP, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
                            [&lubricationCorrectionKernel, &rpdDomain](const size_t idx1, const size_t idx2, auto& ac) {
                                mesa_pd::collision_detection::AnalyticContactDetection acd;
                                acd.getContactThreshold() = lubricationCorrectionKernel.getNormalCutOffDistance();
                                mesa_pd::kernel::DoubleCast double_cast;
                                mesa_pd::mpi::ContactFilter contact_filter;
                                if (double_cast(idx1, idx2, ac, acd, ac))
                                {
                                    if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                                    {
                                        double_cast(acd.getIdx1(), acd.getIdx2(), ac, lubricationCorrectionKernel, ac,
                                                    acd.getContactNormal(), acd.getPenetrationDepth());
                                    }
                                }
                            },
                            *accessor);
                }

                // collision response
                ps->forEachParticlePairHalf(
                        useOpenMP, mesa_pd::kernel::ExcludeInfiniteInfinite(), *accessor,
                        [collisionResponse, rpdDomain](const size_t idx1, const size_t idx2, auto& ac) {
                            mesa_pd::collision_detection::AnalyticContactDetection acd;
                            mesa_pd::kernel::DoubleCast double_cast;
                            mesa_pd::mpi::ContactFilter contact_filter;
                            if (double_cast(idx1, idx2, ac, acd, ac))
                            {
                                if (contact_filter(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), *rpdDomain))
                                {
                                    collisionResponse(acd.getIdx1(), acd.getIdx2(), ac, acd.getContactPoint(), acd.getContactNormal(),
                                                      acd.getPenetrationDepth());
                                }
                            }
                        },
                        *accessor);


                reduceProperty.operator()< mesa_pd::ForceTorqueNotification >(*ps);

                if(useIntegrator) {
                    ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectLocal(), *accessor, vvIntegratorPostForce,
                                        *accessor);
                }

                syncCall();
                particleMapping();


            }
            timeloopTiming["RPD"].end();

            timeloopTiming["Logging"].start();
            logger(timeStep);
            timeloopTiming["Logging"].end();
            ps->forEachParticle(useOpenMP, mesa_pd::kernel::SelectAll(), *accessor, resetHydrodynamicForceTorque, *accessor);

            // TODO: write and add resetElectrostaticForce, see above   ---> completed
            auto particleInfo = evaluateParticleInfo(*accessor);
            auto fluidInfo = evaluateFluidInfo< BoundaryHandling_T >(blocks, pdfFieldID, boundaryHandlingID);

            WALBERLA_LOG_INFO_ON_ROOT("sphere posiiton is:" <<  particleInfo.heightOfMass << "velocity is:"<<" " << particleInfo.maximumVelocity);
            // check for termination

            if (logger.getPosition() < terminationPosition)
            {
                WALBERLA_LOG_INFO_ON_ROOT("Sphere reached terminal position " << logger.getPosition() << " after "
                                                                              << " timesteps!");
                break;
            }




        }


        timeloopTiming.logResultOnRoot();
        real_t relErr = std::fabs(expectedSettlingVelocity - logger.getMaxVelocity()) / expectedSettlingVelocity;
        WALBERLA_LOG_INFO_ON_ROOT("Expected maximum settling velocity: " << expectedSettlingVelocity);
        WALBERLA_LOG_INFO_ON_ROOT("Simulated maximum settling velocity: " << logger.getMaxVelocity());
        WALBERLA_LOG_INFO_ON_ROOT("Relative error: " << relErr);




        return EXIT_SUCCESS;
    }

} // namespace charged_particles

int main(int argc, char** argv) { charged_particles::main(argc, argv); }
