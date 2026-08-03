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
//! \file BlockDataHandling.h
//! \ingroup field
//! \author Florian Schornbaum <florian.schornbaum@fau.de>
//
//======================================================================================================================

#pragma once

#include "blockforest/BlockDataHandling.h"
#include "blockforest/StructuredBlockForest.h"
#include "core/debug/CheckFunctions.h"
#include "core/math/Vector2.h"
#include "core/math/Vector3.h"
#include "field/FlagField.h"

#include <type_traits>


namespace walberla {
namespace field {



// still virtual, one must implement protected member functions 'allocate' and 'reallocate'
// Suppressed because waLBerla relies on templated polymorphic interfaces (see Issue 305).
// NOLINTBEGIN(portability-template-virtual-member-function)
template< typename Field_T, bool Pseudo2D = false >
class BlockDataHandling : public blockforest::BlockDataHandling< Field_T >
{
public:

   using Value_T = typename Field_T::value_type;
   using InitializationFunction_T = std::function<void (Field_T *, IBlock *const)>;

   ~BlockDataHandling() override = default;

   void addInitializationFunction( const InitializationFunction_T & initFunction ) { initFunction_ = initFunction; }

   void serializeGhostLayers( const bool serializeGhostLayers ) { serializeGhostLayers_ = serializeGhostLayers; }
   bool serializeGhostLayers() const { return serializeGhostLayers_; }

   Field_T * initialize( IBlock * const block ) override
   {
      auto * field = allocate( block );

      if( initFunction_ )
         initFunction_( field, block );

      return field;
   }

   inline void serialize( IBlock * const block, const BlockDataID & id, mpi::SendBuffer & buffer ) override;

   void serializeCoarseToFine( Block * const block, const BlockDataID & id, mpi::SendBuffer & buffer, const uint_t child ) override;
   void serializeFineToCoarse( Block * const block, const BlockDataID & id, mpi::SendBuffer & buffer ) override;

   Field_T * deserialize( IBlock * const block ) override { return reallocate( block ); }

   Field_T * deserializeCoarseToFine( Block * const block ) override { return reallocate( block ); }
   Field_T * deserializeFineToCoarse( Block * const block ) override { return reallocate( block ); }   

   void deserialize( IBlock * const block, const BlockDataID & id, mpi::RecvBuffer & buffer ) override;

   void deserializeCoarseToFine( Block * const block, const BlockDataID & id, mpi::RecvBuffer & buffer ) override;
   void deserializeFineToCoarse( Block * const block, const BlockDataID & id, mpi::RecvBuffer & buffer, const uint_t child ) override;

protected:

   /// must be thread-safe !
   virtual Field_T *   allocate( IBlock * const block ) = 0; // used in 'initialize'
   /// must be thread-safe !
   virtual Field_T * reallocate( IBlock * const block ) = 0; // used in all deserialize member functions

   template< typename T > struct Merge
   { static T result( const T & value ) { return Pseudo2D ? static_cast<T>( value / numeric_cast<T>(4) ) : static_cast<T>( value / numeric_cast<T>(8) ); } };

   template< typename T > struct Merge< Vector2<T> >
   { static Vector2<T> result( const Vector2<T> & value ) { return Pseudo2D ? (value / numeric_cast<T>(4)) : (value / numeric_cast<T>(8)); } };

   template< typename T > struct Merge< Vector3<T> >
   { static Vector3<T> result( const Vector3<T> & value ) { return Pseudo2D ? (value / numeric_cast<T>(4)) : (value / numeric_cast<T>(8)); } };

   void sizeCheck( const uint_t xSize, const uint_t ySize, const uint_t zSize )
   {
      WALBERLA_CHECK( (xSize & uint_t{1}) == uint_t{0}, "The x-size of your field must be divisible by 2." )
      WALBERLA_CHECK( (ySize & uint_t{1}) == uint_t{0}, "The y-size of your field must be divisible by 2." )
      if( Pseudo2D )
      { WALBERLA_CHECK( zSize == uint_t{1}, "The z-size of your field must be equal to 1 (pseudo 2D mode)." ) }
      else
      { WALBERLA_CHECK( (zSize & uint_t{1}) == uint_t{0}, "The z-size of your field must be divisible by 2." ) }
   }

   InitializationFunction_T initFunction_;

   bool serializeGhostLayers_{ true };

}; // class BlockDataHandling
// NOLINTEND(portability-template-virtual-member-function)



template< typename Field_T, bool Pseudo2D >
inline void BlockDataHandling< Field_T, Pseudo2D >::serialize( IBlock * const block, const BlockDataID & id, mpi::SendBuffer & buffer )
{
   const auto * field = block->template getData< Field_T >(id);
   WALBERLA_ASSERT_NOT_NULLPTR( field )

#ifndef NDEBUG
   buffer << field->xSize() << field->ySize() << field->zSize() << field->fSize();
#endif

   if( serializeGhostLayers_ )
   {
      for( auto it = field->beginWithGhostLayer(); it != field->end(); ++it )
         buffer << *it;
   }
   else
   {
      for( const auto &value : *field )
         buffer << value;
   }
}



template< typename Field_T, bool Pseudo2D >
void BlockDataHandling< Field_T, Pseudo2D >::serializeCoarseToFine( Block * const block, const BlockDataID & id, mpi::SendBuffer & buffer, const uint_t child )
{
   auto * field = block->template getData< Field_T >(id);
   WALBERLA_ASSERT_NOT_NULLPTR( field )

   const cell_idx_t xSize = cell_idx_c( field->xSize() );
   const cell_idx_t ySize = cell_idx_c( field->ySize() );
   const cell_idx_t zSize = cell_idx_c( field->zSize() );
   const uint_t     fSize = field->fSize();
   sizeCheck( field->xSize(), field->ySize(), field->zSize() );

#ifndef NDEBUG
   buffer << child << uint_c( xSize / cell_idx_c(2) ) << uint_c( ySize / cell_idx_c(2) )
          << ( Pseudo2D ? uint_c( zSize ) : uint_c( zSize / cell_idx_c(2) ) ) << fSize;
#endif

   const cell_idx_t gl = serializeGhostLayers_ ? cell_idx_c( field->nrOfGhostLayers() ) : cell_idx_c(0);
   const cell_idx_t cg = ( gl + cell_idx_c(1) ) / cell_idx_c(2);

   const cell_idx_t xHalf = xSize / cell_idx_c(2);
   const cell_idx_t yHalf = ySize / cell_idx_c(2);
   const cell_idx_t zHalf = zSize / cell_idx_c(2);
   const cell_idx_t x0 = (child & uint_t{1}) ? xHalf : cell_idx_c(0);
   const cell_idx_t y0 = (child & uint_t{2}) ? yHalf : cell_idx_c(0);
   const cell_idx_t z0 = (child & uint_t{4}) ? zHalf : cell_idx_c(0);

   const cell_idx_t zBegin = Pseudo2D ? ( -gl )       : ( z0 - cg );
   const cell_idx_t zEnd   = Pseudo2D ? ( zSize + gl ) : ( z0 + zHalf + cg );
   for( cell_idx_t z = zBegin; z < zEnd; ++z )
      for( cell_idx_t y = y0 - cg; y < y0 + yHalf + cg; ++y )
         for( cell_idx_t x = x0 - cg; x < x0 + xHalf + cg; ++x )
            for( uint_t f = 0; f < fSize; ++f )
               buffer << field->get(x,y,z,f);
}



template< typename Field_T, bool Pseudo2D >
void BlockDataHandling< Field_T, Pseudo2D >::serializeFineToCoarse( Block * const block, const BlockDataID & id, mpi::SendBuffer & buffer )
{
   auto * field = block->template getData< Field_T >(id);
   WALBERLA_ASSERT_NOT_NULLPTR( field )

   const cell_idx_t xSize = cell_idx_c( field->xSize() );
   const cell_idx_t ySize = cell_idx_c( field->ySize() );
   const cell_idx_t zSize = cell_idx_c( field->zSize() );
   const uint_t     fSize = field->fSize();
   sizeCheck( field->xSize(), field->ySize(), field->zSize() );

   const uint_t child = block->getId().getBranchId();

#ifndef NDEBUG
   buffer << child << uint_c( xSize / cell_idx_c(2) ) << uint_c( ySize / cell_idx_c(2) )
          << ( Pseudo2D ? uint_c( zSize ) : uint_c( zSize / cell_idx_c(2) ) ) << fSize;
#endif

   const cell_idx_t gl = serializeGhostLayers_ ? cell_idx_c( field->nrOfGhostLayers() ) : cell_idx_c(0);
   const cell_idx_t cg = gl / cell_idx_c(2);

   const cell_idx_t xHalf = xSize / cell_idx_c(2);
   const cell_idx_t yHalf = ySize / cell_idx_c(2);
   const cell_idx_t zHalf = zSize / cell_idx_c(2);
   const cell_idx_t x0 = (child & uint_t{1}) ? xHalf : cell_idx_c(0);
   const cell_idx_t y0 = (child & uint_t{2}) ? yHalf : cell_idx_c(0);
   const cell_idx_t z0 = (child & uint_t{4}) ? zHalf : cell_idx_c(0);

   const cell_idx_t xBegin = (child & uint_t{1}) ? x0 : ( x0 - cg );
   const cell_idx_t xEnd   = (child & uint_t{1}) ? ( x0 + xHalf + cg ) : ( x0 + xHalf );
   const cell_idx_t yBegin = (child & uint_t{2}) ? y0 : ( y0 - cg );
   const cell_idx_t yEnd   = (child & uint_t{2}) ? ( y0 + yHalf + cg ) : ( y0 + yHalf );
   const cell_idx_t zBegin = Pseudo2D ? ( -gl )       : ( (child & uint_t{4}) ? z0 : ( z0 - cg ) );
   const cell_idx_t zEnd   = Pseudo2D ? ( zSize + gl ) : ( (child & uint_t{4}) ? ( z0 + zHalf + cg ) : ( z0 + zHalf ) );

   for( cell_idx_t z = zBegin; z < zEnd; ++z )
   {
      const cell_idx_t fz = Pseudo2D ? z : ( cell_idx_c(2) * ( z - z0 ) );
      for( cell_idx_t y = yBegin; y < yEnd; ++y )
      {
         const cell_idx_t fy = cell_idx_c(2) * ( y - y0 );
         for( cell_idx_t x = xBegin; x < xEnd; ++x )
         {
            const cell_idx_t fx = cell_idx_c(2) * ( x - x0 );
            for( uint_t f = 0; f < fSize; ++f )
            {
               Value_T result =                                  field->get( fx,                 fy,                 fz,                 f );
                       result = static_cast< Value_T >( result + field->get( fx + cell_idx_c(1), fy                , fz                , f ) );
                       result = static_cast< Value_T >( result + field->get( fx                , fy + cell_idx_c(1), fz                , f ) );
                       result = static_cast< Value_T >( result + field->get( fx + cell_idx_c(1), fy + cell_idx_c(1), fz                , f ) );
               if( ! Pseudo2D )
               {
                       result = static_cast< Value_T >( result + field->get( fx                , fy                , fz + cell_idx_c(1), f ) );
                       result = static_cast< Value_T >( result + field->get( fx + cell_idx_c(1), fy                , fz + cell_idx_c(1), f ) );
                       result = static_cast< Value_T >( result + field->get( fx                , fy + cell_idx_c(1), fz + cell_idx_c(1), f ) );
                       result = static_cast< Value_T >( result + field->get( fx + cell_idx_c(1), fy + cell_idx_c(1), fz + cell_idx_c(1), f ) );
               }

               buffer << Merge< Value_T >::result( result );
            }
         }
      }
   }
}



template< typename Field_T, bool Pseudo2D >
inline void BlockDataHandling< Field_T, Pseudo2D >::deserialize( IBlock * const block, const BlockDataID & id, mpi::RecvBuffer & buffer )
{
   auto * field = block->template getData< Field_T >( id );

#ifndef NDEBUG
   uint_t xSender( 0 );
   uint_t ySender( 0 );
   uint_t zSender( 0 );
   uint_t fSender( 0 );
   buffer >> xSender >> ySender >> zSender >> fSender;
   WALBERLA_ASSERT_EQUAL( xSender, field->xSize() )
   WALBERLA_ASSERT_EQUAL( ySender, field->ySize() )
   WALBERLA_ASSERT_EQUAL( zSender, field->zSize() )
   WALBERLA_ASSERT_EQUAL( fSender, field->fSize() )
#endif

   if( serializeGhostLayers_ )
   {
      for( auto it = field->beginWithGhostLayer(); it != field->end(); ++it )
         buffer >> *it;
   }
   else
   {
      for( auto &value : *field )
         buffer >> value;
   }
}



template< typename Field_T, bool Pseudo2D >
void BlockDataHandling< Field_T, Pseudo2D >::deserializeCoarseToFine( Block * const block, const BlockDataID & id, mpi::RecvBuffer & buffer )
{
   auto * field = block->template getData< Field_T >( id );

   const cell_idx_t xSize = cell_idx_c( field->xSize() );
   const cell_idx_t ySize = cell_idx_c( field->ySize() );
   const cell_idx_t zSize = cell_idx_c( field->zSize() );
   const uint_t     fSize = field->fSize();
   sizeCheck( field->xSize(), field->ySize(), field->zSize() );

#ifndef NDEBUG
   uint_t branchId( 0 );
   uint_t xSender( 0 );
   uint_t ySender( 0 );
   uint_t zSender( 0 );
   uint_t fSender( 0 );
   buffer >> branchId >> xSender >> ySender >> zSender >> fSender;
   WALBERLA_ASSERT_EQUAL( branchId, block->getId().getBranchId() )
   WALBERLA_ASSERT_EQUAL( xSender, uint_c( xSize / cell_idx_c(2) ) )
   WALBERLA_ASSERT_EQUAL( ySender, uint_c( ySize / cell_idx_c(2) ) )
   if( Pseudo2D )
   { WALBERLA_ASSERT_EQUAL( zSender, uint_c( zSize ) ) }
   else
   { WALBERLA_ASSERT_EQUAL( zSender, uint_c( zSize / cell_idx_c(2) ) ) }
   WALBERLA_ASSERT_EQUAL( fSender, fSize )
#endif

   const cell_idx_t gl = serializeGhostLayers_ ? cell_idx_c( field->nrOfGhostLayers() ) : cell_idx_c(0);
   const cell_idx_t cg = ( gl + cell_idx_c(1) ) / cell_idx_c(2);

   const cell_idx_t xHalf = xSize / cell_idx_c(2);
   const cell_idx_t yHalf = ySize / cell_idx_c(2);
   const cell_idx_t zHalf = zSize / cell_idx_c(2);

   const cell_idx_t numX = xHalf + cell_idx_c(2) * cg;
   const cell_idx_t numY = yHalf + cell_idx_c(2) * cg;
   const cell_idx_t numZ = Pseudo2D ? ( zSize + cell_idx_c(2) * gl ) : ( zHalf + cell_idx_c(2) * cg );

   const cell_idx_t xLimit = xSize + gl;
   const cell_idx_t yLimit = ySize + gl;
   const cell_idx_t zLimit = zSize + gl;

   for( cell_idx_t iz = 0; iz < numZ; ++iz )
   {
      const cell_idx_t fz0 = Pseudo2D ? ( iz - gl ) : cell_idx_c(2) * ( iz - cg );
      for( cell_idx_t iy = 0; iy < numY; ++iy )
      {
         const cell_idx_t fy0 = cell_idx_c(2) * ( iy - cg );
         for( cell_idx_t ix = 0; ix < numX; ++ix )
         {
            const cell_idx_t fx0 = cell_idx_c(2) * ( ix - cg );
            for( uint_t f = 0; f < fSize; ++f )
            {
               Value_T value;
               buffer >> value;

               for( cell_idx_t dz = 0; dz < ( Pseudo2D ? cell_idx_c(1) : cell_idx_c(2) ); ++dz )
               {
                  const cell_idx_t fz = fz0 + dz;
                  if( fz < -gl || fz >= zLimit ) continue;
                  for( cell_idx_t dy = 0; dy < cell_idx_c(2); ++dy )
                  {
                     const cell_idx_t fy = fy0 + dy;
                     if( fy < -gl || fy >= yLimit ) continue;
                     for( cell_idx_t dx = 0; dx < cell_idx_c(2); ++dx )
                     {
                        const cell_idx_t fx = fx0 + dx;
                        if( fx < -gl || fx >= xLimit ) continue;
                        field->get( fx, fy, fz, f ) = value;
                     }
                  }
               }
            }
         }
      }
   }
}



template< typename Field_T, bool Pseudo2D >
void BlockDataHandling< Field_T, Pseudo2D >::deserializeFineToCoarse( Block * const block, const BlockDataID & id, mpi::RecvBuffer & buffer, const uint_t child )
{
   auto * field = block->template getData< Field_T >( id );

   const cell_idx_t xSize = cell_idx_c( field->xSize() );
   const cell_idx_t ySize = cell_idx_c( field->ySize() );
   const cell_idx_t zSize = cell_idx_c( field->zSize() );
   const uint_t     fSize = field->fSize();
   sizeCheck( field->xSize(), field->ySize(), field->zSize() );

#ifndef NDEBUG
   uint_t branchId( 0 );
   uint_t xSender( 0 );
   uint_t ySender( 0 );
   uint_t zSender( 0 );
   uint_t fSender( 0 );
   buffer >> branchId >> xSender >> ySender >> zSender >> fSender;
   WALBERLA_ASSERT_EQUAL( branchId, child )
   WALBERLA_ASSERT_EQUAL( xSender, uint_c( xSize / cell_idx_c(2) ) )
   WALBERLA_ASSERT_EQUAL( ySender, uint_c( ySize / cell_idx_c(2) ) )
   if( Pseudo2D )
   { WALBERLA_ASSERT_EQUAL( zSender, uint_c( zSize ) ) }
   else
   { WALBERLA_ASSERT_EQUAL( zSender, uint_c( zSize / cell_idx_c(2) ) ) }
   WALBERLA_ASSERT_EQUAL( fSender, fSize )
#endif

   const cell_idx_t gl = serializeGhostLayers_ ? cell_idx_c( field->nrOfGhostLayers() ) : cell_idx_c(0);
   const cell_idx_t cg = gl / cell_idx_c(2);

   const cell_idx_t xHalf = xSize / cell_idx_c(2);
   const cell_idx_t yHalf = ySize / cell_idx_c(2);
   const cell_idx_t zHalf = zSize / cell_idx_c(2);
   const cell_idx_t x0 = (child & uint_t{1}) ? xHalf : cell_idx_c(0);
   const cell_idx_t y0 = (child & uint_t{2}) ? yHalf : cell_idx_c(0);
   const cell_idx_t z0 = (child & uint_t{4}) ? zHalf : cell_idx_c(0);

   const cell_idx_t xBegin = (child & uint_t{1}) ? x0 : ( x0 - cg );
   const cell_idx_t xEnd   = (child & uint_t{1}) ? ( x0 + xHalf + cg ) : ( x0 + xHalf );
   const cell_idx_t yBegin = (child & uint_t{2}) ? y0 : ( y0 - cg );
   const cell_idx_t yEnd   = (child & uint_t{2}) ? ( y0 + yHalf + cg ) : ( y0 + yHalf );
   const cell_idx_t zBegin = Pseudo2D ? ( -gl )        : ( (child & uint_t{4}) ? z0 : ( z0 - cg ) );
   const cell_idx_t zEnd   = Pseudo2D ? ( zSize + gl ) : ( (child & uint_t{4}) ? ( z0 + zHalf + cg ) : ( z0 + zHalf ) );

   for( cell_idx_t z = zBegin; z < zEnd; ++z )
      for( cell_idx_t y = yBegin; y < yEnd; ++y )
         for( cell_idx_t x = xBegin; x < xEnd; ++x )
            for( uint_t f = 0; f < fSize; ++f )
               buffer >> field->get(x,y,z,f);
}






// allocation helper functions used in class 'DefaultBlockDataHandling' (see below)
namespace internal
{

template< typename GhostLayerField_T >
inline GhostLayerField_T * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl,
                                     const typename GhostLayerField_T::value_type & v, Layout l,
                                     const shared_ptr< field::FieldAllocator<typename GhostLayerField_T::value_type> > & alloc=nullptr)
{
   return new GhostLayerField_T(x,y,z,gl,v,l,alloc);
}
template<>
inline FlagField<uint8_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, const uint8_t &, Layout,
                                      const shared_ptr< field::FieldAllocator<uint8_t> > & alloc)
{
   return new FlagField<uint8_t>(x,y,z,gl,alloc);
}
template<>
inline FlagField<uint16_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, const uint16_t &, Layout,
                                       const shared_ptr< field::FieldAllocator<uint16_t> > & alloc)
{
   return new FlagField<uint16_t>(x,y,z,gl,alloc);
}
template<>
inline FlagField<uint32_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, const uint32_t &, Layout,
                                       const shared_ptr< field::FieldAllocator<uint32_t> > & alloc)
{
   return new FlagField<uint32_t>(x,y,z,gl,alloc);
}
template<>
inline FlagField<uint64_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, const uint64_t &, Layout,
                                       const shared_ptr< field::FieldAllocator<uint64_t> > & alloc)
{
   return new FlagField<uint64_t>(x,y,z,gl,alloc);
}

template< typename GhostLayerField_T >
inline GhostLayerField_T * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, Layout l,
                                     const shared_ptr< field::FieldAllocator<typename GhostLayerField_T::value_type> > & alloc=nullptr)
{
   return new GhostLayerField_T(x,y,z,gl,l, alloc);
}
template<>
inline FlagField<uint8_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, Layout,
                                      const shared_ptr< field::FieldAllocator<uint8_t> > & alloc)
{
   return new FlagField<uint8_t>(x,y,z,gl,alloc);
}
template<>
inline FlagField<uint16_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, Layout,
                                       const shared_ptr< field::FieldAllocator<uint16_t> > & alloc)
{
   return new FlagField<uint16_t>(x,y,z,gl,alloc);
}
template<>
inline FlagField<uint32_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, Layout,
                                       const shared_ptr< field::FieldAllocator<uint32_t> > & alloc)
{
   return new FlagField<uint32_t>(x,y,z,gl,alloc);
}
template<>
inline FlagField<uint64_t> * allocate( const uint_t x, const uint_t y, const uint_t z, const uint_t gl, Layout,
                                       const shared_ptr< field::FieldAllocator<uint64_t> > & alloc)
{
   return new FlagField<uint64_t>(x,y,z,gl,alloc);
}

inline Vector3< uint_t > defaultSize( const shared_ptr< StructuredBlockStorage > & blocks, IBlock * const block )
{
   return Vector3<uint_t>( blocks->getNumberOfXCells( *block ), blocks->getNumberOfYCells( *block ), blocks->getNumberOfZCells( *block ) );
}

} // namespace internal



template< typename GhostLayerField_T >
class DefaultBlockDataHandling : public BlockDataHandling< GhostLayerField_T >
{
public:

   using Value_T = typename GhostLayerField_T::value_type;

   DefaultBlockDataHandling( const weak_ptr< StructuredBlockStorage > & blocks,
                             const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) >& calculateSize = internal::defaultSize,
                             const shared_ptr< field::FieldAllocator<Value_T> > alloc = nullptr) :
      blocks_( blocks ), nrOfGhostLayers_( uint_t{1} ), initValue_(), layout_( fzyx ), calculateSize_( calculateSize ), alloc_(alloc)
   {}

   DefaultBlockDataHandling( const weak_ptr< StructuredBlockStorage > & blocks, const uint_t nrOfGhostLayers,
                             const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) >& calculateSize = internal::defaultSize,
                             const shared_ptr< field::FieldAllocator<Value_T> > alloc = nullptr) :
      blocks_( blocks ), nrOfGhostLayers_( nrOfGhostLayers ), initValue_(), layout_( fzyx ), calculateSize_( calculateSize ), alloc_(alloc)
   {}

   DefaultBlockDataHandling( const weak_ptr< StructuredBlockStorage > & blocks, const uint_t nrOfGhostLayers,
                             const Value_T & initValue, const Layout layout = fzyx,
                             const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) >& calculateSize = internal::defaultSize,
                             const shared_ptr< field::FieldAllocator<Value_T> > alloc = nullptr) :
      blocks_( blocks ), nrOfGhostLayers_( nrOfGhostLayers ), initValue_( initValue ), layout_( layout ), calculateSize_( calculateSize ), alloc_(alloc)
   {
      static_assert( !std::is_same_v< GhostLayerField_T, FlagField< Value_T > >,
                     "When using class FlagField, only constructors without the explicit specification of an initial value and the field layout are available!" );
   }

protected:

   GhostLayerField_T * allocate( IBlock * const block ) override
   {
      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( blocks, "Trying to access 'DefaultBlockDataHandling' for a block storage object that doesn't exist anymore" )
      const auto size = calculateSize_( blocks, block );
      return internal::allocate< GhostLayerField_T >( size[0], size[1], size[2],
                                                      nrOfGhostLayers_, initValue_, layout_, alloc_ );
   }

   GhostLayerField_T * reallocate( IBlock * const block ) override
   {
      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( blocks, "Trying to access 'DefaultBlockDataHandling' for a block storage object that doesn't exist anymore" )
      const auto size = calculateSize_( blocks, block );
      return internal::allocate< GhostLayerField_T >( size[0], size[1], size[2],
                                                      nrOfGhostLayers_, layout_, alloc_ );
   }

private:

   weak_ptr< StructuredBlockStorage > blocks_;

   uint_t  nrOfGhostLayers_;
   Value_T initValue_;
   Layout  layout_;
   const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) > calculateSize_;
   const shared_ptr< field::FieldAllocator<Value_T> > alloc_;

}; // class DefaultBlockDataHandling






template< typename GhostLayerField_T >
class AlwaysInitializeBlockDataHandling : public blockforest::AlwaysInitializeBlockDataHandling< GhostLayerField_T >
{
public:

   using Value_T = typename GhostLayerField_T::value_type;
   using InitializationFunction_T = std::function<void (GhostLayerField_T *, IBlock *const)>;

   AlwaysInitializeBlockDataHandling( const weak_ptr< StructuredBlockStorage > & blocks,
                                      const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) >& calculateSize = internal::defaultSize,
                                      const shared_ptr< field::FieldAllocator<Value_T> > alloc = nullptr) :
      blocks_( blocks ), nrOfGhostLayers_( uint_t{1} ), initValue_(), layout_( fzyx ), calculateSize_( calculateSize ), alloc_(alloc)
   {}

   AlwaysInitializeBlockDataHandling( const weak_ptr< StructuredBlockStorage > & blocks, const uint_t nrOfGhostLayers,
                                      const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) >& calculateSize = internal::defaultSize,
                                      const shared_ptr< field::FieldAllocator<Value_T> > alloc = nullptr) :
      blocks_( blocks ), nrOfGhostLayers_( nrOfGhostLayers ), initValue_(), layout_( fzyx ), calculateSize_( calculateSize ), alloc_(alloc)
   {}

   AlwaysInitializeBlockDataHandling( const weak_ptr< StructuredBlockStorage > & blocks, const uint_t nrOfGhostLayers,
                                      const Value_T & initValue, const Layout layout,
                                      const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) >& calculateSize = internal::defaultSize,
                                      const shared_ptr< field::FieldAllocator<Value_T> > alloc = nullptr) :
      blocks_( blocks ), nrOfGhostLayers_( nrOfGhostLayers ), initValue_( initValue ), layout_( layout ), calculateSize_( calculateSize ), alloc_(alloc)
   {
      static_assert( ! std::is_same_v< GhostLayerField_T, FlagField< Value_T > >,
                     "When using class FlagField, only constructors without the explicit specification of an initial value and the field layout are available!" );
   }

   void addInitializationFunction( const InitializationFunction_T & initFunction ) { initFunction_ = initFunction; }

   GhostLayerField_T * initialize( IBlock * const block ) override
   {
      auto blocks = blocks_.lock();
      WALBERLA_CHECK_NOT_NULLPTR( blocks, "Trying to access 'AlwaysInitializeBlockDataHandling' for a block storage object that doesn't exist anymore" )
      const auto size = calculateSize_( blocks, block );
      auto * field = internal::allocate< GhostLayerField_T >( size[0], size[1], size[2],
                                                              nrOfGhostLayers_, initValue_, layout_, alloc_ );
      if( initFunction_ )
         initFunction_( field, block );

      return field;
   }

private:

   weak_ptr< StructuredBlockStorage > blocks_;

   uint_t  nrOfGhostLayers_;
   Value_T initValue_;
   Layout  layout_;
   const std::function< Vector3< uint_t > ( const shared_ptr< StructuredBlockStorage > &, IBlock * const ) > calculateSize_;
   const shared_ptr< field::FieldAllocator<Value_T> > alloc_;

   InitializationFunction_T initFunction_;

}; // class AlwaysInitializeBlockDataHandling






template< typename Field_T >
class CloneBlockDataHandling : public blockforest::AlwaysInitializeBlockDataHandling< Field_T >
{
public:

   CloneBlockDataHandling( const ConstBlockDataID & fieldToClone ) :
      fieldToClone_( fieldToClone )
   {}

   Field_T * initialize( IBlock * const block ) override
   {
      const auto * toClone = block->template getData< Field_T >( fieldToClone_ );
      return toClone->clone();
   }

private:

   ConstBlockDataID fieldToClone_;

}; // class CloneBlockDataHandling






template< typename Field_T >
class FlattenedShallowCopyBlockDataHandling : public blockforest::AlwaysInitializeBlockDataHandling< typename Field_T::FlattenedField >
{
public:

   FlattenedShallowCopyBlockDataHandling( const ConstBlockDataID & fieldToClone ) :
      fieldToClone_( fieldToClone )
   {}

   typename Field_T::FlattenedField * initialize( IBlock * const block ) override
   {
      const auto * toClone = block->template getData< Field_T >( fieldToClone_ );
      return toClone->flattenedShallowCopy();
   }

private:

   ConstBlockDataID fieldToClone_;

}; // class FlattenedShallowCopyBlockDataHandling



} // namespace field
} // namespace walberla
