
/*
  TODO: Explicit space() method?
        Explicit shape() and dtype() methods?
          Can't use any dynamic allocation with IPC, so "Object _params" is out
          Limit max no. dims to say 10
  TODO: Work out final pieces to make RingBuffer inter-process shareable
          See IPCObject.hpp
          Need ring-wide shutdown event triggered by destructor?

  TODO: Need shutdown() method for writers (due to guaranteed readers blocking open_write)?
 */

#include "strided_iterator.hpp"
#include "utils.hpp"
#include "event_flag.hpp"
#include "StackMultiset.hpp"
#include "StackVector.hpp"
#include "Mutex.hpp"
#include "Memory.hpp"

#include <cuda_runtime_api.h>

#include <cassert>
#include <algorithm>
#include <limits>
#include <atomic>

//#include <unistd.h> // For usleep (testing only)

template<typename T> class RingReader;
template<typename T> class RingWriter;
template<typename T> class RingWriteBlock;

template<typename T=uint8_t,
         int MAX_READERS=64,
         int MAX_SHAPE_DIMS=10,
         int MAX_DTYPE_LEN=64>
class RingBufferImpl {
	typedef pthread::Mutex             mutex_type;
	typedef LockGuard<mutex_type>      lock_guard_type;
	typedef UniqueLock<mutex_type>     unique_lock_type;
	typedef pthread::ConditionVariable condition_type;
public:
	typedef T                 value_type;
	typedef value_type*       pointer;
	typedef value_type const* const_pointer;
	typedef size_t            size_type;
	//typedef ssize_t           size_type;
	typedef ssize_t           offset_type;
	// TODO: Make these two private and expose via iterators/generic container?
	typedef StackVector<size_type, MAX_SHAPE_DIMS> shape_type;
	typedef StackVector<char,      MAX_SHAPE_DIMS> dtype_type;
	class ShutdownError : public std::runtime_error {
		typedef std::runtime_error super_t;
	public:
		ShutdownError(std::string const& what_arg) : super_t(what_arg) {}
	};
public:
	RingBufferImpl(space_type space=SPACE_SYSTEM);
	RingBufferImpl(RingBufferImpl const& ) = delete;
	RingBufferImpl& operator=(RingBufferImpl const& ) = delete;
	~RingBufferImpl();
#if __cplusplus >= 201103L
	RingBufferImpl(RingBufferImpl&& );
	RingBufferImpl& operator=(RingBufferImpl&& );
#endif
	void request_size(size_type gulp_size, size_type buffer_size);
	// Note: This is not used internally, it's just for convenience
	template<typename Iterator>
	void set_shape(Iterator first, Iterator last) {
		lock_guard_type lock(_mutex);
		_shape.assign(first, last);
	}
	template<typename Container,
	         typename Sfinae=typename Container::const_iterator>
	void set_shape(Container const& shape) {
		this->set_shape(shape.begin(), shape.end());
	}
	void set_shape(std::vector<size_type> shape) {
		this->set_shape(shape.begin(), shape.end());
	}
	void set_shape(size_type size) {
		this->set_shape(std::vector<size_type>(1, size));
	}
	// WARNING: Must call this before performing any allocations
	void set_space(space_type space) {
		_space = space;
	}
	inline size_type     size()         const { return _size; }
	inline size_type     capacity()     const {
		lock_guard_type lock(_mutex);
		return _size + _ghost_size;
	}
	inline size_type     max_gulp_size() const { return std::min(size(), _ghost_size); }
	inline offset_type   head()         const { return _head; }
	inline offset_type   reserve_head() const { return _reserve_head; }
	inline offset_type   tail()         const { return _tail; }
	inline pointer       data()               { return _buf; }
	inline const_pointer data()         const { return _buf; }
	inline space_type    space()        const { return _space; }
	//inline shape_type    shape()        const {
	//	lock_guard_type lock(_mutex);
	//	return _shape;
	//}
	inline std::vector<size_t> shape() const {
		lock_guard_type lock(_mutex);
		return std::vector<size_t>(_shape.begin(), _shape.end());
	}
	//inline dtype_type  dtype()    const {
	//	lock_guard_type lock(_mutex);
	//	return _dtype;
	//}
private:
	offset_type buf_offset( offset_type offset) const;
	pointer     buf_pointer(offset_type offset) const;
	template<typename U>
	friend class RingReader;
	void notify_of_shutdown() {
		_read_condition.notify_all();
	}
	// TODO: Any more elegant way to define these methods? Bit of a mess.
	pointer open_read_at(offset_type offset, size_type size, bool guarantee,
	                     event_flag const& shutdown_event);
	pointer open_read_at_head(offset_type& offset,
	                          size_type size, bool guarantee,
	                          event_flag const& shutdown_event);
	pointer open_read_at_tail(offset_type& offset,
	                          size_type size, bool guarantee,
	                          event_flag const& shutdown_event);
	void  close_read(     offset_type offset, size_type size, bool guaranteed);
	//bool still_valid(offset_type offset, size_type size) const {
	bool still_valid(offset_type offset) const {
		return (offset >= _tail);
	}
	pointer advance_read_by(offset_type& offset, size_type size, bool guaranteed,
	                        size_type advance,
	                        event_flag const& shutdown_event);
	pointer _open_read_at(offset_type& offset, size_type size,
	                      bool guarantee,
	                      event_flag const& shutdown_event,
	                      unique_lock_type& lock,
	                      bool at_tail=false);
	void  _close_read(offset_type offset, size_type size, bool guaranteed);
	
	template<typename U>
	friend class RingWriteBlock;
	template<typename U>
	friend class RingWriter;
	// WARNING: Must be called before any open_* calls
	// TODO: This complicates open_read if allowed at any time
	//         E.g., a waiting reader would need to be informed of the reset
	//           This functionality could be added (say via a ResetError), but
	//             wait until there is a solid use-case for it.
	void init_write(offset_type offset) {
		unique_lock_type lock(_mutex);
		assert( _nwrite_open == 0 );
		assert( _guarantees.empty() );
		_write_condition.wait(lock, [&](){
				return (_nrealloc_pending == 0);
			});
		//assert( _nrealloc_pending == 0 );
		/*
		_write_condition.wait(lock, [&](){
				return ((_guarantees.empty() ||
				         *_guarantees.begin() >= offset) &&
				        _nrealloc_pending == 0);
			});
		*/
		//std::cout << "RingBuffer::init_write offset = " << offset << std::endl;
		_head         = offset;
		_tail         = head();
		_reserve_head = head();
		_buf_offset0  = head();
		_ghost_dirty  = false;
	}
	pointer open_write(size_type size, offset_type& offset);
	void   close_write(offset_type offset, size_type size);
	
	void _ghost_write(offset_type offset, size_type size);
	void _ghost_read( offset_type offset, size_type size);
	void _copy_to_ghost(  offset_type buf_offset, size_type size);
	void _copy_from_ghost(offset_type buf_offset, size_type size);
	
	// Note: Readable frames lie in the range of absolute offsets [_tail,_head)
	std::atomic<offset_type> _tail;
	std::atomic<offset_type> _head;
	// Note: Frames reserved by writers lie in the range [_head, _reserve_head)
	std::atomic<offset_type> _reserve_head;
	std::atomic<  size_type> _size;
	//std::atomic<  size_type> _capacity;
	pointer                  _buf;
	offset_type              _buf_offset0;
	space_type               _space;
	
	typedef std::pair<offset_type,size_type> guarantee_value_type;
	typedef StackMultiset<guarantee_value_type, MAX_READERS> guarantee_map;
	typedef typename guarantee_map::iterator guarantee_iterator;
	guarantee_map            _guarantees;
	
	size_type _ghost_size;
	bool      _ghost_dirty;
	
	shape_type               _shape;
	//dtype_type               _dtype;
	
	mutable mutex_type       _mutex;
	condition_type              _read_condition;
	condition_type             _write_condition;
	condition_type           _realloc_condition;
	condition_type       _write_close_condition;
	size_type                _nread_open;
	size_type                _nwrite_open;
	size_type                _nrealloc_pending;
};

template<class T, int A, int B, int C>
RingBufferImpl<T,A,B,C>::RingBufferImpl(space_type space)
	: _tail(0), _head(0), _reserve_head(0),
	  _size(0),
	  _buf(0), _buf_offset0(0), _space(space),
	  _ghost_size(0), _ghost_dirty(false),
	  _nread_open(0), _nwrite_open(0), _nrealloc_pending(0) {}
template<class T, int A, int B, int C>
RingBufferImpl<T,A,B,C>::~RingBufferImpl() {
	if( _buf ) {
		deallocate(_buf, _space);
	}
}
// TODO: Round buffer_size up to multiple of 256 bytes for alignment?
//         Safe to do this?
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::request_size(size_type gulp_size,
                                           size_type buffer_size) {
	unique_lock_type lock(_mutex);
	if(   gulp_size <= _ghost_size &&
	    buffer_size <= _size ) {
		return;
	}
	++_nrealloc_pending;
	_realloc_condition.wait(lock, [&](){
			return (_nwrite_open == 0 &&
			         _nread_open == 0);
		});
	if( !(  gulp_size <= _ghost_size &&
	      buffer_size <= _size) ) {
		// Must reallocate (everything)
		size_type new_size       = std::max(this->size(), buffer_size);
		size_type new_ghost_size = std::max(_ghost_size, gulp_size);
		size_type new_capacity   = new_size + new_ghost_size;
		pointer   new_buf        = allocate<T>(new_capacity, _space);
		if( _buf ) { // Deal with existing buffer
			if( buf_offset(_tail) < buf_offset(_head) ) {
				// Copy middle to beginning
				copy(_buf + buf_offset(_tail),
				     _buf + buf_offset(_head),
				     new_buf);
				_buf_offset0 = _tail;
			}
			else {
				// Copy beg to beg and end to end, with larger gap between
				copy(_buf,
				     _buf + buf_offset(_head),
				     new_buf);
				copy(_buf + buf_offset(_tail),
				     _buf + _size,
				     new_buf + (buf_offset(_tail)+(new_size-_size)));
				_buf_offset0 = _head - buf_offset(_head);
			}
			// Copy old ghost region to new buffer
			copy(_buf + _size,
			     _buf + _size + _ghost_size,
			     new_buf + new_size);
			// Copy the part of the beg corresponding to the extra ghost space
			//std::cout << "_size:          " << _size << std::endl;
			//std::cout << "new_size:       " << new_size << std::endl;
			//std::cout << "_ghost_size:    " << _ghost_size << std::endl;
			//std::cout << "new_ghost_size: " << new_ghost_size << std::endl;
			//std::cout << "this->size():   " << this->size() << std::endl;
			copy(_buf + _ghost_size,
			     _buf + std::min(new_ghost_size, this->size()),
			     new_buf + new_size + _ghost_size);
			// TODO: Is this the right thing to do?
			_ghost_dirty = true;
			// Free the old buffer
			deallocate(_buf, _space);
			_buf        = 0;
			_size       = 0;
			_ghost_size = 0;
		}
		_buf        = new_buf;
		_size       = new_size;
		_ghost_size = new_ghost_size;
	}
	--_nrealloc_pending;
	_read_condition.notify_all();
	_write_condition.notify_all();
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::pointer
RingBufferImpl<T,A,B,C>::open_read_at(offset_type offset, size_type size,
                                  bool guarantee,
                                  event_flag const& shutdown_event) {
	unique_lock_type lock(_mutex);
	return this->_open_read_at(offset, size, guarantee, shutdown_event, lock);
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::pointer
RingBufferImpl<T,A,B,C>::open_read_at_head(offset_type& offset,
                                           size_type size, bool guarantee,
                                           event_flag const& shutdown_event) {
	unique_lock_type lock(_mutex);
	offset = _head;
	return this->_open_read_at(offset, size, guarantee, shutdown_event, lock);
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::pointer
RingBufferImpl<T,A,B,C>::open_read_at_tail(offset_type& offset,
                                           size_type size, bool guarantee,
                                           event_flag const& shutdown_event) {
	unique_lock_type lock(_mutex);
	return this->_open_read_at(offset, size, guarantee, shutdown_event, lock,
	                           true);
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::pointer
RingBufferImpl<T,A,B,C>::_open_read_at(offset_type& offset, size_type size,
                                       bool guarantee,
                                       event_flag const& shutdown_event,
                                       unique_lock_type& lock,
                                       bool at_tail) {
	// TODO: Including !guarantee here is not a great idea, but is needed
	//         because _tail is advanced in open_write _before_ the writer
	//         waits for the guarantee.
	if( !at_tail && !guarantee && offset < (offset_type)_tail ) {
		std::cout << offset << " < " << (offset_type)_tail << std::endl;
		throw std::out_of_range("Requested read offset has been overwritten");
	}
	guarantee_iterator guarantee_iter = _guarantees.end();
	if( at_tail ) {
		_read_condition.wait(lock, [&](){
				return ((_head - _tail >= size &&
				         _nrealloc_pending == 0) ||
				        shutdown_event.is_set());
			});
		offset = _tail;
		if( guarantee ) {
			//std::cout << "GUARANTEE INSERTED at tail: " << offset << std::endl;
			guarantee_iter = _guarantees.insert(std::make_pair(offset, size));
		}
	}
	else {
		if( guarantee ) {
			//std::cout << "GUARANTEE INSERTED at " << offset << std::endl;
			guarantee_iter = _guarantees.insert(std::make_pair(offset, size));
		}
		_read_condition.wait(lock, [&](){
				return ((_head >= offset + size &&
				         _nrealloc_pending == 0) ||
				        shutdown_event.is_set());
			});
	}
	// Note: This must be done here because a shutdown will trigger close_read
	++_nread_open;
	if( shutdown_event.is_set() ) {
		if( !at_tail && guarantee ) {
			_guarantees.erase(guarantee_iter);
		}
		throw ShutdownError("Shutdown during RingBufferImpl::open_read_at");
	}
	this->_ghost_read(offset, size);
	return buf_pointer(offset);
}
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::close_read(offset_type offset, size_type size,
                                         bool guaranteed) {
	lock_guard_type lock(_mutex);
	return this->_close_read(offset, size, guaranteed);
}
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::_close_read(offset_type offset, size_type size,
                                          bool guaranteed) {
	assert( _nread_open > 0 );
	if( guaranteed ) {
		_guarantees.erase(_guarantees.find(std::make_pair(offset, size)));
		_write_condition.notify_all();
	}
	--_nread_open;
	_realloc_condition.notify_all();
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::pointer
RingBufferImpl<T,A,B,C>::advance_read_by(offset_type& offset, size_type size,
                                         bool guarantee,
                                         size_type advance,
                                         event_flag const& shutdown_event) {
	unique_lock_type lock(_mutex);
	this->_close_read(offset, size, guarantee);
	offset += advance;
	return this->_open_read_at(offset, size, guarantee, shutdown_event, lock);
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::pointer
RingBufferImpl<T,A,B,C>::open_write(size_type size, offset_type& offset) {
	unique_lock_type lock(_mutex);
	//std::cout << "RingBuffer::open_write reserve_head = " << reserve_head() << std::endl;
	offset = _reserve_head;
	// Push the write head
	_reserve_head += size;
	// Pull the tail along
	// Note: This may cause still_valid() to return false, but the write
	//         condition will still wait for guaranteed reads before
	//         allowing data to actually be overwritten. This also means
	//         that guaranteed reads cannot "cover for" unguaranteed
	//         siblings that are too slow.
	//offset_type old_tail = _tail;
	//*_tail = std::max(this->tail(),(offset_type)(_reserve_head - this->size()));
	//std::cout << "-------- " << _guarantees.empty() << " ";
	//std::cout << "Guarantees empty: " << _guarantees.empty() << std::endl;
	//if( !_guarantees.empty() ) {
	//	std::cout << _guarantees.begin()->first << std::endl;
	//}
	//std::cout << "write waiting at tail = " << this->tail() << std::endl;
	offset_type new_tail = std::max(this->tail(),(offset_type)(_reserve_head - this->size()));
	_tail = new_tail;
	_write_condition.wait(lock, [&](){
			return ((_guarantees.empty() ||
			         _guarantees.begin()->first >= _tail) &&
			         //_guarantees.begin()->first >= new_tail) &&
			        _nrealloc_pending == 0);
		});
	//std::cout << "write done waiting at tail = " << this->tail() << std::endl;
	//std::cout << _guarantees.begin()->first << " vs. " << (offset_type)_tail << std::endl;
	// HACK TESTING
	//_tail = new_tail;
	//std::cout << "new tail = " << this->tail() << std::endl;
	++_nwrite_open;
	return buf_pointer(offset);
}
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::close_write(offset_type offset, size_type size) {
	//lock_guard_type lock(_mutex);
	unique_lock_type lock(_mutex);
	this->_ghost_write(offset, size);
	// TODO: Any way to support closing write blocks out of order?
	//         Any need to?
	//std::cout << offset << " vs. " << _head << std::endl;
	//assert( offset == this->head() );
	//*assert( offset == _head );
	
	// Wait until this block is at the head
	// Note: This allows write blocks to be closed out of order
	//         See RecvUDP for a use-case
	//std::cout << "RingBuffer: Waiting for close condition" << std::endl;
	_write_close_condition.wait(lock, [&](){
			return (offset == _head);
		});
	_write_close_condition.notify_all();
	//std::cout << "RingBuffer: Done waiting" << std::endl;
	
	//if( offset == _head ) {
	// Advance the readable head
	_head += size;
	_read_condition.notify_all();
	//}
	--_nwrite_open;
	_realloc_condition.notify_all();
}
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::_ghost_write(offset_type offset, size_type size) {
	offset_type buf_offset_beg = buf_offset(offset);
	offset_type buf_offset_end = buf_offset(offset + size);
	if( buf_offset_end < buf_offset_beg ) {
		// The write went into the ghost region, so copy to the ghosted part
		this->_copy_from_ghost(0, buf_offset_end);
		//_ghost_dirty_beg = buf_offset_end; // TODO: Do this inside copy_from_ghost
		// if( _ghost_dirty_beg
	}
	else if( buf_offset_beg < (offset_type)_ghost_size ) {
		// The write touched the ghosted front of the buffer
		_ghost_dirty = true;
		/*
		// TODO: Implement dirty region tracking
		offset_type ghost_offset_end = std::min(buf_offset_end,
		(offset_type)_ghost_size);
		_ghost_dirty_beg = std::min(_ghost_dirty_beg, buf_offset_beg);
		_ghost_dirty_end = std::max(_ghost_dirty_end, ghost_offset_end);
		*/
	}
}
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::_ghost_read(offset_type offset, size_type size) {
	offset_type buf_offset_beg = buf_offset(offset);
	offset_type buf_offset_end = buf_offset(offset + size);
	if( buf_offset_end < buf_offset_beg ) {
		// The read will enter the ghost region, so copy from the ghosted part
		if( _ghost_dirty ) {
			this->_copy_to_ghost(0, _ghost_size);
			_ghost_dirty = false;
		}
	}
}
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::_copy_to_ghost(offset_type buf_offset, size_type size) {
	// Copy the frames at the front of the buffer to the ghost region
	copy(// Src: buf_offset from start of buffer
	        _buf + buf_offset,
	        _buf + (buf_offset + size),
	        // Dst: buf_offset into ghost region
	        _buf + (this->size() + buf_offset));
}
template<class T, int A, int B, int C>
void RingBufferImpl<T,A,B,C>::_copy_from_ghost(offset_type buf_offset, size_type size) {
	// Copy the frames in the ghost region to the front of the buffer
	copy(// Src: buf_offset into ghost region
	        _buf + (this->size() + buf_offset),
	        _buf + (this->size() + buf_offset + size),
	        // Dst: buf_offset from start of buffer
	        _buf + buf_offset);
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::offset_type
RingBufferImpl<T,A,B,C>::buf_offset(offset_type offset) const {
	offset_type ret = (offset - _buf_offset0) % _size;
	return (ret < 0) ? (_size + ret) : ret;
}
template<class T, int A, int B, int C>
typename RingBufferImpl<T,A,B,C>::pointer
RingBufferImpl<T,A,B,C>::buf_pointer(offset_type offset) const {
	return _buf + buf_offset(offset);
}

typedef RingBufferImpl<> RingBuffer;

template<typename T>
class RingAccessorBase {
public:
	typedef T        value_type;
	typedef typename RingBuffer::offset_type offset_type;
	typedef typename RingBuffer::size_type   size_type;
	typedef T*                               pointer;
	typedef T const*                         const_pointer;
	typedef T&                               reference;
	typedef T const&                         const_reference;
	typedef std::vector<size_t> shape_type;
	void swap(RingAccessorBase& other) {
		std::swap(_ring,        other._ring);
		std::swap(_frame_size,  other._frame_size);
		std::swap(_frame_shape, other._frame_shape);
		std::swap(_nframe,      other._nframe);
	}
	void init(RingBuffer* ring) {
		//shape_type frame_shape = lookup_list<size_t>(ring->params(), "shape");
		auto ring_shape = ring->shape();
		shape_type frame_shape(ring_shape.begin(), ring_shape.end());
		return this->init(ring, frame_shape);
	}
	void init(RingBuffer* ring,
	          size_type   frame_size) {
		shape_type frame_shape({frame_size});
		return this->init(ring, frame_shape);
	}
	void init(RingBuffer* ring,
	          shape_type  frame_shape) {
		_ring = ring;
		_frame_shape = frame_shape;
		_frame_size  = product_of(frame_shape);
		_nframe = 1;
	}
	void request_size_frames(size_type gulp_size, size_type buffer_size) {
		assert( _ring );
		// Allow passing gulp_size==0 to only update buffer_size
		//   (nframe must already have been set).
		if( gulp_size != 0 ) {
			_nframe = gulp_size;
		}
		else {
			assert( _nframe != 0 );
		}
		_ring->request_size(  gulp_size*frame_bytes(),
		                    buffer_size*frame_bytes());
	}
	void request_size_bytes(size_type gulp_size, size_type buffer_size) {
		this->request_size_frames(div_ceil(  gulp_size, frame_bytes()),
		                          div_ceil(buffer_size, frame_bytes()));
	}
	// HACK TODO: This is only used by Depacketize.cpp as a WAR for
	//              needing to specify "scatter_factor" instead of
	//              "buffer_factor" as assumed by ConsumerTask.
	//              A better solution might be to allow giving
	//                input and output buffer factors to ConsumerTask.
	void request_buffer_factor(size_type factor) {
		this->request_size_frames(0, factor*_nframe);
	}
	inline shape_type     frame_shape() const { return _frame_shape; }
	inline size_type      frame_size()  const { return _frame_size; }
	inline size_type      frame_bytes() const { return (frame_size() *
	                                                    sizeof(value_type)); }
	// TODO: Rename to nframe?
	//         size() is useful for consistency with containers,
	//         but nframe() is more obvious.
	inline size_type         size()        const { return _nframe; }
	inline size_type         nframe()      const { return _nframe; }
	inline size_type         size_bytes()  const { return size()*frame_bytes(); }
	inline RingBuffer*       ring()        { return _ring; }
	inline RingBuffer const* ring() const  { return _ring; }
	// Warning: These values can potentially change immediately after the call
	inline offset_type       head() const  { return _ring->head() / (offset_type)frame_bytes(); }
	inline offset_type       tail() const  { return _ring->tail() / (offset_type)frame_bytes(); }
	inline space_type        space() const { return _ring->space(); }
protected:
	RingAccessorBase()
		: _ring(0), _frame_size(1), _frame_shape({1}), _nframe(0) {}
	RingAccessorBase(RingBuffer* ring,
	                 size_type   frame_size,
	                 shape_type  frame_shape,
	                 size_type   nframe)
		: _ring(ring),
		  _frame_size(frame_size),
		  _frame_shape(frame_shape),
		  _nframe(nframe) {}
private:
	RingBuffer* _ring;
	size_type   _frame_size;
	shape_type  _frame_shape;
	size_type   _nframe;
};
template<typename T>
class RingAccessor : public RingAccessorBase<T> {
	typedef RingAccessorBase<T> super_type;
public:
	typedef typename super_type::value_type  value_type;
	typedef T*                               pointer;
	typedef T const*                         const_pointer;
	typedef T&                               reference;
	typedef T const&                         const_reference;
	typedef strided_iterator<T>              iterator;
	typedef strided_iterator<const T>        const_iterator;
	typedef typename super_type::offset_type offset_type;
	typedef typename super_type::size_type   size_type;
	typedef typename super_type::shape_type  shape_type;
	
	void init(RingBuffer* ring) {
		super_type::init(ring);
		_data.set_stride(this->frame_size());
	}
	void init(RingBuffer* ring,
	          size_type   frame_size) {
		super_type::init(ring, frame_size);
		_data.set_stride(this->frame_size());
	}
	void init(RingBuffer* ring,
	          shape_type  frame_shape) {
		super_type::init(ring, frame_shape);
		_data.set_stride(this->frame_size());
	}
	// TODO: Consider renaming frame0 --> offset
	inline offset_type    frame0() const { return _frame0; }
	inline iterator       begin()        { return _data; }
	inline const_iterator begin()  const { return _data; }
	inline iterator       end()          { return _data + this->size(); }
	inline const_iterator end()    const { return _data + this->size(); }
	inline reference       operator[](size_t n)       { return _data[n]; }
	inline const_reference operator[](size_t n) const { return _data[n]; }
	void swap(RingAccessor& other) {
		super_type::swap(other);
		std::swap(_data,   other._data);
		std::swap(_frame0, other._frame0);
	}
protected:
	RingAccessor() : super_type(), _frame0(0) {}
	RingAccessor(RingBuffer* ring,
	             size_type   frame_size,
	             shape_type  frame_shape,
	             size_type   nframe)
		: super_type(ring, frame_size, frame_shape, nframe),
		  _data(0, frame_size), _frame0(0) {}
	inline offset_type byte0() const { return frame0()*(offset_type)this->frame_bytes(); }
	inline void        set_data(pointer data)         { _data = data; }
	inline void        set_frame0(offset_type frame0) { _frame0 = frame0; }
private:
	strided_iterator<T> _data;
	offset_type         _frame0;
};
template<typename T>
class RingReader : public RingAccessor<const T> {
	typedef RingAccessor<const T> super_type;
public:
	typedef typename super_type::offset_type offset_type;
	typedef typename super_type::size_type   size_type;
	typedef typename super_type::pointer     pointer;
	void open(bool guarantee_read=false) {
		// Open the ealiest written data
		assert( this->ring() );
		_guaranteed = guarantee_read;
		offset_type byte_offset;
		this->set_data( (pointer)this->ring()->open_read_at_tail(byte_offset,
		                                                         this->size_bytes(),
		                                                         _guaranteed,
		                                                         _shutdown_event) );
		// TODO: Is alignment of the returned head byte offset wrt frame_bytes
		//         here a concern?
		this->set_frame0(byte_offset / (offset_type)this->frame_bytes());
	}
	void open_at(offset_type frame_idx, bool guarantee_read=false) {
		assert( this->ring() );
		_guaranteed = guarantee_read;
		offset_type byte_offset = frame_idx * (offset_type)this->frame_bytes();
		this->set_data( (pointer)this->ring()->open_read_at(byte_offset,
		                                                    this->size_bytes(),
		                                                    _guaranteed,
		                                                    _shutdown_event) );
		this->set_frame0(frame_idx);
	}
	bool guaranteed() const { return _guaranteed; }
	bool still_valid(offset_type frame_offset=0) const {
		// Note: This shortcut is required because ring->still_valid() can
		//         return true even for guaranteed readers (without breaking
		//         the guarantee).
		if( _guaranteed ) {
			return true;
		}
		assert( this->ring() );
		offset_type byte_offset = (this->frame0()+frame_offset)*(offset_type)this->frame_bytes();
		return this->ring()->still_valid(byte_offset);
	}
	// 'Advance-by' operator
	RingReader& operator+=(size_type nframe) {
		assert( this->ring() );
		size_type   advance_bytes = nframe * (offset_type)this->frame_bytes();
		offset_type offset_bytes = this->byte0();
		pointer ptr = (pointer)this->ring()->advance_read_by(offset_bytes,
		                                                     this->size_bytes(),
		                                                     _guaranteed, advance_bytes,
		                                                     _shutdown_event);
		this->set_data(ptr);
		this->set_frame0(offset_bytes / (offset_type)this->frame_bytes());
		return *this;
	}
	// 'Advance' operator
	RingReader& operator++() { return *this += this->size(); }
	void close() {
		if( this->begin() ) {
			assert( this->ring() );
			this->ring()->close_read(this->byte0(),
			                   this->size_bytes(),
			                   _guaranteed);
			this->set_data(0);
		}
	}
	RingReader() : super_type() {}
	~RingReader() {
		this->close();
	}
	           RingReader(RingReader const& other) = delete;
	RingReader& operator=(RingReader const& other) = delete;
	void swap(RingReader& other) {
		super_type::swap(other);
		std::swap(_guaranteed,     other._guaranteed);
		std::swap(_shutdown_event, other._shutdown_event);
	}
	RingReader(RingReader&& other)
		: super_type() {
		this->swap(other);
	}
	RingReader& operator=(RingReader&& other) {
		this->close();
		this->swap(other);
		return *this;
	}
	void shutdown() {
		_shutdown_event.set();
		this->ring()->notify_of_shutdown();
	}
private:
	bool       _guaranteed;
	event_flag _shutdown_event;
};
template<typename T>
class RingWriter : public RingAccessorBase<T> {
	typedef RingAccessorBase<T> super_type;
public:
	typedef typename super_type::size_type   size_type;
	typedef typename super_type::offset_type offset_type;
	typedef RingWriteBlock<T> block_type;
	inline offset_type reserve_head() const { return this->ring()->reserve_head();}
	block_type open() {
		return block_type(this->ring(),
		                  this->frame_size(),
		                  this->frame_shape(),
		                  this->size());
	}
	void reset(offset_type frame0) {
		assert( this->ring() );
		print_shape(this->frame_shape());
		//std::cout << std::endl;
		//std::cout << "*** reset frame0 = " << frame0 << ", frame_bytes = " << this->frame_bytes() << std::endl;
		offset_type fbytes = (offset_type)this->frame_bytes();
		if( product_would_overflow(frame0, fbytes) ) {
			std::cout << "For Product of " << frame0
			          << " and " << fbytes << std::endl;
			throw std::invalid_argument("Byte offset for given frame0 "
			                            "would overflow");
		}
		
		offset_type byte0 = frame0 * fbytes;
		// ** TODO: Decide on name
		//this->ring()->reset_write(byte0);
		this->ring()->init_write(byte0);
	}
};
template<typename T>
class RingWriteBlock : public RingAccessor<T> {
	typedef RingAccessor<T> super_type;
public:
	typedef typename super_type::size_type   size_type;
	typedef typename super_type::shape_type  shape_type;
	typedef typename super_type::offset_type offset_type;
	typedef typename super_type::pointer     pointer;
private:
	friend class RingWriter<T>;
	RingWriteBlock(RingBuffer* ring,
	               size_type   frame_size,
	               shape_type  frame_shape,
	               size_type   nframe)
		: super_type(ring, frame_size, frame_shape, nframe) {
		assert( this->ring() );
		offset_type byte_offset;
		pointer data = (pointer)this->ring()->open_write(this->size_bytes(),
		                                                 byte_offset);
		offset_type frame0 = byte_offset / (offset_type)this->frame_bytes();
		//std::cout << "RingWriteBlock() byte_offset = " << byte_offset
		//          << ", frame_bytes = " << this->frame_bytes()
		//          << ", frame0 = " << frame0
		//          << std::endl;
		this->set_data(data);
		this->set_frame0(frame0);
	}
public:
	RingWriteBlock() : super_type() {}
	~RingWriteBlock() {
		this->close();
	}
	           RingWriteBlock(RingWriteBlock const& other) = delete;
	RingWriteBlock& operator=(RingWriteBlock const& other) = delete;
	RingWriteBlock(RingWriteBlock&& tmp) : RingAccessor<T>() {
		this->swap(tmp);
	}
	RingWriteBlock& operator=(RingWriteBlock&& tmp) {
		this->close();
		this->swap(tmp);
		return *this;
	}
	void close() {
		if( this->begin() ) {
			assert( this->ring() );
			offset_type byte_offset = this->frame0() * (offset_type)this->frame_bytes();
			this->ring()->close_write(byte_offset, this->size_bytes());
			this->set_data(0);
		}
	}
};
