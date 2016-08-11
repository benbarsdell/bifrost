
/*
  Simple pthreads mutex and condition-variable wrappers
  Ben Barsdell (2015)
  BSD 3-Clause license
  
  Note: Unlike the C++11 std::mutex, the pthreads-based mutex here supports
          inter-process sharing.
  Note: According to the docs, setting a mutex as recursive can lead to problems
          with condition variables.
  
  Example
  -------
  Mutex             mutex(Mutex::PROCESS_SHARED);
  ConditionVariable cv(ConditionVariable::PROCESS_SHARED);
  void foo() {
    UniqueLock<Mutex> lock(mutex);
    if( !cv.wait_for(lock, timeout_secs, [](){ return x == y; }) ) {
      // Timed out
    }
    // ...
  }
*/

#pragma once

#include <cerrno>
#include <cstring>
#include <limits>
#include <pthread.h>

template<typename MutexType>
class LockGuard {
public:
	typedef MutexType mutex_type;
	explicit LockGuard(mutex_type& m) : _mutex(m) { _mutex.lock(); }
	~LockGuard() { _mutex.unlock(); }
private:
	// Prevent copying or assignment
	void operator=(const LockGuard&);
	LockGuard(const LockGuard&);
	mutex_type& _mutex;
};
template<typename MutexType>
class UniqueLock {
public:
	typedef MutexType mutex_type;
	explicit UniqueLock(mutex_type& m, bool locked=true)
		: _mutex(m), _locked(locked) { if( locked ) { _mutex.lock(); } }
	~UniqueLock() { this->unlock(); }
	void unlock() {
		if( !_locked ) {
			return;
		}
		_mutex.unlock();
		_locked = false;
	}
	void lock() {
		if( _locked ) {
			return; 
		}
		_mutex.lock();
		_locked = true;
	}
	mutex_type const& mutex()     const { return _mutex; }
	mutex_type&       mutex()           { return _mutex; }
	bool              owns_lock() const { return _locked; }
	operator bool()                     { return this->owns_lock(); }
private:
	// Not copy-assignable
	UniqueLock& operator=(const UniqueLock& );
	UniqueLock(const UniqueLock& );
	mutex_type& _mutex;
	bool        _locked;
};

namespace pthread {

class Mutex {
	pthread_mutex_t _mutex;
	// Not copy-assignable
	Mutex(const Mutex& );
	Mutex& operator=(const Mutex& );
	static void check_error(int ret) {
		if( ret ) { throw Mutex::Error(std::strerror(ret)); }
	}
	class Attrs {
		pthread_mutexattr_t _attr;
		Attrs(Attrs const& );
		Attrs& operator=(Attrs const& );
	public:
		Attrs()  { check_error( pthread_mutexattr_init(&_attr) ); }
		~Attrs() { pthread_mutexattr_destroy(&_attr); }
		pthread_mutexattr_t* handle() { return &_attr; }
	};
public:
	struct Error : public std::runtime_error {
		Error(const std::string& what_arg) : std::runtime_error(what_arg) {}
	};
	enum {
		RECURSIVE      = 1<<0,
		PROCESS_SHARED = 1<<1
	};
	Mutex(unsigned flags=0) {
		Mutex::Attrs attrs;
		if( flags & Mutex::RECURSIVE )      {
			check_error( pthread_mutexattr_settype(attrs.handle(),
			                                       PTHREAD_MUTEX_RECURSIVE) );
		}
		if( flags & Mutex::PROCESS_SHARED ) {
			check_error( pthread_mutexattr_setpshared(attrs.handle(),
			                                          PTHREAD_PROCESS_SHARED) );
		}
		check_error( pthread_mutex_init(&_mutex, attrs.handle()) );
	}
	~Mutex()       { pthread_mutex_destroy(&_mutex); }
	void lock()    { check_error( pthread_mutex_lock(&_mutex) ); }
	bool trylock() {
		int ret = pthread_mutex_trylock(&_mutex);
		if( ret == EBUSY ) {
			return false;
		}
		else {
			check_error(ret);
			return true;
		}
	}
	void unlock()  { check_error( pthread_mutex_unlock(&_mutex) ); }
	operator const pthread_mutex_t&() const { return _mutex; }
	operator       pthread_mutex_t&()       { return _mutex; }
};

class ConditionVariable {
	pthread_cond_t _cond;
	// Not copy-assignable
	ConditionVariable(ConditionVariable const& );
	ConditionVariable& operator=(ConditionVariable const& );
	static void check_error(int ret) {
		if( ret) { throw ConditionVariable::Error(strerror(ret)); }
	}
	class Attrs {
		pthread_condattr_t _attr;
		Attrs(Attrs const& );
		Attrs& operator=(Attrs const& );
	public:
		Attrs()  { check_error( pthread_condattr_init(&_attr) ); }
		~Attrs() { pthread_condattr_destroy(&_attr); }
		pthread_condattr_t* handle() { return &_attr; }
	};
public:
	struct Error : public std::runtime_error {
		Error(const std::string& what_arg) : std::runtime_error(what_arg) {}
	};
	enum {
		PROCESS_SHARED = 1<<1
	};
	ConditionVariable(unsigned flags=0) {
		ConditionVariable::Attrs attrs;
		if( flags & ConditionVariable::PROCESS_SHARED ) {
			check_error( pthread_condattr_setpshared(attrs.handle(),
			                                         PTHREAD_PROCESS_SHARED) );
		}
		check_error( pthread_cond_init(&_cond, attrs.handle()) );
	}
	~ConditionVariable()    { pthread_cond_destroy(&_cond); }
	void wait(UniqueLock<Mutex>& lock) {
		check_error( pthread_cond_wait(&_cond, &(pthread_mutex_t&)lock.mutex()) );
	}
	template<typename Predicate>
	void wait(UniqueLock<Mutex>& lock, Predicate pred) {
		while( !pred() ) {
			this->wait(lock);
		}
	}
	bool wait_until(UniqueLock<Mutex>& lock, timespec const& abstime) {
		int ret = pthread_cond_timedwait(&_cond, &(pthread_mutex_t&)lock.mutex(),
		                                 &abstime);
		if( ret == ETIMEDOUT ) {
			return false;
		}
		else {
			check_error(ret);
			return true;
		}
	}
	template<typename Predicate>
	bool wait_until(UniqueLock<Mutex>& lock, timespec const& abstime, Predicate pred) {
		while( !pred() ) {
			if( !this->wait_until(lock, abstime) ) {
				return pred();
			}
		}
		return true;
	}
	template<typename Predicate>
	bool wait_for(UniqueLock<Mutex>& lock, double timeout_secs, Predicate pred) {
		struct timespec abstime = {0};
		check_error( clock_gettime(CLOCK_REALTIME, &abstime) );
		//time_t secs = (time_t)(std::min(timeout_secs, double(INT_MAX))+0.5);
		time_t secs = (time_t)(std::min(timeout_secs,
		                                double(std::numeric_limits<time_t>::max()))+0.5);
		abstime.tv_sec  += secs;
		abstime.tv_nsec += (long)((timeout_secs - secs)*1e9 + 0.5);
		return this->wait_until(lock, abstime, pred);
	}
	void notify_one()       { check_error( pthread_cond_signal(&_cond) ); }
	void notify_all()       { check_error( pthread_cond_broadcast(&_cond) ); }
};

} // namespace pthread
