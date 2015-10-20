
/*
  TODO: Add logging(?)
        Add profiling?
 */

#pragma once

#include "Pipeline.hpp"
#include "Object.hpp"
#include "Log.hpp"
#include "Profiler.hpp"
//#include "sequence_map.hpp"

#include <string>
#include <set>
#include <map>
#include <thread>
#include <atomic>
#include <chrono>

#include <iostream> // Debugging only

// Subclasses implement this to return new Subclass(pipeline, definition);
extern "C" Task* create(Pipeline*     pipeline,
                        const Object* definition);

//class TaskMonitor;

class Task {
public:
	struct DependencyError : public std::runtime_error {
		DependencyError(std::string s="") : std::runtime_error(s) {}
		virtual const char* what() const throw() {
			return "Task dependency error";
		}
	};
	typedef Pipeline::ring_type   ring_type;
	typedef std::set<std::string> string_set;
	
	// TODO: Can any of this be made protected/private? Friend Pipeline?
	// Subclasses create all outputs in their constructor
	// Note: Be careful when using get_input() here to avoid cyclic dependency
	Task(Pipeline*     pipeline,
	     const Object* definition);
	// Subclasses initialise all inputs/outputs/etc. here
	// Note: This is always run single-threaded
	virtual void init() = 0;
protected:
	// Subclasses implement main processing here
	// Note: This is run in a new thread
	virtual void main() = 0;
public:
	// TODO: Rule of 3/5
	virtual ~Task();
	
	//inline virtual std::string default_ring_name(std::string output_name) {
	//	return this->name()+"."+output_name;
	//}
	inline std::string   name()           const { return _name; }
	inline Object const& get_definition() const { return _definition; }
	/*
	template<typename T>
	inline T const& get_property(std::string name) const {
		return get_key<T>(_definition, name);
	}
	template<typename T>
	inline T const& get_property(std::string name, T const& default_val) const {
		return get_key(_definition, name, default_val);
	}
	*/
	inline Object&       params()       { return _definition; }
	inline Object const& params() const { return _definition; }
	
	template<typename SpaceType>
	ring_type* create_output(std::string name,
	                         SpaceType   space);
	// Convenience function for checking dependencies in subclasses' create()
	static bool input_ring_exists(std::string   input_name,
	                              Pipeline*     pipeline,
	                              const Object* definition);
	ring_type* get_output_ring(std::string name);
	ring_type* get_input_ring( std::string name);
	// Run asynchronously in new thread
	void launch();
private:
	void _launch();
public:
	// Run synchronously in current thread
	inline void run() {
		this->main();
	}
	//inline void request_stop() {
	//	_running_flag.clear();
	//}
	inline virtual void shutdown() {
		std::cout << this->name() << "::shutdown" << std::endl;
		_shutdown.set();
	}
	// Wait for task to complete after calling request_stop()
	inline void wait() {
		//_running_flag.clear();
		//this->shutdown();
		if( _thread.joinable() ) {
			_thread.join();
		}
	}
	inline Pipeline*       pipeline()       { return _pipeline; }
	inline Pipeline const* pipeline() const { return _pipeline; }
	//inline Log&       log()       { return _log; }
	//inline Log const& log() const { return _log; }
	inline Profiler&       profiler()       { return _prof; }
	inline Profiler const& profiler() const { return _prof; }
protected:
	/*
	// TODO: add_output/input(name, default=true)?
	void add_output(std::string name="") {
		_outputs.insert(std::make_pair(name, ring_type()));
	}
	void add_input(std::string name="") {
		_inputs.insert(std::make_pair(name, 0));
	}
	*/
	const std::set<std::string>& get_input_names() const {
		return _input_names;
	}
	const std::map<std::string,std::string>& get_input_targets() const {
		return _input_targets;
	}
	/*
	inline bool stop_requested() {
		// Note: This would be simpler with atomic<bool>, but atomic_flag works
		//         and is guarantee_readsd to be lock-free.
		// TODO: This is now horribly unclear!
		bool was_true = _running_flag.test_and_set();
		if( !was_true ) {
			_running_flag.clear();
		}
		return !was_true;
	}
	*/
public:
	// HACK TODO: This should probably be protected, but need to friend CommandThread
	inline bool shutdown_requested() const { return _shutdown.is_set(); }
	
	void broadcast(std::string topic,
	               Object      metadata,
	               char const* data=0,
	               size_t      size=0) const {
		topic = _pipeline->name() + "." + _name + "." + topic;
		metadata["__date__"]   = Value(get_current_utc_string());
		metadata["__time__"]   = Value(get_current_clock_ns());
		metadata["__period__"] = Value((int64_t)1000000000ll);
		std::string s = topic + " " + Value(metadata).serialize();
		_pipeline->broadcast_socket().send(s, !bool(data));
		if( data ) {
			_pipeline->broadcast_socket().send(data, size);
		}
	}
	void log(std::string topic, std::string msg, ...) const {
		va_list va; va_start(va, msg); this->vlog(topic, msg, va); va_end(va);
	}
#define TASK_DEFINE_LOG_LEVEL(name, level)	  \
	void log_##name(std::string msg, ...) const { \
		if( level <= _log_verbosity ) { \
			va_list va; va_start(va, msg); this->log(#name, msg, va); va_end(va); \
		} \
	}
	TASK_DEFINE_LOG_LEVEL(critical, TASK_LOG_CRIT)
	TASK_DEFINE_LOG_LEVEL(error,    TASK_LOG_ERR)
	TASK_DEFINE_LOG_LEVEL(warning,  TASK_LOG_WARNING)
	TASK_DEFINE_LOG_LEVEL(notice,   TASK_LOG_NOTICE)
	TASK_DEFINE_LOG_LEVEL(info,     TASK_LOG_INFO)
	TASK_DEFINE_LOG_LEVEL(debug,    TASK_LOG_DEBUG)
	TASK_DEFINE_LOG_LEVEL(trace,    TASK_LOG_TRACE)
#undef TASK_DEFINE_LOG_LEVEL
	
protected:
	// Convenience methods for interruptable sleeps
	//   Return false if sleep was interrupted by shutdown event
	template<class Rep, class Period>
	bool sleep_for(const std::chrono::duration<Rep,Period>& rel_time) const {
		return !_shutdown.wait_for(rel_time);
	}
	template<class Clock, class Duration>
	bool sleep_until(const std::chrono::time_point<Clock,Duration>& abs_time) const{
		return !_shutdown.wait_until(abs_time);
	}
private:
	enum {
		TASK_LOG_EMERG   = 0, // system is unusable
		TASK_LOG_ALERT   = 1, // action must be taken immediately
		TASK_LOG_CRIT    = 2, // critical conditions
		TASK_LOG_ERR     = 3, // error conditions
		TASK_LOG_WARNING = 4, // warning conditions
		TASK_LOG_NOTICE  = 5, // normal but significant condition
		TASK_LOG_INFO    = 6, // informational
		TASK_LOG_DEBUG   = 7, // debug-level messages
		TASK_LOG_TRACE   = 8  // call-tracing messages
	};
	void vlog(std::string topic, std::string msg, va_list args) const {
		// Note: Must lock due to use of _buffer
		std::lock_guard<std::mutex> lock(_log_mutex);
		int ret = vsnprintf(&_log_buffer[0], _log_buffer.capacity(),
		                    msg.c_str(), args);
		if( (size_t)ret >= _log_buffer.capacity() ) {
			_log_buffer.resize(ret+1); // Note: +1 for NULL terminator
			ret = vsnprintf(&_log_buffer[0], _log_buffer.capacity(),
			                msg.c_str(), args);
		}
		_log_buffer.resize(ret+1); // Note: +1 for NULL terminator
		if( ret < 0 ) {
			// TODO: How to handle encoding error?
		}
		Object metadata; // Note: No metadata is used here
		this->broadcast(topic, metadata, &_log_buffer[0], _log_buffer.size());
	}
	//friend class TaskMonitor;
	typedef std::map<std::string,ring_type*> ring_map;
	Pipeline*        _pipeline;
	Object           _definition;
	std::string      _name;
	std::set<std::string>             _input_names;
	std::map<std::string,std::string> _input_targets;
	std::map<std::string,ring_type*>  _input_rings;
	std::map<std::string,ring_type*>  _output_rings;
	std::thread      _thread;
	//std::atomic_flag _running_flag;
	event_flag       _shutdown;
	//ring_map         _outputs;
	//ring_map         _inputs;
	//Log              _log;
	//ZMQStream        _log_stream;
	mutable std::mutex        _log_mutex;
	mutable std::vector<char> _log_buffer;
	int                       _log_verbosity;
	Profiler         _prof;
};
