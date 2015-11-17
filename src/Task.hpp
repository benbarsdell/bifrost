
#pragma once

#include "Pipeline.hpp"
#include "Object.hpp"
#include "Log.hpp"
#include "Profiler.hpp"

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

class Task {
public:
	struct DependencyError : public std::runtime_error {
		DependencyError(std::string s="") : std::runtime_error(s) {}
		virtual const char* what() const throw() {
			return "Task dependency error";
		}
	};
	typedef Pipeline::ring_type   ring_type;
	typedef Logger<Task>          log_type;
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
	// TODO: Rule of 3
	virtual ~Task();
	
	//inline virtual std::string default_ring_name(std::string output_name) {
	//	return this->name()+"."+output_name;
	//}
	inline std::string   name()           const { return _name; }
	inline Object const& get_definition() const { return _definition; }
	inline Object&       params()               { return _definition; }
	inline Object const& params()         const { return _definition; }
	
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
		//std::cout << this->name() << "::shutdown" << std::endl;
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
	inline Pipeline*           pipeline()       { return _pipeline; }
	inline Pipeline const*     pipeline() const { return _pipeline; }
	inline log_type&           log()            { return _log; }
	inline log_type const&     log()      const { return _log; }
	inline Profiler&           profiler()       { return _prof; }
	inline Profiler const&     profiler() const { return _prof; }
	void broadcast(std::string topic,
	               Object      metadata,
	               char const* data=0,
	               size_t      size=(size_t)-1) const;
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
private:
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
	log_type         _log;
	Profiler         _prof;
};
