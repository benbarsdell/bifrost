
#include "Task.hpp"
#include "affinity.hpp"

bool Task::input_ring_exists(std::string   input_name,
                             Pipeline*     pipeline,
                             const Object* definition) {
	
	Object const& input_conns = lookup_object(*definition, "__inputs__",
	                                          Object());
	std::string ring_name = lookup_string(input_conns, input_name);
	try {
		pipeline->get_ring(ring_name);
		return true;
	}
	catch( std::out_of_range ) {
		return false;
	}
}

Task::Task(Pipeline*         pipeline,
           const Object*     definition)
	: _pipeline(pipeline), _definition(*definition) {
	
	//_name = get_property<std::string>("__name__");
	_name = lookup_string(params(), "__name__");
	/*
	_log.id        = _name;
	_log.verbosity = Log::SLOG_WARNING;
	int verbosity = lookup_integer(params(), "verbosity", 0);
	_log.verbosity += verbosity;
	_log.use_utc     = true;
	_log.time_format = "%Y-%m-%dT%H:%M:%SZ";
	//_log.setStream(_log_stream);
	*/
	// Note: Because listeners can decide which log topics to tune in to,
	//         this here is really just an optimisation to avoid slowdowns with
	//         excessive debug/trace logging. Having this setting allows
	//         implementers to go crazy with logging without worrying about
	//         performance issues.
	_log_verbosity = TASK_LOG_INFO + lookup_integer(params(), "verbosity", 0);
	
	//for( auto const& kv : get_property("__inputs__", Object()) ) {
	Object const& input_conns = lookup_object(params(), "__inputs__",
	                                          Object());
	for( auto const& kv : input_conns ) {
		std::string name   = kv.first;
		std::string target = kv.second.get<std::string>();
		_input_names.insert(name);
		_input_targets.insert(std::make_pair(name, target));
	}
	Object const& output_conns = lookup_object(params(), "__output__",
	                                           Object());
	for( auto const& kv : output_conns ) {
		std::string name      = kv.first;
		std::string ring_name = kv.second.get<std::string>();
		try {
			// Find and cache the referenced ring
			Task::ring_type* ring = _pipeline->get_ring(ring_name);
			_output_rings.insert(std::make_pair(name, ring));
		}
		catch( std::out_of_range ) {
			// The referenced ring hasn't been created by its owner yet
			throw Task::DependencyError(ring_name);
		}
	}
	
	// TODO: Task::Task() parses __inputs__  --> map input_names: input_conns
	//           and also parses __outputs__ --> pipeline.get_output(output_conn)
	//       Task::get_input() calls pipeline.get_input(input_conns[input_names]) and throws DependencyError if not found
		
	/*
	// Initialise empty ring maps with given input and output names
	for( string_set::const_iterator it=inputs.begin(); it!=inputs.end(); ++it ) {
		_inputs[*it] = 0;
	}
	for( string_set::const_iterator it=outputs.begin(); it!=outputs.end(); ++it ) {
		_outputs[*it] = 0;
	}
	// Fill in ring maps with pointers to named rings as specified in definition
	const Object& input_conns = get_key<Object>(*definition, "__inputs__", Object());
	for( Object::const_iterator it=input_conns.begin(); it!=input_conns.end(); ++it ) {
		std::string key = it->first;
		ring_map::iterator ring_it = _inputs.find(key);
		if( ring_it == _inputs.end() ) {
			throw std::out_of_range("No input named "+key);
		}
		std::string ring_name = it->second.get<std::string>();
		try {
			ring_it->second = _pipeline->get_ring(ring_name);
		}
		catch( std::out_of_range ) {
			throw Task::DependencyError();
		}
	}
	const Object& output_conns = get_key<Object>(*definition, "__outputs__", Object());
	for( Object::const_iterator it=output_conns.begin(); it!=output_conns.end(); ++it ) {
		std::string key = it->first;
		ring_map::iterator ring_it = _outputs.find(key);
		if( ring_it == _outputs.end() ) {
			throw std::out_of_range("No output named "+key);
		}
		std::string ring_name = it->second.get<std::string>();
		ring_it->second = _pipeline->get_ring(ring_name);
	}
	// Fill in unspecified outputs with automatically-named rings
	for( ring_map::iterator it=_outputs.begin(); it!=_outputs.end(); ++it ) {
		if( !it->second ) {
			std::string ring_name = this->default_ring_name(it->first);
			// ** TODO: This should be create_ring
			it->second = _pipeline->get_ring(ring_name);
		}
	}
	*/
}
Task::~Task() {}
template<typename SpaceType>
Task::ring_type* Task::create_output(std::string name, SpaceType space) {
	bool already_exists = _output_rings.count(name);
	if( already_exists ) {
		// Output ring is owned by another task
		return _output_rings[name];
	}
	else {
		//std::string ring_name = this->default_ring_name(name);
		std::string ring_name = this->name() + "." + name;
		Task::ring_type* ring = _pipeline->create_ring(ring_name, space);
		_output_rings.insert(std::make_pair(name, ring));
		return ring;
	}
}
// Explicit template instantiations
template Task::ring_type* Task::create_output<std::string>(std::string name, std::string  space);
template Task::ring_type* Task::create_output<space_type> (std::string name, space_type   space);
template Task::ring_type* Task::create_output<char const*> (std::string name, char const* space);

Task::ring_type* Task::get_output_ring(std::string name) {
	ring_map::const_iterator it = _output_rings.find(name);
	if( it == _output_rings.end() ) {
		throw std::out_of_range("No output named "+name);
	}
	return it->second;
}
Task::ring_type* Task::get_input_ring(std::string name) {
	try {
		// Look for cached ring
		return lookup(_input_rings, name);
	}
	catch( std::out_of_range ) {
		// Ring hasn't been cached yet
		std::string ring_name = lookup(_input_targets, name);
		try {
			// Look for target ring name
			Task::ring_type* ring = _pipeline->get_ring(ring_name);
			// Cache target ring
			_input_rings.insert(std::make_pair(name, ring));
			return ring;
		}
		catch( std::out_of_range ) {
			throw Task::DependencyError(ring_name);
		}
	}
}
// Run asynchronously in new thread
void Task::launch() {
	//_running_flag.test_and_set();
	_shutdown.clear();
	_thread = std::thread(&Task::_launch, this);
	/*_thread = std::thread([&]() {
	  if( cpu_core >= 0 ) {
	  bind_to_core(cpu_core);
	  }
	  _running_flag.test_and_set();
	  this->main();
	  });*/
}
void Task::_launch() {
	// Automatically bind the CPU core if specified
	if( params().count("cpu_cores") ) {
		std::vector<int> cpu_cores = lookup_list<int>(params(), "cpu_cores");
		if( !cpu_cores.empty() ) {
			int first_core = cpu_cores[0];
			if( first_core >= 0 ) {
				bind_to_core(first_core);
			}
		}
	}
	
	// Automatically set the GPU device if specified
	if( params().count("gpu_devices") ) {
		cudaError_t cuda_ret;
		cuda_ret = cudaGetLastError();
		if( cuda_ret != cudaSuccess ) {
			throw std::runtime_error(cudaGetErrorString(cuda_ret));
		}
		// Automatically default to not spinning the CPU
		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
		int gpu_idx;
		// TODO: This won't allow mixed list of int/string
		try {
			gpu_idx = lookup_list<int>(params(), "gpu_devices")[0];
		}
		catch( TypeError ) {
			std::string first_device =
				lookup_list<std::string>(params(), "gpu_devices")[0];
			cuda_ret = cudaDeviceGetByPCIBusId(&gpu_idx, first_device.c_str());
			if( cuda_ret != cudaSuccess ) {
				throw std::runtime_error(cudaGetErrorString(cuda_ret));
			}
		}
		cuda_ret = cudaSetDevice(gpu_idx);
		if( cuda_ret != cudaSuccess ) {
			throw std::runtime_error(cudaGetErrorString(cuda_ret));
		}
	}
	
	this->main();
}
