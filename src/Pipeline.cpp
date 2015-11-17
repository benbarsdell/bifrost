
#include <sstream>

#include "Pipeline.hpp"
#include "Task.hpp"

Value& erase_all_string_prefix(Value& val,
                               std::string prefix) {
	// TODO: Could theoretically add support for list entries too
	if( val.is<Object>() ) {
		Object& obj = val.get<Object>();
		Object::iterator it = obj.begin();
		while( it != obj.end() ) {
			std::string const& key = it->first;
			if( key.substr(0, prefix.size()) == prefix ) {
				obj.erase(it++);
			}
			else {
				erase_all_string_prefix((it++)->second, prefix);
			}
		}
	}
	return val;
}

Value& replace_all_string_prefix(Value&      val,
                                 std::string prefix,
                                 std::string replacement) {
	if( val.is<std::string>() ) {
		std::string& sval = val.get<std::string>();
		if( sval.substr(0, prefix.size()) == prefix ) {
			sval = replacement + sval.substr(prefix.size());
		}
	}
	else if( val.is<Object>() ) {
		Object& obj = val.get<Object>();
		for( Object::iterator it=obj.begin(); it!=obj.end(); ++it ) {
			replace_all_string_prefix(it->second, prefix, replacement);
		}
	}
	else if( val.is<List>() ) {
		List& lst = val.get<List>();
		for( List::iterator it=lst.begin(); it!=lst.end(); ++it ) {
			replace_all_string_prefix(*it, prefix, replacement);
		}
	}
	return val;
}
Value& substitute_references(Value& val, Value& src,
                             std::string ref_prefix, std::string sep,
                             std::string opensym, std::string closesym);
Value& find_key_by_reference(std::string reference,
                             Value& src,
                             Value& rootsrc,
                             std::string ref_prefix, std::string sep,
                             std::string opensym, std::string closesym) {
	// TODO: Could use list parsing to simplify things and allow nested references?
	//         Downside: The [] syntax is a bit ugly
	//std::cout << "Ref: " << reference << std::endl;
	if( reference.empty() ) {
		return src;
	}
	else if( src.is<Object>() ) {
		if( reference.substr(0, sep.size()) != sep ) {
			throw std::invalid_argument("Expected '"+sep+
			                            "' before "+reference);
		}
		reference = reference.substr(sep.size()); // Crop off leading sep
		size_t end = reference.size();
		end = std::min(end, reference.find(sep));
		end = std::min(end, reference.find(opensym));
		std::string key = reference.substr(0, end);
		std::string rem = reference.substr(end);
		return find_key_by_reference(rem,
		                             lookup(src.get<Object>(), key),
		                             rootsrc,
		                             ref_prefix, sep,
		                             opensym, closesym);
	}
	else if( src.is<List>() ) {
		if( reference.substr(0, opensym.size()) != opensym ) {
			throw std::invalid_argument("Expected '"+opensym+
			                            "' before "+reference);
		}
		reference = reference.substr(opensym.size()); // Crop off leading opensym
		size_t end = reference.find(closesym);
		if( end == std::string::npos ) {
			throw std::invalid_argument("Expected '"+closesym+
			                            "' before "+reference);
		}
		std::string key = reference.substr(0, end);
		std::string rem = reference.substr(end+closesym.size());
		Value keyval = parse_value(key);
		// Allow "$a[$b]"
		substitute_references(keyval, rootsrc,
		                      ref_prefix, sep,
		                      opensym, closesym);
		if( !keyval.is<int64_t>() ) {
			throw std::invalid_argument("Reference list index is not an"
			                            " integer: "+reference);
		}
		return find_key_by_reference(rem,
		                             src.get<List>().at(keyval.get<int64_t>()),
		                             rootsrc,
		                             ref_prefix, sep,
		                             opensym, closesym);
	}
	else {
		//throw std::invalid_argument("Expected "+sep+" or '[' before "+reference);
		throw std::invalid_argument("Invalid reference: "+reference);
	}
}
Value& substitute_references(Value& val, Value& src,
                             std::string ref_prefix, std::string sep,
                             std::string opensym, std::string closesym) {
	if( val.is<std::string>() ) {
		std::string sval = val.get<std::string>();
		if( sval.substr(0, ref_prefix.size()) == ref_prefix ) {
			std::string key = sval.substr(1);
			key = sep + key; // Add implicit root separator
			val = substitute_references(find_key_by_reference(key, src, src,
			                                                  ref_prefix, sep,
			                                                  opensym, closesym),
			                            src,
			                            ref_prefix, sep,
			                            opensym, closesym);
		}
	}
	else if( val.is<Object>() ) {
		Object& obj = val.get<Object>();
		for( Object::iterator it=obj.begin(); it!=obj.end(); ++it ) {
			substitute_references(it->second, src,
			                      ref_prefix, sep,
			                      opensym, closesym);
		}
	}
	else if( val.is<List>() ) {
		List& lst = val.get<List>();
		for( List::iterator it=lst.begin(); it!=lst.end(); ++it ) {
			substitute_references(*it, src,
			                      ref_prefix, sep,
			                      opensym, closesym);
		}
	}
	return val;
}

// Replaces string values of the form "$obj.key1.key2" with the value at document[obj][key1][key2]
// Chained references are supported
// "$~" can be used as an alias for "$__pipeline__.__tasks__"
// Actual '$' characters at the beginning of a string can be escaped as "\$"
void apply_reference_substitutions(Value& val,
                                   Value& src,
                                   //Object const& process_attrs,
                                   std::string ref_prefix="$",
                                   std::string sep=".",
                                   std::string opensym="[",
                                   std::string closesym="]") {
	// Erase 'commented out' keys
	erase_all_string_prefix(val, "#");
	//std::cout << Value(document) << std::endl;
	//Object& docobj = document.get<Object>();
	// "$~" --> "__pipeline__.__tasks__"
	replace_all_string_prefix(val,//lookup(val, "__pipeline__"),
	                          ref_prefix+"~",
	                          ref_prefix+"__pipeline__.__tasks__");
	//std::cout << std::endl;
	//std::cout << Value(document) << std::endl;
	// Note: We copy process_attrs into the root of the document for convenience
	//**docobj.insert(process_attrs.begin(), process_attrs.end());
	substitute_references(val,//lookup(docobj, "__pipeline__"),
	                      src,//document,
	                      ref_prefix, sep,
	                      opensym, closesym);
	// "\$" --> "$"
	replace_all_string_prefix(val,//lookup(docobj, "__pipeline__"),
	                          "\\"+ref_prefix,
	                          ref_prefix);
}
Pipeline::Pipeline(int io_threads)
	: _log(this), _zmq_ctx(io_threads), _pubsock(_zmq_ctx) {}
void Pipeline::load(std::string filename, Object process_attrs) {
	std::ifstream f(filename.c_str());
	Value document_value;
	std::string err;
	parse(document_value,
	      std::istream_iterator<char>(f),
	      std::istream_iterator<char>(),
	      &err);
	if( !err.empty() ) {
		throw std::runtime_error(err);
	}
	if( !document_value.is<Object>() ) {
		throw std::runtime_error("Document is not an object");
	}
	Object& document = document_value.get<Object>();
	//std::cout << document_value << std::endl;
	//document.insert(process_attrs.begin(), process_attrs.end());
	for( auto const& attr : process_attrs ) {
		document[attr.first] = attr.second;
	}
	apply_reference_substitutions(lookup(document, "__pipeline__"),
	                              document_value);
	// Save the pipeline definition for use by dynamically-created tasks
	_definition = document_value;
	//std::cout << std::endl;
	//std::cout << document_value << std::endl;
	//std::string api_version = document["api_version"].get<std::string>();
	Object& pipeline = lookup_object(document, "__pipeline__");
	//pipeline_value.get<Object>();
	//cout << pipeline["name"] << endl;
	_name = lookup_string(pipeline, "__name__");
	//const Object& pipeline_attrs = get_key<Object>(pipeline, "attributes");
	//process_attrs.insert(pipeline_attrs.begin(), pipeline_attrs.end());
	// TODO: Interpret attrs
	// this->init(pipeline);
	
	// Note: This uses a trial-and-error approach to creating tasks that may
	//         have dependencies on other as-yet-uncreated tasks.
	Object new_tasks = lookup_object(pipeline, "__tasks__");
	Object deferred_tasks;
	size_t nsuccess = 0;
	while( !new_tasks.empty() ) {
		Object::iterator it = new_tasks.begin();
		// HACK TESTING
		std::string task_name = it->first;
		if( task_name == "__comment__" ) {
			new_tasks.erase(it);
			continue;
		}
		//Object& task_def = it->second.get<Object>();
		Value& task_def = it->second;
		// Automatically insert name attribute
		task_def.get<Object>().insert(Object::value_type("__name__",
		                                                 Value(task_name)));
		//try {
		//	// Note: Refs have already been substituted, so mustn't do again
		//	this->create_task(/*task_name, */task_def, false);
		//	++nsuccess;
		//}
		//catch( Task::DependencyError ) {
		//	deferred_tasks.insert(*it);
		//}
		if( this->create_task(task_def, false) ) {
			++nsuccess;
		}
		else {
			deferred_tasks.insert(*it);
		}
		new_tasks.erase(it);
		if( new_tasks.empty() ) {
			if( nsuccess == 0 ) {
				// TODO: Print out list of task names and dependencies in
				//         deferred_tasks to aid debugging.
				throw std::runtime_error("Pipeline contains missing or cyclic"
				                         " dependency");
			}
			std::swap(new_tasks, deferred_tasks);
		}
	}
	this->init();
}
Pipeline::task_pointer Pipeline::create_task(//std::string name,
                                             Value       definition_value,
                                             bool        substitute_refs) {
	Object& definition = definition_value.get<Object>();
	std::string name = lookup_string(definition, "__name__");
	/*
	// Allow specification of name via function arg or existing __name__ entry
	if( definition.count("__name__") ) {
		
	}
	else {
		definition.insert(Object::value_type("__name__", Value(name)));
	}
	*/
	if( substitute_refs ) {
		apply_reference_substitutions(definition_value, this->_definition);
	}
	std::string classname = lookup_string(definition, "__class__");
	const char* plugins_dir_c = getenv("BIFROST_PLUGINS_DIR");
	if( !plugins_dir_c ) {
		plugins_dir_c = "./plugins";
		this->log().warning("BIFROST_PLUGINS_DIR not set; using default of %s",
		                    plugins_dir_c);
		//std::cout << "Warning: BIFROST_PLUGINS_DIR not set; using default of "
		//          << plugins_dir_c << std::endl;
	}
	std::string plugins_dir = plugins_dir_c;
#if __linux__
	std::string libext = "so";
#elif __APPLE__
	std::string libext = "dylib";
#elif _WIN32
	std::string libext = "dll";
#else
	#error "Unknown OS; do not know dynamic library extension"
#endif
	std::string filename = plugins_dir + "/" + classname + "." + libext;
	auto lib_entry = std::make_pair(classname, DynLib(filename));
	DynLib& lib = _libs.insert(lib_entry).first->second;
	typedef Task* (*create_func_type)(Pipeline* , Object );
	create_func_type create_func = (create_func_type)lib.symbol("create");
	// Try to construct the task
	task_pointer task( (*create_func)(this, definition) );
	//if( !task ) {
	//	// Construction failed due to dependency constraint
	//	throw Task::DependencyError();
	//}
	if( task ) {
		auto ret = _tasks.insert(std::make_pair(name, task));
		if( !ret.second ) {
			throw std::runtime_error("Task with name "+name+" already exists");
		}
	}
	return task;
}
/*
void Pipeline::stop_task(std::string name) {
	auto task = lookup(_tasks, name);
	task->shutdown();
	task->wait();
}
void Pipeline::destroy_task(std::string name) {
	stop_task(name);
	_tasks.erase(name);
}
*/
Pipeline::ring_type* Pipeline::create_ring(std::string name,
                                           space_type  space) {
	// WAR for difficulty in non-default-constructing RingBuffer into map
	//   Should really use _rings.insert, but it's a pain to deal with
	bool already_exists = _rings.count(name);
	if( already_exists ) {
		throw std::invalid_argument("Ring already exists: "+name);
	}
	ring_type* ring = &_rings[name];
	//ring->set_allocator(allocator_type(space));
	ring->set_space(space);
	return ring;
}
Pipeline::ring_type* Pipeline::create_ring(std::string name,
                                           std::string space) {
	if( space == "system" ) {
		return this->create_ring(name, SPACE_SYSTEM);
	}
	else if( space == "cuda" ) {
		return this->create_ring(name, SPACE_CUDA);
	}
	else {
		throw std::invalid_argument("Invalid memory space");
	}
}
//template<class R>
//std::shared_ptr<R> get_ring(std::string name) {
Pipeline::ring_type* Pipeline::get_ring(std::string name) {
	return &lookup(_rings, name);
	//ring_map_type::iterator it = _rings.find(name);
	//if( it ==
	//// TODO: Support non-default construction of ring buffers
	//return &_rings[name];
}
void Pipeline::init() {
	for( task_map_type::iterator t=_tasks.begin(); t!=_tasks.end(); ++t ) {
		task_pointer task = t->second;
		task->init();
	}
	// Initialise pipeline-wide publish socket
	std::stringstream addr_ss;
	int port = lookup_integer(params(), "broadcast_port", DEFAULT_BROADCAST_PORT);
	// TODO: Bind to IP (or interface) to avoid broadcast over data network?
	addr_ss << "tcp://*:" << port;
	_pubsock.bind(addr_ss.str().c_str());
}
void Pipeline::launch() {
	for( task_map_type::iterator t=_tasks.begin(); t!=_tasks.end(); ++t ) {
		task_pointer task = t->second;
		task->launch();
	}
}
void Pipeline::shutdown() {
	for( task_map_type::iterator t=_tasks.begin(); t!=_tasks.end(); ++t ) {
		task_pointer task = t->second;
		task->shutdown();
	}
}
void Pipeline::wait() {
	for( task_map_type::iterator t=_tasks.begin(); t!=_tasks.end(); ++t ) {
		task_pointer task = t->second;
		task->wait();
	}
}
void Pipeline::broadcast(std::string topic,
                         Object      metadata,
                         char const* data,
                         size_t      size) const {
	topic = this->name() + "." + topic;
	metadata["__date__"]   = Value(get_current_utc_string());
	metadata["__time__"]   = Value((int64_t)get_current_clock_ns());
	metadata["__period__"] = Value((int64_t)1000000000ll);
	std::string s = topic + " " + Value(metadata).serialize();
	bool complete = !bool(data);
	_pubsock.send(s, complete);
	if( data ) {
		if( size == (size_t)-1 ) {
			// Allow passing C-style string as data without size
			size = std::strlen(data);
		}
		_pubsock.send(data, size);
	}
}
