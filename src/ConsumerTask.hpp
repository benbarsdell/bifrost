
/*
  TODO: This contains a lot of messy stuff due mostly to lack of generic lambdas
 */

#include "Task.hpp"
#include "utils.hpp"

#include <tuple>

// TODO: Replace with generic lambdas in C++14
// Note: These can't be in local scope due to this C++ limitation:
//         "invalid declaration of member template in local class"
template<typename InputsType>
struct init_input {
	Task*                           task;
	InputsType&                     inputs;
	std::vector<std::string> const& names;
	init_input(Task* task_,
	           InputsType& inputs_,
	           std::vector<std::string> const& names_)
		: task(task_), inputs(inputs_), names(names_) {}
	template<typename StaticIndex>
	void operator()(StaticIndex ) const {
		std::string name = names[StaticIndex::value];
		//std::cout << "*** " << name << ", " << task->get_input_ring(name) << std::endl;
		std::get<StaticIndex::value>(inputs).init(task->get_input_ring(name));
	}
};
template<typename OutputsType>
struct init_output {
	Task*                           task;
	OutputsType&                    outputs;
	std::vector<std::string> const& names;
	init_output(Task* task_,
	            OutputsType& outputs_,
	            std::vector<std::string> const& names_)
		: task(task_), outputs(outputs_), names(names_) {}
	template<typename StaticIndex>
	void operator()(StaticIndex ) const {
		std::string name = names[StaticIndex::value];
		//std::cout << "*** " << name << ", " << task->get_input_ring(name) << std::endl;
		std::get<StaticIndex::value>(outputs).init(task->get_output_ring(name));
	}
};
/*
template<typename OutputsType>
struct init_output {
	OutputsType&                    outputs;
	std::vector<std::string> const& names;
	init_output(OutputsType& outputs_,
	           std::vector<std::string> const& names_)
		: outputs(outputs_), names(names_) {}
	template<typename T>
	void operator()(T static_index) const {
		std::string name = names[static_index::value];
		std::get<static_index::value>(outputs).init(task->get_output_ring(name),
		                                            ??);//shape);
	}
};
*/
struct request_size_frames {
	ssize_t nframe;
	ssize_t nframe_buf;
	request_size_frames(ssize_t nframe_,
	                    ssize_t nframe_buf_)
		: nframe(nframe_), nframe_buf(nframe_buf_) {}
	template<typename B>
	void operator()(B& buffer) {
		//std::cout << "* " << nframe << ", " << nframe_buf << std::endl;
		buffer.request_size_frames(nframe, nframe_buf);
	}
};
struct open_input {
	bool guarantee;
	open_input(bool guarantee_) : guarantee(guarantee_) {}
	template<typename T>
	void operator()(T& input) { input.open(guarantee); }
};
struct input_valid {
	template<typename T>
	bool operator()(T const& input) { return input.still_valid(); }
};
struct advance_input {
	template<typename T>
	void operator()(T& input) { ++input; }
};
struct close_input {
	template<typename T>
	void operator()(T& input) { input.close(); }
};
struct shutdown_input {
	template<typename T>
	void operator()(T& input) { input.shutdown(); }
};

template<typename InputTypes,
         typename OutputTypes>
class ConsumerTask2;

// WAR for C++ suckiness
template<class CT, bool HAVE_INPUTS, bool HAVE_OUTPUTS>
struct RequestPrimarySize {
	RequestPrimarySize(CT* task_) {}
	size_t operator()(size_t buffer_factor) { return 0; }
};
template<class CT, bool HAVE_OUTPUTS>
struct RequestPrimarySize<CT,true,HAVE_OUTPUTS> {
	CT* task;
	RequestPrimarySize(CT* task_) : task(task_) {}
	size_t operator()(size_t buffer_factor) {
		auto& primary_input = std::get<0>(task->_inputs);
		if( task->params().count("gulp_nframe") ) {
			// Gulps specified exactly in no. frames
			size_t gulp_nframe = lookup_integer(task->params(), "gulp_nframe");
			primary_input.request_size_frames(gulp_nframe,
			                                  gulp_nframe*buffer_factor);
		}
		else {
			// Gulps specified by memory size upper limit
			size_t gulp_size_min = lookup_integer(task->params(), "gulp_size",
			                                      CT::DEFAULT_GULP_SIZE_MIN);
			primary_input.request_size_bytes(gulp_size_min,
			                                 gulp_size_min*buffer_factor);
		}
		return primary_input.size();
	}
};
template<class CT>
struct RequestPrimarySize<CT,false,true> {
	CT* task;
	RequestPrimarySize(CT* task_) : task(task_) {}
	size_t operator()(size_t buffer_factor) {
		auto& primary_output = std::get<0>(task->_outputs);
		//std::cout << "RequestPrimarySize: " << task << std::endl;
		if( task->params().count("gulp_nframe") ) {
			// Gulps specified exactly in no. frames
			size_t gulp_nframe = lookup_integer(task->params(), "gulp_nframe");
			primary_output.request_size_frames(gulp_nframe,
			                                   gulp_nframe*buffer_factor);
		}
		else {
			// Gulps specified by memory size upper limit
			size_t gulp_size_min = lookup_integer(task->params(), "gulp_size",
			                                      CT::DEFAULT_GULP_SIZE_MIN);
			//task->Task::log_info("gulp_size_min: %lu", gulp_size_min);
			//std::cout << "gulp_size_min: " << gulp_size_min << std::endl;
			primary_output.request_size_bytes(gulp_size_min,
			                                  gulp_size_min*buffer_factor);
		}
		return primary_output.size();
	}
};

// Input/OutputTypes are tuples of types
template<typename InputTypes,
         typename OutputTypes=std::tuple<> >
class ConsumerTask2 : public Task {
	typedef Task super_type;
	typedef typename tuple_of<RingReader,  InputTypes>::type  inputs_type;
	typedef typename tuple_of<RingWriter, OutputTypes>::type outputs_type;
public:
	enum {
		DEFAULT_GULP_SIZE_MIN = 67108864,
		DEFAULT_BUFFER_FACTOR = 3
	};
	ConsumerTask2(std::vector<std::string> input_names,
	              std::vector<std::string> output_names,
	              Pipeline*     pipeline,
	              const Object* definition)
		: Task(pipeline, definition),
		  _input_names(input_names),
		  _output_names(output_names) {

		// Note: Subclasses can call ring->set_space and/or ring->set_shape in
		//         their constructor to override the defaults here.
		std::string default_output_space = "system";
		for( auto const& name : output_names ) {
			// HACK TODO: This try/catch is a WAR for constructors being called
			//              multiple times when trying to resolve task
			//              dependencies. The dependency issue may need to be
			//              re-thought.
			try {
				this->create_output(name, default_output_space);
			}
			catch( std::invalid_argument ) {}
		}
	}
	template<int I>
	typename std::tuple_element<I,inputs_type>::type&        get_input()        { return std::get<I>(_inputs); }
	template<int I>
	typename std::tuple_element<I,inputs_type>::type const&  get_input()  const { return std::get<I>(_inputs); }
	template<int I>
	typename std::tuple_element<I,outputs_type>::type&       get_output()       { return std::get<I>(_outputs); }
	template<int I>
	typename std::tuple_element<I,outputs_type>::type const& get_output() const { return std::get<I>(_outputs); }
protected:
	template<class CT, bool HI, bool HO>
	friend class RequestPrimarySize;
	
	virtual void init() {
		auto in_inds = make_index_tuple(_inputs);
		auto in_func = init_input<inputs_type>(this, _inputs, _input_names);
		for_each(in_inds, in_func);
		
		auto out_inds = make_index_tuple(_outputs);
		auto out_func = init_output<outputs_type>(this, _outputs, _output_names);
		for_each(out_inds, out_func);
		
		size_t buffer_factor = lookup_integer(params(), "buffer_factor",
		                                      DEFAULT_BUFFER_FACTOR);
		size_t nframe = RequestPrimarySize<ConsumerTask2,
		                                   std::tuple_size<inputs_type>::value,
		                                   std::tuple_size<outputs_type>::value>(this)(buffer_factor);
		// Now make all inputs and outputs request the same no. frames
		// Note: Doing this twice for the primary input or output is fine
		//transform(_inputs, request_size_frames(nframe,
		for_each(_inputs, request_size_frames(nframe,
		                                      nframe*buffer_factor));
		//transform(_outputs, request_size_frames(nframe,
		for_each(_outputs, request_size_frames(nframe,
		                                       nframe*buffer_factor));
	}
	virtual void open() {
		// TODO: Replace with generic lambda in C++14
		bool guarantee_reads = lookup_bool(params(), "guarantee_reads", false);
		for_each(_inputs, open_input(guarantee_reads));
	}
	virtual bool still_valid() {
		return reduce(transform(_inputs, input_valid()),
		              true,
		              std::logical_and<bool>());
	}
	virtual void advance() {
		for_each(_inputs, advance_input());
	}
	virtual void close() {
		for_each(_inputs, close_input());
	}
	virtual void process() = 0; // Process input buffers
	// Called in 'advance' if input buffers were overwritten during processing
	virtual void overwritten() {
		this->log_error("Input data overwritten");
		throw std::out_of_range("Input data overwritten");
	}
	
	inline virtual void shutdown() {
		super_type::shutdown();
		for_each(_inputs, shutdown_input());
	}
	virtual void main() {
		//std::cout << "Before " << this->name() << "::open" << std::endl;
		try {
			this->open();
		}
		catch( RingBuffer::ShutdownError ) {}
		//std::cout << "After " << this->name() << "::open" << std::endl;
		while( !this->shutdown_requested() ) {
			{
				//std::cout << "Before " << this->name() << "::process" << std::endl;
				ScopedTracer trace(this->profiler(), "process");
				//if( _enabled_sequence.pop() ) {
				this->process();
				//std::cout << "After " << this->name() << "::process" << std::endl;
				//}
			}
			{
				ScopedTracer trace(this->profiler(), "advance");
				if( this->still_valid() ) {
					//std::cout << "Before " << this->name() << "::advance" << std::endl;
					try {
						this->advance();
					}
					catch( RingBuffer::ShutdownError ) {}
					//std::cout << "After " << this->name() << "::advance" << std::endl;
					//catch( std::out_of_range ) {
						//this->overwritten();
					//	throw std::runtime_error("Unexpected overwrite condition (check code!)");
					//}
				}
				else {
					//std::cout << "Before " << this->name() << "::overwritten" << std::endl;
					try {
						this->overwritten();
					}
					catch( RingBuffer::ShutdownError ) {}
					//std::cout << "After " << this->name() << "::overwritten" << std::endl;
				}
			}
			this->broadcast("perf", this->profiler().export_object());
			this->profiler().clear();
		}
		this->close();
	}
private:
	inputs_type  _inputs;
	outputs_type _outputs;
	std::vector<std::string> _input_names;
	std::vector<std::string> _output_names;
};
