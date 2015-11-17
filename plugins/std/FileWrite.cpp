
#include "FileWrite.hpp"

#include <bifrost/affinity.hpp>

#include <string>

Task* create(Pipeline*     pipeline,
             const Object* definition) {
	return new FileWrite(pipeline, definition);
}

FileWrite::FileWrite(Pipeline*     pipeline,
                     const Object* definition)
	: super_type({"data"}, // inputs
	             //{{1}},    // input shapes
	             {},       // input shapes
	             {},       // outputs
	             pipeline, definition) {}
void FileWrite::init() {
	super_type::init();
	std::string filename = lookup_string(params(), "filename");
	_file.open(filename);
}
void FileWrite::open() {
  auto cpu_cores = lookup_list<int>(params(), "cpu_cores", {});
  bind_to_core(cpu_cores[0]);
  return super_type::open();
}
void FileWrite::process() {
	std::cout << this->name() << "::process()" << std::endl;
	auto const& data_input = this->get_input<0>();
	std::cout << "** " << data_input.ring()->tail() << ", " << data_input.ring()->head() << std::endl;
	std::cout << "** " << data_input.frame0() << std::endl;
	_file.write(&data_input[0], data_input.size_bytes());
	_file.flush();
}
