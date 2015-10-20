
#include "FileWrite.hpp"

#include <string>

Task* create(Pipeline*     pipeline,
             const Object* definition) {
	return new FileWrite(pipeline, definition);
}

FileWrite::FileWrite(Pipeline*     pipeline,
                     const Object* definition)
	: super_type({"data"},
	             {},
	             pipeline, definition) {}
void FileWrite::init() {
	super_type::init();
	std::string filename = lookup_string(params(), "filename");
	_file.open(filename);
}
void FileWrite::process() {
	std::cout << this->name() << "::process()" << std::endl;
	auto const& data_input = this->get_input<0>();
	std::cout << "** " << data_input.ring()->tail() << ", " << data_input.ring()->head() << std::endl;
	std::cout << "** " << data_input.frame0() << std::endl;
	_file.write(&data_input[0], data_input.size_bytes());
	_file.flush();
}
