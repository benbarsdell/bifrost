
#pragma once

#include <bifrost/ConsumerTask.hpp>

#include <iostream> // Debugging only
#include <fstream>

typedef std::tuple<char // data
                   >     input_types;
typedef std::tuple<>     output_types;

class FileWrite
	: public ConsumerTask2<input_types,output_types> {
	typedef  ConsumerTask2<input_types,output_types> super_type;
public:
	FileWrite(Pipeline*     pipeline,
	        const Object* definition);
  virtual void open();
	virtual void init();
	virtual void process();
private:
	std::ofstream _file;
};
