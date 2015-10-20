
#include "Depacketize.hpp"

extern "C" Task* create(Pipeline*     pipeline,
                        const Object* definition) {
	// Check construction order dependencies
	// If output space not specified explicitly, then cannot construct until
	//   input ring has been constructed.
	if( lookup_string(*definition, "output_space", "auto") == "auto" &&
	    !Task::input_ring_exists("payloads", pipeline, definition) ) {
		return 0;
	}
	return new Depacketize(pipeline, definition);
}
