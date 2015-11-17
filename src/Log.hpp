
/*
  Logger class that calls owner->broadcast()
 */

#pragma once

#include "Object.hpp"

#include <vector>
#include <mutex>
#include <stdarg.h>

template<class T>
class Logger {
public:
	enum {
		LOGGER_EMERG   = 0, // system is unusable
		LOGGER_ALERT   = 1, // action must be taken immediately
		LOGGER_CRIT    = 2, // critical conditions
		LOGGER_ERR     = 3, // error conditions
		LOGGER_WARNING = 4, // warning conditions
		LOGGER_NOTICE  = 5, // normal but significant condition
		LOGGER_INFO    = 6, // informational
		LOGGER_DEBUG   = 7, // debug-level messages
		LOGGER_TRACE   = 8  // call-tracing messages
	};
	typedef T owner_type;
	Logger(owner_type* owner=0) : _owner(owner) {
		if( !_owner ) {
			// Allow use as mixin
			_owner = (owner_type*)this;
		}
	}
	void set_verbosity(int v)  { _verbosity = v; }
	int      verbosity() const { return _verbosity; }
	void log(std::string topic, std::string msg, ...) const {
		va_list va; va_start(va, msg); this->vlog(topic, msg, va); va_end(va);
	}
#define LOGGER_DEFINE_LOG_LEVEL(name, level)	  \
	void name(std::string msg, ...) const { \
		if( level <= _verbosity ) { \
			va_list va; va_start(va, msg); this->log(#name, msg, va); va_end(va); \
		} \
	}
	LOGGER_DEFINE_LOG_LEVEL(critical, LOGGER_CRIT)
	LOGGER_DEFINE_LOG_LEVEL(error,    LOGGER_ERR)
	LOGGER_DEFINE_LOG_LEVEL(warning,  LOGGER_WARNING)
	LOGGER_DEFINE_LOG_LEVEL(notice,   LOGGER_NOTICE)
	LOGGER_DEFINE_LOG_LEVEL(info,     LOGGER_INFO)
	LOGGER_DEFINE_LOG_LEVEL(debug,    LOGGER_DEBUG)
	LOGGER_DEFINE_LOG_LEVEL(trace,    LOGGER_TRACE)
#undef LOGGER_DEFINE_LOG_LEVEL
private:
	void vlog(std::string topic, std::string msg, va_list args) const {
		// Note: Must lock due to use of _buffer
		std::lock_guard<std::mutex> lock(_mutex);
		int ret = vsnprintf(&_buffer[0], _buffer.capacity(),
		                    msg.c_str(), args);
		if( (size_t)ret >= _buffer.capacity() ) {
			_buffer.resize(ret+1); // Note: +1 for NULL terminator
			ret = vsnprintf(&_buffer[0], _buffer.capacity(),
			                msg.c_str(), args);
		}
		_buffer.resize(ret+1); // Note: +1 for NULL terminator
		if( ret < 0 ) {
			// TODO: How to handle encoding error?
		}
		Object metadata; // Note: No metadata is used here
		_owner->broadcast("log."+topic, metadata, &_buffer[0],
		                  _buffer.size());
	}
	owner_type*               _owner;
	mutable std::mutex        _mutex;
	mutable std::vector<char> _buffer;
	int                       _verbosity;
};
