
/*
  A simple message logging class
  Ben Barsdell (2014)
  Apache v2 license
  
  Supports verbosity levels, timestamping, and mixing of
    C and C++ formatting syntax.

  TODO: Additional features to consider implementing:
          [DONE]Thread safety, via a mutex
          Log handler objects with filters and fixed-size history (e.g., limit file size)
          Publication of messages to remote subscribers
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdarg>
#include <cassert>
#include <mutex>

class Log {
	// TODO: Pretty sure this doesn't work when compiling shared libraries
	//         Need to use an extern global instead, and so need a Log.cpp
	inline static std::mutex& global_mutex() {
		static std::mutex m;
		return m;
	}
	struct nullstream_t : std::ostream {
		struct nullbuf : std::streambuf {
			int overflow(int c) { return traits_type::not_eof(c); }
		} m_sbuf;
		nullstream_t()
			: std::ios(&m_sbuf), std::ostream(&m_sbuf) {}
		// Note: Must explicitly define copy/assign because std::ostream does
		//         not allow them.
		nullstream_t(const nullstream_t& other)
			: std::ios(&m_sbuf), std::ostream(&m_sbuf) {}
		nullstream_t& operator=(const nullstream_t& other) {
			return *this;
		}
	};
	
	std::ostream*         _stream;
	mutable nullstream_t  _nullstream;
	std::vector<char>     _buffer;
	
	std::string utc_time_string(std::string format) const {
		time_t rawtime;
		time(&rawtime);
		char buffer[64];
		strftime(buffer, 64, format.c_str(), gmtime(&rawtime));
		return buffer;
	}
	std::string local_time_string(std::string format) const {
		time_t rawtime;
		time(&rawtime);
		char buffer[64];
		strftime(buffer, 64, format.c_str(), localtime(&rawtime));
		return buffer;
	}
	std::ostream& vlog(int level, std::string msg, va_list args) {
		if( level > this->verbosity ) {
			return _nullstream;
		}
		assert(_stream);
		std::lock_guard<std::mutex> lock(Log::global_mutex());
		std::ostream& stream = *_stream;
		stream << "[";
		if( this->use_timestamp ) {
			if( this->use_utc ) {
				stream << utc_time_string(this->time_format);
			}
			else {
				stream << local_time_string(this->time_format);
			}
		}
		if( !this->id.empty() ) {
			stream << " " << this->id;
		}
		stream << "] ";
		switch( level ) {
		case SLOG_EMERG:   stream << "**EMERGENCY** "; break;
		case SLOG_ALERT:   stream << "   *ALERT*    "; break;
		case SLOG_CRIT:    stream << "  =CRITICAL=  "; break;
		case SLOG_ERR:     stream << "   -ERROR-    "; break;
		case SLOG_WARNING: stream << "   WARNING    "; break;
		default: break;
		}
		// Note: The approach here enables mixing of both C and C++ style
		//         formatting syntax. This requires making the newline the
		//         user's responsibility.
		if( !msg.empty() ) {
			// Apply C-style formatting
			int ret = vsnprintf(&_buffer[0], _buffer.capacity(), msg.c_str(), args);
			if( (size_t)ret >= _buffer.capacity() ) {
				_buffer.resize(ret+1); // Note: +1 for NULL terminator
				ret = vsnprintf(&_buffer[0], _buffer.capacity(), msg.c_str(), args);
			}
			if( ret < 0 ) {
				// TODO: How to handle encoding error?
			}
			stream << &_buffer[0];
		}
		stream.flush();
		return stream; // Enables subsequent operations by the user
	}
public:
	// Note: Direct access for simplicity
	int           verbosity;
	std::string   id;
	bool          use_timestamp;
	bool          use_utc;
	std::string   time_format;
	enum {
		SLOG_EMERG   = 0, // system is unusable
		SLOG_ALERT   = 1, // action must be taken immediately
		SLOG_CRIT    = 2, // critical conditions
		SLOG_ERR     = 3, // error conditions
		SLOG_WARNING = 4, // warning conditions
		SLOG_NOTICE  = 5, // normal but significant condition
		SLOG_INFO    = 6, // informational
		SLOG_DEBUG   = 7, // debug-level messages
		SLOG_TRACE   = 8  // call-tracing messages
	};
	Log(std::string id_="")
		: _stream(&std::cout),
		  verbosity(SLOG_NOTICE),
		  id(id_),
		  use_timestamp(true),
		  use_utc(false),
		  time_format("%Y-%m-%d_%H:%M:%S") {
		_buffer.reserve(256); // Avoid reallocations until a long msg is written
	}
	      std::ostream& stream()       { return *_stream; }
	const std::ostream& stream() const { return *_stream; }
	void setStream(std::ostream& stream) { _stream = &stream; }
	
	std::ostream& log(int level, std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(level, msg, va); va_end(va); return s;
	}
	std::ostream& critical(std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(SLOG_CRIT, msg, va); va_end(va); return s;
	}
	std::ostream& error(std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(SLOG_ERR, msg, va); va_end(va); return s;
	}
	std::ostream& warning(std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(SLOG_WARNING, msg, va); va_end(va); return s;
	}
	std::ostream& notice(std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(SLOG_NOTICE, msg, va); va_end(va); return s;
	}
	std::ostream& info(std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(SLOG_INFO, msg, va); va_end(va); return s;
	}
	std::ostream& debug(std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(SLOG_DEBUG, msg, va); va_end(va); return s;
	}
	std::ostream& trace(std::string msg="", ...) {
		va_list va; va_start(va, msg); std::ostream& s = vlog(SLOG_TRACE, msg, va); va_end(va); return s;
	}
};
