
/*
  Note: Due to ZeroMQ's async send support, which requires taking
          ownership of the send buffer, it is in general very
          difficult to avoid reallocations and memcpy'ing.
 */

#pragma once

#include "Object.hpp"

//#include <zmq.hpp>
#include "zmq.hpp"

class ZMQSocket : public zmq::socket_t {
	bool   _async;
public:
	class TimeoutError : public std::runtime_error {
		typedef std::runtime_error super_t;
	public:
		TimeoutError(const std::string& what_arg) : super_t(what_arg) {}
	};
	inline ZMQSocket(zmq::context_t& ctx, int type);
	void set_async(bool async) {
		_async = async;
	}
	void set_timeout(double secs) {
		int ms = secs * 1000;
		this->setsockopt(ZMQ_RCVTIMEO, &ms, sizeof(ms));
		this->setsockopt(ZMQ_SNDTIMEO, &ms, sizeof(ms));
	}
	
	bool send(zmq::message_t& msg, bool complete=true, int flags=0) {
		if( !complete ) {
			flags |= ZMQ_SNDMORE;
		}
		if( _async ) {
			flags |= ZMQ_DONTWAIT;
		}
		return zmq::socket_t::send(msg, flags);
	}
	bool send(char const* data, size_t size, bool complete=true, int flags=0) {
		zmq::message_t msg(size);
		::memcpy(msg.data(), data, size);
		return this->send(msg, complete, flags);
	}
	bool send(std::string str, bool complete=true, int flags=0) {
		return this->send(str.data(), str.size(), complete, flags);
	}
	/*
	//bool send_raw(const std::string& msg, bool complete=true, int flags=0) {
	bool send_raw(const char* begin,
	              const char* end,
	              //bool complete=true,
	              int flags=0) {
		zmq::message_t message(end-begin);
		::memcpy(message.data(), begin, end-begin);
		//if( !complete ) {
		//	flags |= ZMQ_SNDMORE;
		//}
		if( _async ) {
			flags |= ZMQ_DONTWAIT;
		}
		return zmq::socket_t::send(message, flags);
	}
	bool send(const Value& val, bool complete=true, int flags=0) {
		const std::string& str = val.serialize();
		return this->send_raw(str.data(), str.data()+str.size(), flags);
	}
	*/
	/*
	template<typename T>
	bool send(const T& val, bool complete=true, int flags=0) {
		return this->send(Value(val), complete, flags);
	}
	*/
	/*
	// Send bytes directly from typed variable
	template<typename T>
	bool send(const T& val, bool complete=true, int flags=0) {
		return this->send_raw(&val, &val + sizeof(val), complete, flags);
	}
	*/
	/*
	bool recv_raw(std::string* msg, int flags=0) {
		if( _async ) {
			flags |= ZMQ_DONTWAIT;
		}
		// TODO: Check for socket option ZMQ_RCVMORE and loop until it is 0
		zmq::message_t message;
		bool ret = zmq::socket_t::recv(&message, flags);
		msg->assign(static_cast<char*>(message.data()), message.size());
		return ret;
	}
	*/
	typedef zmq::message_t message_type;
	
	inline void recv_raw(message_type* message, int flags=0) {
		if( _async ) {
			flags |= ZMQ_DONTWAIT;
		}
		// TODO: Check for socket option ZMQ_RCVMORE and loop until it is 0
		//message_type message;
		if( !zmq::socket_t::recv(message, flags) ) {
			throw TimeoutError("Socket recv timed out");
		}
		//return message;
	}
	inline std::string recv_string(int flags=0) {
		message_type message;
		this->recv_raw(&message, flags);
		return std::string((char*)message.data(),
		                   (char*)message.data() + message.size());
	}
	inline Value recv_json(int flags=0) {
		//message_type message = this->recv_raw(flags);
		message_type message;
		this->recv_raw(&message, flags);
		//std::cout << "Raw msg: " << std::string((char*)message.data(),
		//                                        (char*)message.data() + message.size()) << std::endl;
		return parse_value((char*)message.data(),
		                   (char*)message.data() + message.size());
	}
	/*
	bool recv(Value* val, int flags=0) {
		std::string packed;
		bool ret = this->recv_raw(&packed, flags);
		if( ret ) {
			*val = parse_value(packed);
		}
		return ret;
	}
	*/
	/*
	template<typename T>
	bool recv(T* tval, int flags=0) {
		Value val;
		bool ret = this->recv(&val, flags);
		if( ret ) {
			*tval = val.get<T>();
		}
		return ret;
	}
	*/
	/*
	// Receive bytes directly into typed variable
	template<typename T>
	bool recv(T* tval, int flags=0) {
		//std::string msg;
		message_type msg;
		bool ret = this->recv_raw(&msg, flags);
		if( ret ) {
			assert( msg.size() >= sizeof(T) );
			::memcpy(tval, msg.data(), sizeof(T));
		}
		return ret;
	}
	*/
};
#define DEFINE_ZMQ_SOCKET_TYPE_CLASS(type)	\
	class type##Socket : public ZMQSocket { \
	public: \
		type##Socket(zmq::context_t& ctx) : ZMQSocket(ctx, ZMQ_##type) {} \
	}
DEFINE_ZMQ_SOCKET_TYPE_CLASS(PUB);
DEFINE_ZMQ_SOCKET_TYPE_CLASS(SUB);
DEFINE_ZMQ_SOCKET_TYPE_CLASS(PUSH);
DEFINE_ZMQ_SOCKET_TYPE_CLASS(PULL);
DEFINE_ZMQ_SOCKET_TYPE_CLASS(REQ);
DEFINE_ZMQ_SOCKET_TYPE_CLASS(REP);
#undef DEFINE_ZMQ_SOCKET_TYPE_CLASS

inline ZMQSocket::ZMQSocket(zmq::context_t& ctx, int type)
	: zmq::socket_t(ctx, type), _async(false) {}
/*
// TODO: This is untested but should work
// This can be used to create an ostream object that wraps a socket
//ZMQStreamBuffer buf(&socket);
//std::ostream    stream(&buf);
//Log             log(&stream);
class ZMQStreamBuffer : public: std::stringbuf {
	typedef std::stringbuf super_type;
protected:
	// Flush to socket
	virtual int sync() {
		_socket->send(this->str(), true, _flags);
		this->str(""); // Clear buffer
		return 0;
	}
	// Read from socket
	virtual int underflow() {
		int ret = super_type::underflow();
		if( ret == char_traits<char>::eof() ) {
			// ** TODO: Need to reset read pointer?
			this->str(_socket->recv_string(_flags));
			ret = super_type::underflow();
		}
		return ret;
	}
public:
	ZMQStreamBuffer(ZMQSocket* socket, int flags=0)
		: _socket(socket), _flags(flags) {}
private:
	ZMQSocket* _socket;
	int        _flags;
};

class ZMQStream : public std::ostream {
	typedef std::ostream super_type;
public:
	ZMQStream(ZMQSocket* socket, int flags=0)
		: super_type(&_buf), _buf(socket, flags) {}
private:
	ZMQStreamBuffer _buf;
};
*/
