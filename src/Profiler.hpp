
/*
  TODO: Consider re-implementing using std::chrono (should be an easy port)
        Add nicer NVVP integration as per this:
	  http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-generate-custom-application-profile-timelines-nvtx/
 */

#pragma once

#include <list>
#include <string>
#include <iostream>

#ifdef PROFILER_USE_GETTIMEOFDAY
#include <sys/time.h> // For gettimeofday(2)
#else
//#include <time.h>     // For clock_gettime
#include "time_portable.h" // For cross-platform clock_gettime
#endif

#ifdef PROFILER_USE_NVTX
#include <nvToolsExt.h> // For NVIDIA Visual Profiler integration (nvtx)
#endif

#include "Object.hpp"

// This is designed for a single thread
// Multiple threads could be done via a vector of curnodes,
//   but probably easier just to use separate instances.
class Profiler {
public:
	typedef long time_type;
private:
	struct ProfilerNode {
		typedef std::list<ProfilerNode>    child_list;
		typedef child_list::iterator       iterator;
		typedef child_list::const_iterator const_iterator;
		ProfilerNode* parent;
		child_list    children;
		std::string   name;
		time_type     begin_time;
		time_type     end_time;
		ProfilerNode(ProfilerNode* parent_,
		             std::string   name_,
		             time_type     begin_time_,
		             time_type     end_time_=-1)
			: parent(parent_),
			  name(name_),
			  begin_time(begin_time_),
			  end_time(end_time_) {}
		operator Object() const {
			Object ret;
			ret["name"]       = Value(name);
			ret["begin_time"] = Value((int64_t)begin_time);
			ret["end_time"]   = Value((int64_t)end_time);
			ret["duration"]   = Value((int64_t)(end_time - begin_time));
			List child_values;
			child_values.reserve(children.size());
			for( ProfilerNode::const_iterator it=children.begin();
			     it!=children.end(); ++it ) {
				child_values.push_back(Value((Object)*it));
			}
			ret["children"] = Value(child_values);
			return ret;
		}
	};
	ProfilerNode  m_rootnode;
	ProfilerNode* m_curnode;
	time_type     m_reftime;
	
	// TODO: These may be better implemented as methods of ProfilerNode instead
	// Deletes nodes whose end times fall before time_cut
	void clear_old_impl(time_type time_cut, ProfilerNode& node) {
		ProfilerNode::iterator it = node.children.begin();
		while( it != node.children.end() ) {
			ProfilerNode& child = *it;
			if( child.begin_time >= time_cut ) {
				// Do nothing
				++it;
			}
			else if( child.end_time < time_cut ) {
				// Delete node
				it = node.children.erase(it);
			}
			else {
				// Recurse into child node
				clear_old_impl(time_cut, child);
				++it;
			}
		}
	}
	void export_json_impl(std::ostream&       stream,
	                      const ProfilerNode& node,
	                      size_t              depth=0) const {
		std::string indent(depth, '\t');
		stream << indent << "{\n"
		       << indent << "\t\"name\":       \"" << node.name << "\"" << ",\n"
		       << indent << "\t\"begin_time\": "   << node.begin_time   << ",\n"
		       << indent << "\t\"end_time\":   "   << node.end_time     << ",\n"
		       << indent << "\t\"duration\":   "   << node.end_time-node.begin_time << ",\n"
		       << indent << "\t\"children\":   [";
		// Recursively export children
		bool first = true;
		for( ProfilerNode::const_iterator it=node.children.begin();
		     it!=node.children.end();
		     ++it ) {
			const ProfilerNode& child = *it;
			// Silly json not supporting trailing commas!
			if( !first ) {
				// Insert comma
				stream << ",";
			}
			else {
				first = false;
			}
			stream << "\n";
			export_json_impl(stream, child, depth+1);
		}
		stream << "]\n"
		       << indent << "}";
	}
public:
	enum {
		PERIOD = 1000000000l
	};
	// Returns time in seconds since the epoch
	static time_type time() {
#ifdef PROFILER_USE_GETTIMEOFDAY
		struct timeval tv;
		assert( gettimeofday(&tv, 0) == 0 );
		//return tv.tv_sec + 1e-6*tv.tv_usec;
		return (tv.tv_sec*(time_type)PERIOD +
		        tv.tv_usec*(time_type)(PERIOD/1000000l));
#else
		struct timespec ts;
		assert( clock_gettime(CLOCK_MONOTONIC, &ts) == 0 );
		//return ts.tv_sec + 1e-9*ts.tv_nsec;
		return (ts.tv_sec*(time_type)PERIOD +
		        ts.tv_nsec*(time_type)(PERIOD/1000000000l));
#endif
	}
	
	Profiler(std::string name="Application",
	         time_type      reference_time=Profiler::time())
		: m_rootnode(0, name, 0.f), m_curnode(&m_rootnode),
		  m_reftime(reference_time) {
		// TODO: This doesn't work because the obj is created prior to spawning the thread
		//nvtxNameOsThread(pthread_self(), name.c_str()); // TESTING
	}
	std::string name() const { return m_rootnode.name; }
	void set_name(std::string name) {
		m_rootnode.name = name;
	}
	time_type reference_time() const { return m_reftime; }
	void set_reference_time(time_type reftime) { m_reftime = reftime; }
	void push(std::string name) {
		time_type curtime = this->split();
		ProfilerNode newnode(m_curnode, name, curtime);
		m_curnode->children.push_back(newnode);
		m_curnode = &m_curnode->children.back();
#ifdef PROFILER_USE_NVTX
		nvtxRangePushA(name.c_str());
#endif
	}
	void pop() {
#ifdef PROFILER_USE_NVTX
		nvtxRangePop();
#endif
		
		m_curnode->end_time = this->split();
		m_curnode = m_curnode->parent;
	}
	void event(std::string name) {
		time_type curtime = this->split();
		// Note: This adds an 'impulse' node
		ProfilerNode newnode(m_curnode, name, curtime, curtime);
		m_curnode->children.push_back(newnode);
	}
	void clear() {
		m_rootnode.children.clear();
	}
	// Delete data older than max_age
	void clear_old(time_type max_age) {
		time_type time_cut = this->split() - max_age;
		this->clear_old_impl(time_cut, m_rootnode);
	}
	void export_json(std::ostream& stream) const {
		stream << "{\n";
		stream << "\t\"period\":  " << PERIOD << ",\n";
		stream << "\t\"profile\": ";
		this->export_json_impl(stream, m_rootnode, 1);
		stream << "\n}";
		//stream.flush();
	}
	Object export_object() const {
		Object ret;
		ret["period"]  = Value((int64_t)(time_type)PERIOD);
		ret["profile"] = Value((Object)m_rootnode);
		return ret;
	}
	time_type split() const {
		return Profiler::time() - m_reftime;
	}
};
// RAII wrapper object for convenience
class ScopedTracer {
	Profiler& m_profiler;
public:
	ScopedTracer(Profiler& profiler, std::string name)
		: m_profiler(profiler) {
		m_profiler.push(name);
	}
	~ScopedTracer() {
		m_profiler.pop();
	}
	      Profiler& profiler()       { return m_profiler; }
	const Profiler& profiler() const { return m_profiler; }
};
