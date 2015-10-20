
#pragma once

#include <pthread.h>
//#include <sched.h>
#include <unistd.h>
#include <errno.h>

// Note: Pass core_id = -1 to unbind
inline int bind_to_core(int core_id, pthread_t tid=0) {
#if __linux__
	// Check for valid core_id
	int num_cores = sysconf(_SC_NPROCESSORS_ONLN);
	if (core_id < -1 || core_id >= num_cores) {
		return EINVAL;
	}
	// Create core mask
	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	if( core_id >= 0 ) {
		// Set specified core
		CPU_SET(core_id, &cpuset);
	}
	else {
		// Set all valid cores (i.e., 'un-bind')
		for( int c=0; c<num_cores; ++c ) {
			CPU_SET(c, &cpuset);
		}
	}
	// Default to current thread
	if( tid == 0 ) {
		tid = pthread_self();
	}
	// Set affinity (note: non-portable)
	return pthread_setaffinity_np(tid, sizeof(cpu_set_t), &cpuset);
	//return sched_setaffinity(tid, sizeof(cpu_set_t), &cpuset);
#else
#warning CPU core binding/affinity not supported on this OS
#endif
}
