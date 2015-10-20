
/*
  Similar to atomic_flag but provides a (timed) wait() method,
    acting like Python's threading.Event.

  This can be used to implement an interruptable sleep
    if( interrupt_event.wait_for(sleep_time) ) {
      // Sleep was interrupted
    }
 */

#pragma once

#include <mutex>
#include <condition_variable>
#include <chrono>

// Based on code from here: http://stackoverflow.com/a/14921115
class event_flag {
    mutable std::mutex m_;
    mutable std::condition_variable cv_;
    bool flag_;
public:
    event_flag() : flag_(false) {}
    bool is_set() const {
        std::lock_guard<std::mutex> lock(m_);
        return flag_;
    }
    void set() {
	    std::lock_guard<std::mutex> lock(m_);
	    flag_ = true;
	    cv_.notify_all();
    }
	//void reset() {
	void clear() {
		std::lock_guard<std::mutex> lock(m_);
		flag_ = false;
		cv_.notify_all();
	}
	void wait() const {
		std::unique_lock<std::mutex> lock(m_);
		cv_.wait(lock, [this] { return flag_; });
	}
	template<typename Rep, typename Period>
	bool wait_for(const std::chrono::duration<Rep, Period>& rel_time) const {
		std::unique_lock<std::mutex> lock(m_);
		return cv_.wait_for(lock, rel_time, [this] { return flag_; });
	}
	template<typename Clock, typename Duration>
	bool wait_until(const std::chrono::time_point<Clock,Duration>& abs_time) const {
		std::unique_lock<std::mutex> lock(m_);
		return cv_.wait_until(lock, abs_time, [this] { return flag_; });
	}
};
