
/*
  Wrapper for iterator-pairs that allows use in range-based for statements
  From http://stackoverflow.com/a/23223402
*/

#pragma once

#include <utility>

template<class Iter>
struct range_pair : public std::pair<Iter, Iter> {
    using pair_t = std::pair<Iter, Iter>;
    range_pair(pair_t&& src)
    : std::pair<Iter, Iter>(std::forward<pair_t>(src))
    {}

    using std::pair<Iter, Iter>::first;
    using std::pair<Iter, Iter>::second;

    Iter begin() const { return first; }
    Iter end() const { return second; }
};

template<class Iter>
range_pair<Iter> make_range(std::pair<Iter, Iter> p) {
    return range_pair<Iter>(std::move(p));
}

template<class Iter>
range_pair<Iter> make_range(Iter i1, Iter i2) {
    return range_pair<Iter>(std::make_pair(std::move(i1), std::move(i2)));
}
