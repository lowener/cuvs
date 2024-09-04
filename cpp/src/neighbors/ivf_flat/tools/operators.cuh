#pragma once

namespace raft {

/**
 * @defgroup operators Commonly used functors.
 * The optional unused arguments are useful for kernels that pass the index along with the value.
 * @{
 */

struct identity_op {
  template <typename Type, typename... UnusedArgs>
  constexpr inline auto operator()(const Type& in, UnusedArgs...) const
  {
    return in;
  }
};
} // namespace raft