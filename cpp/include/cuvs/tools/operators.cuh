#pragma once

#include <cuda/std/type_traits>
#include <raft/core/detail/macros.hpp>

namespace cuvs::tools {

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

struct void_op {
  template <typename... UnusedArgs>
  constexpr inline void operator()(UnusedArgs...) const
  {
    return;
  }
};

template <typename OutT>
struct cast_op {
  template <typename InT, typename... UnusedArgs>
  constexpr inline auto operator()(InT in, UnusedArgs...) const
  {
    return static_cast<OutT>(in);
  }
};

struct key_op {
  template <typename KVP, typename... UnusedArgs>
  constexpr inline auto operator()(const KVP& p, UnusedArgs...) const
  {
    return p.key;
  }
};

struct value_op {
  template <typename KVP, typename... UnusedArgs>
  constexpr inline auto operator()(const KVP& p, UnusedArgs...) const
  {
    return p.value;
  }
};

struct sqrt_op {
  template <typename Type, typename... UnusedArgs>
  inline auto operator()(const Type& in, UnusedArgs...) const
  {
    return ::sqrt(in);
  }
};

struct nz_op {
  template <typename Type, typename... UnusedArgs>
  constexpr inline auto operator()(const Type& in, UnusedArgs...) const
  {
    return in != Type(0) ? Type(1) : Type(0);
  }
};

struct abs_op {
  template <typename Type, typename... UnusedArgs>
  inline auto operator()(const Type& in, UnusedArgs...) const
  {
    return ::abs(in);
  }
};

struct sq_op {
  template <typename Type, typename... UnusedArgs>
  constexpr inline auto operator()(const Type& in, UnusedArgs...) const
  {
    return in * in;
  }

  template <typename... UnusedArgs>
  constexpr inline auto operator()(const half& in, UnusedArgs...) const
  {
    return __half2float(in) * __half2float(in);
  }
};

struct add_op {
  template <typename T1, typename T2>
  constexpr inline auto operator()(const T1& a, const T2& b) const
  {
    if constexpr (cuda::std::is_same_v<T1, half> && cuda::std::is_same_v<T2, half>) {
      return __half2float(a) + __half2float(b);
    } else if constexpr (cuda::std::is_same_v<T1, half>) {
      return __half2float(a) + b;
    } else if constexpr (cuda::std::is_same_v<T2, half>) {
      return a + __half2float(b);
    } else {
      return a + b;
    }
  }
};

struct sub_op {
  template <typename T1, typename T2>
  constexpr inline auto operator()(const T1& a, const T2& b) const
  {
    return a - b;
  }
};

struct mul_op {
  template <typename T1, typename T2>
  constexpr inline auto operator()(const T1& a, const T2& b) const
  {
    return a * b;
  }
};

struct div_op {
  template <typename T1, typename T2>
  constexpr inline auto operator()(const T1& a, const T2& b) const
  {
    return a / b;
  }
};

struct div_checkzero_op {
  template <typename T1, typename T2>
  constexpr inline auto operator()(const T1& a, const T2& b) const
  {
    if (b == T2{0}) { return T1{0} / T2{1}; }
    return a / b;
  }
};

/**
 * @brief Wraps around a binary operator, passing a constant on the right-hand side.
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/operators.hpp>
 *
 *  raft::plug_const_op<float, raft::mul_op> op(2.0f);
 *  std::cout << op(2.1f) << std::endl;  // 4.2
 * @endcode
 *
 * @tparam ConstT
 * @tparam BinaryOpT
 */
template <typename ConstT, typename BinaryOpT>
struct plug_const_op {
  const ConstT c;
  const BinaryOpT composed_op;

  template <typename OpT     = BinaryOpT,
            typename UnusedT = cuda::std::enable_if_t<cuda::std::is_default_constructible_v<OpT>>>
  constexpr explicit plug_const_op(const ConstT& s)
    : c{s}, composed_op{}  // The compiler complains if composed_op is not initialized explicitly
  {
  }
  constexpr plug_const_op(const ConstT& s, BinaryOpT o) : c{s}, composed_op{o} {}

  template <typename InT>
  constexpr inline auto operator()(InT a) const
  {
    return composed_op(a, c);
  }
};

template <typename Type>
using add_const_op = plug_const_op<Type, add_op>;

template <typename Type>
using sub_const_op = plug_const_op<Type, sub_op>;

template <typename Type>
using mul_const_op = plug_const_op<Type, mul_op>;

template <typename Type>
using div_const_op = plug_const_op<Type, div_op>;

template <typename Type>
using div_checkzero_const_op = plug_const_op<Type, div_checkzero_op>;
/*
template <typename Type>
using pow_const_op = plug_const_op<Type, pow_op>;

template <typename Type>
using mod_const_op = plug_const_op<Type, mod_op>;

template <typename Type>
using mod_const_op = plug_const_op<Type, mod_op>;

template <typename Type>
using equal_const_op = plug_const_op<Type, equal_op>;*/
}  // namespace cuvs::tools