#pragma once
#include "bitonic.cuh"
#include "integer_utils.cuh"
#include "operators.cuh"
#include "pow2_utils.cuh"

namespace cuvs::tools::matrix::detail::select::warpsort {

namespace {

/** Whether 'left` should indeed be on the left w.r.t. `right`. */
template <bool Ascending, typename T>
_RAFT_DEVICE _RAFT_FORCEINLINE auto is_ordered(T left, T right) -> bool
{
  if constexpr (Ascending) { return left < right; }
  if constexpr (!Ascending) { return left > right; }
}

}  // namespace

template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort {
  static_assert(is_a_power_of_two(Capacity));
  static_assert(cuda::std::is_default_constructible_v<IdxT>);

 public:
  /**
   *  The `empty` value for the chosen binary operation,
   *  i.e. `Ascending ? upper_bound<T>() : lower_bound<T>()`.
   */
  static constexpr T kDummy = Ascending ? upper_bound<T>() : lower_bound<T>();
  /** Width of the subwarp. */
  static constexpr int kWarpWidth = cuvs::tools::min<int>(Capacity, WarpSize);
  /** The number of elements to select. */
  const int k;

  /** Extra memory required per-block for keeping the state (shared or global). */
  constexpr static auto mem_required(uint32_t block_size) -> size_t { return 0; }

  /**
   * Construct the warp_sort empty queue.
   *
   * @param k
   *   number of elements to select.
   */
  _RAFT_DEVICE warp_sort(int k) : k(k)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_arr_[i] = kDummy;
      idx_arr_[i] = IdxT{};
    }
  }

  /**
   * Load k values from the pointers at the given position, and merge them in the storage.
   *
   * When it actually loads the values, it always performs some collective warp operations in the
   * end, thus enforcing warp sync. This means, it's safe to call `store` with the same arguments
   * after `load_sorted` without extra sync. Note, however, that this is not necessarily true for
   * the reverse order, because the access patterns of `store` and `load_sorted` are different.
   *
   * @param[in] in
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] in_idx
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] do_merge
   *    must be the same for all threads within a subwarp of size `kWarpWidth`.
   *    It serves as a conditional; when `false` the function does nothing.
   *    We need it to ensure threads within a full warp don't diverge calling `bitonic::merge()`.
   */
  _RAFT_DEVICE void load_sorted(const T* in, const IdxT* in_idx, bool do_merge = true)
  {
    if (do_merge) {
      int idx = Pow2<kWarpWidth>::mod(laneId()) ^ Pow2<kWarpWidth>::Mask;
#pragma unroll
      for (int i = kMaxArrLen - 1; i >= 0; --i, idx += kWarpWidth) {
        if (idx < k) {
          T t = in[idx];
          if (is_ordered<Ascending>(t, val_arr_[i])) {
            val_arr_[i] = t;
            idx_arr_[i] = in_idx[idx];
          }
        }
      }
    }
    if (kWarpWidth < WarpSize || do_merge) {
      util::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
    }
  }

  /**
   *  Save the content by the pointer location.
   *
   * @param[out] out
   *   device pointer to a contiguous array, unique per-subwarp of size `kWarpWidth`
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[out] out_idx
   *   device pointer to a contiguous array, unique per-subwarp of size `kWarpWidth`
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param valF (optional) postprocess values (T -> OutT)
   * @param idxF (optional) postprocess indices (IdxT -> OutIdxT)
   */
  template <typename OutT,
            typename OutIdxT,
            typename ValF = identity_op,
            typename IdxF = identity_op>
  _RAFT_DEVICE void store(OutT* out,
                          OutIdxT* out_idx,
                          ValF valF = cuvs::tools::identity_op{},
                          IdxF idxF = cuvs::tools::identity_op{}) const
  {
    int idx = Pow2<kWarpWidth>::mod(laneId());
#pragma unroll kMaxArrLen
    for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
      out[idx]     = valF(val_arr_[i]);
      out_idx[idx] = idxF(idx_arr_[i]);
    }
  }

 protected:
  static constexpr int kMaxArrLen = Capacity / kWarpWidth;

  T val_arr_[kMaxArrLen];
  IdxT idx_arr_[kMaxArrLen];

  /**
   * Merge another array (sorted in the opposite direction) in the queue.
   * Thanks to the other array being sorted in the opposite direction,
   * it's enough to call bitonic.merge once to maintain the valid state
   * of the queue.
   *
   * @tparam PerThreadSizeIn
   *   the size of the other array per-thread (compared to `kMaxArrLen`).
   *
   * @param keys_in
   *   the values to be merged in. Pointers are unique per-thread. The values
   *   must already be sorted in the opposite direction.
   *   The layout of `keys_in` must be the same as the layout of `val_arr_`.
   * @param ids_in
   *   the associated indices of the elements in the same format as `keys_in`.
   */
  template <int PerThreadSizeIn>
  _RAFT_DEVICE _RAFT_FORCEINLINE void merge_in(const T* __restrict__ keys_in,
                                               const IdxT* __restrict__ ids_in)
  {
#pragma unroll
    for (int i = cuda::std::min(kMaxArrLen, PerThreadSizeIn); i > 0; i--) {
      T& key  = val_arr_[kMaxArrLen - i];
      T other = keys_in[PerThreadSizeIn - i];
      if (is_ordered<Ascending>(other, key)) {
        key                      = other;
        idx_arr_[kMaxArrLen - i] = ids_in[PerThreadSizeIn - i];
      }
    }
    util::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
  }
};

/**
 * A fixed-size warp-level priority queue.
 * By feeding the data through this queue, you get the `k <= Capacity`
 * smallest/greatest values in the data.
 *
 * @tparam Capacity
 *   maximum number of elements in the queue.
 * @tparam Ascending
 *   which comparison to use: `true` means `<`, collect the smallest elements,
 *   `false` means `>`, collect the greatest elements.
 * @tparam T
 *   the type of keys (what is being compared)
 * @tparam IdxT
 *   the type of payload (normally, indices of elements), i.e.
 *   the content sorted alongside the keys.
 */

template <template <int, bool, typename, typename> class WarpSortWarpWide,
          int Capacity,
          bool Ascending,
          typename T,
          typename IdxT>
class block_sort {
 public:
  using queue_t = WarpSortWarpWide<Capacity, Ascending, T, IdxT>;

  template <typename... Args>
  __device__ block_sort(int k, Args... args) : queue_(queue_t::init_blockwide(k, args...))
  {
  }

  __device__ void add(T val, IdxT idx) { queue_.add(val, idx); }

  /**
   * At the point of calling this function, the warp-level queues consumed all input
   * independently. The remaining work to be done is to merge them together.
   *
   * Here we tree-merge the results using the shared memory and block sync.
   */
  __device__ void done(uint8_t* smem_buf)
  {
    queue_.done();

    int nwarps    = subwarp_align::div(blockDim.x);
    auto val_smem = reinterpret_cast<T*>(smem_buf);
    auto idx_smem = reinterpret_cast<IdxT*>(
      smem_buf + Pow2<256>::roundUp(ceildiv(nwarps, 2) * sizeof(T) * queue_.k));

    const int warp_id = subwarp_align::div(threadIdx.x);
    // NB: there is no need for the second __synchthreads between .load_sorted and .store:
    //     we shift the pointers every iteration, such that individual warps either access the same
    //     locations or do not overlap with any of the other warps. The access patterns within warps
    //     are different for the two functions, but .load_sorted implies warp sync at the end, so
    //     there is no need for __syncwarp either.
    for (int shift_mask = ~0, split = (nwarps + 1) >> 1; nwarps > 1;
         nwarps = split, split = (nwarps + 1) >> 1) {
      if (warp_id < nwarps && warp_id >= split) {
        int dst_warp_shift = (warp_id - (split & shift_mask)) * queue_.k;
        queue_.store(val_smem + dst_warp_shift, idx_smem + dst_warp_shift);
      }
      __syncthreads();

      shift_mask = ~shift_mask;  // invert the mask
      {
        int src_warp_shift = (warp_id + (split & shift_mask)) * queue_.k;
        // The last argument serves as a condition for loading
        //  -- to make sure threads within a full warp do not diverge on `bitonic::merge()`
        queue_.load_sorted(
          val_smem + src_warp_shift, idx_smem + src_warp_shift, warp_id < nwarps - split);
      }
    }
  }

  /** Save the content by the pointer location. */
  template <typename OutT,
            typename OutIdxT,
            typename ValF = identity_op,
            typename IdxF = identity_op>
  __device__ void store(OutT* out,
                        OutIdxT* out_idx,
                        ValF valF = cuvs::tools::identity_op{},
                        IdxF idxF = cuvs::tools::identity_op{}) const
  {
    if (threadIdx.x < subwarp_align::Value) { queue_.store(out, out_idx, valF, idxF); }
  }

 private:
  using subwarp_align = Pow2<queue_t::kWarpWidth>;
  queue_t queue_;
};

/**
 * This version of warp_sort compares each input element against the current
 * estimate of k-th value before adding it to the intermediate sorting buffer.
 * This makes the algorithm do less sorting steps for long input sequences
 * at the cost of extra checks on each step.
 *
 * This implementation is preferred for large len values.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_filtered : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;
  using warp_sort<Capacity, Ascending, T, IdxT>::mem_required;

  explicit __device__ warp_sort_filtered(int k, T limit = kDummy)
    : warp_sort<Capacity, Ascending, T, IdxT>(k), buf_len_(0), k_th_(limit)
  {
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
      idx_buf_[i] = IdxT{};
    }
  }

  __device__ __forceinline__ static auto init_blockwide(int k, uint8_t* = nullptr, T limit = kDummy)
  {
    return warp_sort_filtered<Capacity, Ascending, T, IdxT>{k, limit};
  }

  __device__ void add(T val, IdxT idx)
  {
    // comparing for k_th should reduce the total amount of updates:
    // `false` means the input value is surely not in the top-k values.
    bool do_add = is_ordered<Ascending>(val, k_th_);
    // merge the buf if it's full and we cannot add an element anymore.
    if (any(buf_len_ + do_add > kMaxBufLen)) {
      // still, add an element before merging if possible for this thread
      if (do_add && buf_len_ < kMaxBufLen) {
        add_to_buf_(val, idx);
        do_add = false;
      }
      merge_buf_();
    }
    // add an element if necessary and haven't already.
    if (do_add) { add_to_buf_(val, idx); }
  }

  __device__ void done()
  {
    if (any(buf_len_ != 0)) { merge_buf_(); }
  }

 private:
  __device__ __forceinline__ void set_k_th_()
  {
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);
  }

  __device__ __forceinline__ void merge_buf_()
  {
    util::bitonic<kMaxBufLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
    this->merge_in<kMaxBufLen>(val_buf_, idx_buf_);
    buf_len_ = 0;
    set_k_th_();  // contains warp sync
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
    }
  }

  __device__ __forceinline__ void add_to_buf_(T val, IdxT idx)
  {
    // NB: the loop is used here to ensure the constant indexing,
    //     to not force the buffers spill into the local memory.
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }
    buf_len_++;
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  static constexpr int kMaxBufLen = (Capacity <= 64) ? 2 : 4;

  T val_buf_[kMaxBufLen];
  IdxT idx_buf_[kMaxBufLen];
  int buf_len_;

  T k_th_;
};

/**
 * This version of warp_sort compares each input element against the current
 * estimate of k-th value before adding it to the intermediate sorting buffer.
 * In contrast to `warp_sort_filtered`, it keeps one distributed buffer for
 * all threads in a warp (independently of the subwarp size), which makes its flushing less often.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_distributed : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;
  using warp_sort<Capacity, Ascending, T, IdxT>::mem_required;

  explicit _RAFT_DEVICE warp_sort_distributed(int k, T limit = kDummy)
    : warp_sort<Capacity, Ascending, T, IdxT>(k),
      buf_val_(kDummy),
      buf_idx_(IdxT{}),
      buf_len_(0),
      k_th_(limit)
  {
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE static auto init_blockwide(int k,
                                                            uint8_t* = nullptr,
                                                            T limit  = kDummy)
  {
    return warp_sort_distributed<Capacity, Ascending, T, IdxT>{k, limit};
  }

  _RAFT_DEVICE void add(T val, IdxT idx)
  {
    // mask tells which lanes in the warp have valid items to be added
    uint32_t mask = ballot(is_ordered<Ascending>(val, k_th_));
    if (mask == 0) { return; }
    // how many elements to be added
    uint32_t n_valid = __popc(mask);
    // index of the source lane containing the value to put into the current lane.
    uint32_t src_ix = 0;
    // remove a few smallest set bits from the mask.
    for (uint32_t i = std::min(n_valid, Pow2<WarpSize>::mod(uint32_t(laneId()) - buf_len_)); i > 0;
         i--) {
      src_ix = __ffs(mask) - 1;
      mask ^= (0x1u << src_ix);
    }
    // now the least significant bit of the mask corresponds to the lane id we want to get.
    // for not-added (invalid) indices, the mask is zeroed by now.
    src_ix = __ffs(mask) - 1;
    // rearrange the inputs to be ready to put them into the tmp buffer
    val = shfl(val, src_ix);
    idx = shfl(idx, src_ix);
    // for non-valid lanes, src_ix should be uint(-1)
    if (mask == 0) { val = kDummy; }
    // save the values into the free slots of the warp tmp buffer
    if (laneId() >= buf_len_) {
      buf_val_ = val;
      buf_idx_ = idx;
    }
    buf_len_ += n_valid;
    if (buf_len_ < WarpSize) { return; }
    // merge the warp tmp buffer into the queue
    merge_buf_();
    buf_len_ -= WarpSize;
    // save the inputs that couldn't fit before the merge
    if (laneId() < buf_len_) {
      buf_val_ = val;
      buf_idx_ = idx;
    }
  }

  _RAFT_DEVICE void done()
  {
    if (buf_len_ != 0) {
      merge_buf_();
      buf_len_ = 0;
    }
  }

 private:
  _RAFT_DEVICE _RAFT_FORCEINLINE void set_k_th_()
  {
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE void merge_buf_()
  {
    util::bitonic<1>(!Ascending, kWarpWidth).sort(buf_val_, buf_idx_);
    this->merge_in<1>(&buf_val_, &buf_idx_);
    set_k_th_();  // contains warp sync
    buf_val_ = kDummy;
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  T buf_val_;
  IdxT buf_idx_;
  uint32_t buf_len_;  // 0 <= buf_len_ <= WarpSize

  T k_th_;
};

}  // namespace cuvs::tools::matrix::detail::select::warpsort