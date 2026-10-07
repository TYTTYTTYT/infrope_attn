/******************************************************************************
 * infrope fork: sink-scale.  Per (query row, head) threshold b and shift delta on the raw QK scores (q.k before the
 * softmax scale): scores below b are shifted by -delta through a linear ramp of width tau around b (hard step when
 * tau_inv <= 0).  Rows are addressed as in Mask; with PackGQA a tile row packs (query position, q head in the kv group).
 *
 * Fast path: if the whole row of a tile sits below the ramp (tile max < b - tau/2), every element would get exactly
 * -delta, so nothing is touched and the shift is returned as a per-row offset that the online softmax folds into its
 * row max (softmax.h, *_off variants).  Only tiles holding a score above that bound pay the elementwise gate; with a
 * sink threshold these are a few per row.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, typename TiledMma, bool PackGQA>
struct SinkScale {

    int const thread_idx;
    int const seqlen_q;
    float const* const pb;          // threshold b, indexed [h * head_stride + row] from the block's first head
    float const* const pd;          // shift delta, same indexing
    int64_t const head_stride;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    float const tau_inv;

    CUTLASS_DEVICE
    SinkScale(const int thread_idx, const int seqlen_q, float const* pb, float const* pd, int64_t head_stride,
              cutlass::FastDivmod const &qhead_per_khead_divmod, float tau_inv)
        : thread_idx(thread_idx), seqlen_q(seqlen_q), pb(pb), pd(pd), head_stride(head_stride),
          qhead_per_khead_divmod(qhead_per_khead_divmod), tau_inv(tau_inv) {};

    template <typename Engine, typename Layout, typename TensorOff>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, TensorOff &off) const {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));
        CUTE_STATIC_ASSERT_V(size<0>(tSrS_rowcol) == size(off));
        // row max of this tile (the 4 threads of a row together), on the raw scores
        Tensor tmax = make_fragment_like(off);
        MaxOp<float> max_op;
        flash::reduce_<true>(tSrS_rowcol, tmax, max_op);
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const prow = int(get<0>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;
            float b = 0.f, d = 0.f;
            if constexpr (!PackGQA) {
                if (prow < seqlen_q) { b = pb[prow]; d = pd[prow]; }
            } else {
                int row, h;
                qhead_per_khead_divmod(row, h, prow);
                if (row < seqlen_q) { b = pb[h * head_stride + row]; d = pd[h * head_stride + row]; }
            }
            float o = 0.f;
            if (d != 0.f) {
                float const lo = tau_inv > 0.f ? b - 0.5f / tau_inv : b;       // below lo the ramp is fully on
                if (tmax(m) < lo) {
                    o = d;                                                     // whole row-tile shifted: fold into the max
                } else {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        float s = tSrS_rowcol(m, n);
                        if (tau_inv > 0.f) {
                            float g = fmaf(b - s, tau_inv, 0.5f);
                            g = fminf(fmaxf(g, 0.f), 1.f);
                            s = fmaf(-d, g, s);
                        } else if (s < b) {
                            s -= d;
                        }
                        tSrS_rowcol(m, n) = s;
                    }
                }
            }
            off(m) = o;
        }
    }
};

}  // namespace flash
