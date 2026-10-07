/******************************************************************************
 * fs fork: sink-scale.  Per (query row, head) threshold b and shift delta on the raw QK scores: scores below b are
 * shifted by -delta (linear ramp of width tau around b; hard step when tau_inv <= 0).  Rows are addressed exactly as in Mask
 * (no PackGQA, no SwapAB).
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, typename TiledMma>
struct SinkScale {

    int const thread_idx;
    int const seqlen_q;
    float const* const pb;
    float const* const pd;
    float const tau_inv;

    CUTLASS_DEVICE
    SinkScale(const int thread_idx, const int seqlen_q, float const* pb, float const* pd, float tau_inv)
        : thread_idx(thread_idx), seqlen_q(seqlen_q), pb(pb), pd(pd), tau_inv(tau_inv) {};

    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block) const {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/false>(tScS.layout()));
        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int const row = int(get<0>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;
            float b = 0.f, d = 0.f;
            if (row < seqlen_q) { b = pb[row]; d = pd[row]; }
            if (d != 0.f) {
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    float s = tSrS_rowcol(m, n);
                    if (tau_inv > 0.f) {       // linear ramp of width tau centred on b (cheap stand-in for the sigmoid)
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
    }
};

}  // namespace flash
