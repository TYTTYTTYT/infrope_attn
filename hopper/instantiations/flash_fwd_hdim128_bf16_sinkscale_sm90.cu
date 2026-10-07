// fs fork: head dim 128 forward kernel with the sink-scale score modifier compiled in.
#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false, true>(Flash_fwd_params &params, cudaStream_t stream);
