// fs fork: narrow QK (64) x full V (128) forward kernel.
#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 128, false, false, false, false>(Flash_fwd_params &params, cudaStream_t stream);
