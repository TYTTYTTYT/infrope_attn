// infrope fork.
#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 128, true, false, false, true, true>(Flash_fwd_params &params, cudaStream_t stream);
