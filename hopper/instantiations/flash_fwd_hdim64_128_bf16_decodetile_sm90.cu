// infrope fork, decode tiles: the kernels with a 64-row query tile (tile_size.h, get_decode_tile in flash_api_stable.cpp).
// Template arguments: Arch, T, kHeadDim, kHeadDimV, Split, PagedKVNonTMA, Has_softcap, PackGQA, Has_sinkscale, Decode_tile.

#include "flash_fwd_launch_template.h"

template void run_mha_fwd_<90, cutlass::bfloat16_t, 64, 128, false, false, false, false, false, true>(Flash_fwd_params &params, cudaStream_t stream);
