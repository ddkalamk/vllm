
#include "ops.h"
#include "cache.h"
#include "cuda_utils.h"
#include <torch/extension.h>

void paged_attention_v1(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void paged_attention_v2(
  torch::Tensor& out,
  torch::Tensor& exp_sums,
  torch::Tensor& max_logits,
  torch::Tensor& tmp_out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  int num_kv_heads,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void fused_add_rms_norm(
  torch::Tensor& input,
  torch::Tensor& residual,
  torch::Tensor& weight,
  float epsilon)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void rotary_embedding(
  torch::Tensor& positions,
  torch::Tensor& query,
  torch::Tensor& key,
  int head_size,
  torch::Tensor& cos_sin_cache,
  bool is_neox)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void silu_and_mul(
  torch::Tensor& out,
  torch::Tensor& input)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void gelu_new(
  torch::Tensor& out,
  torch::Tensor& input)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void gelu_fast(
  torch::Tensor& out,
  torch::Tensor& input)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

#ifndef USE_ROCM
torch::Tensor awq_gemm(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  int split_k_iters)
{
  printf("%s:%d  called\n", __func__, __LINE__);
  return torch::Tensor();
}
#endif

void squeezellm_gemm(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

torch::Tensor gptq_gemm(
  torch::Tensor a,
  torch::Tensor b_q_weight,
  torch::Tensor b_gptq_qzeros,
  torch::Tensor b_gptq_scales,
  torch::Tensor b_g_idx,
  bool use_exllama)
{
  printf("%s:%d  called\n", __func__, __LINE__);
  return torch::Tensor();
}

void gptq_shuffle(
  torch::Tensor q_weight,
  torch::Tensor q_perm)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}


void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping)
{
  printf("%s:%d  called\n", __func__, __LINE__);
}

int get_device_attribute(
    int attribute,
    int device_id)
{
  printf("%s:%d  called\n", __func__, __LINE__);
  return 0;
}
