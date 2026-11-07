#include "llama.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
struct Q4Bench {
  llama_model *model;
  llama_context *context;
  std::vector<llama_token> prompt;
};
static void quiet(enum ggml_log_level level, const char *text, void *) {
  if (level == GGML_LOG_LEVEL_ERROR)
    fputs(text, stderr);
}
extern "C" void *q4_bench_create(const char *path, int f32kv, int flash) {
  llama_log_set(quiet, nullptr);
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 999;
  auto model = llama_model_load_from_file(path, mp);
  if (!model)
    return nullptr;
  auto cp = llama_context_default_params();
  cp.n_ctx = 512;
  cp.n_batch = 128;
  cp.n_ubatch = 128;
  cp.n_threads = 8;
  cp.n_threads_batch = 8;
  cp.no_perf = true;
  cp.type_k = f32kv ? GGML_TYPE_F32 : GGML_TYPE_F16;
  cp.type_v = cp.type_k;
  cp.flash_attn_type = static_cast<llama_flash_attn_type>(flash);
  auto ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    llama_model_free(model);
    return nullptr;
  }
  return new Q4Bench{model, ctx, std::vector<llama_token>(128, 0)};
}
extern "C" int q4_bench_prefill(void *ptr) {
  auto b = static_cast<Q4Bench *>(ptr);
  llama_memory_clear(llama_get_memory(b->context), false);
  auto batch = llama_batch_get_one(b->prompt.data(), 128);
  int rc = llama_decode(b->context, batch);
  llama_synchronize(b->context);
  return rc;
}
extern "C" double q4_bench_decode(void *ptr, int count, int sample) {
  auto b = static_cast<Q4Bench *>(ptr);
  llama_token token = 0;
  auto started = std::chrono::steady_clock::now();
  for (int i = 0; i < count; i++) {
    auto batch = llama_batch_get_one(&token, 1);
    if (llama_decode(b->context, batch) != 0)
      return -1;
    llama_synchronize(b->context);
    if (sample) {
      auto logits = llama_get_logits_ith(b->context, -1);
      int n = llama_vocab_n_tokens(llama_model_get_vocab(b->model));
      float best = logits[0];
      int index = 0;
      for (int j = 1; j < n; j++) {
        if (logits[j] > best) {
          best = logits[j];
          index = j;
        }
      }
      token = index;
    }
  }
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       started)
      .count();
}
extern "C" void q4_bench_free(void *ptr) {
  auto b = static_cast<Q4Bench *>(ptr);
  llama_free(b->context);
  llama_model_free(b->model);
  delete b;
}
