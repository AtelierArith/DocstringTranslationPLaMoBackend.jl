# PLaMo inference throughput

`inference.jl` measures model execution without loading, tokenization, sampling,
or model download. It warms each path once, then times a 128-token prefill and
64 single-token decode calls by default. Decode starts with the prefill in the
KV cache. Set the three positional arguments to change the prompt length, the
decode length, and repetitions. A fixed valid token prevents early stopping.

```sh
# CPU
JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  julia --project benchmark/inference.jl 128 64 3

# Metal (after instantiating test/metal)
PLAMO_BENCH_DEVICE=metal JULIA_NUM_THREADS=1 \
  julia --project=test/metal benchmark/inference.jl 128 64 3
```

`PLAMO_BENCH_STORAGE=f32` expands the weights to Float32; the default `bf16`
retains the official BF16 weight payload. Metal's `PLAMO_BENCH_PREFILL=auto`
uses the same memory-dependent Float32 prompt cache as the public API; set it
to `bf16` or `f32` to force a specific cache. The default `logits` output
mode synchronizes and reads the full logits after every forward call; set
`PLAMO_BENCH_OUTPUT=token` to measure the production greedy token selection.

Inspect warm GPU kernel activity with:

```sh
JULIA_NUM_THREADS=1 julia --project=test/metal benchmark/profile_metal.jl decode
JULIA_NUM_THREADS=1 julia --project=test/metal benchmark/profile_metal.jl prefill
```

The integrated Metal profiler changes command batching and omits MPS matrix
multiplications from its device activity table. Use these profiles to locate
costly kernels and `inference.jl` to compare actual throughput.

Check allocations in a separate diagnostic environment:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.instantiate(; allow_build=false)'
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=benchmark benchmark/allocations_metal.jl
```

This runs AllocCheck static analysis, warm `@allocated` measurements and a
`Profile.Allocs` trace for decode. It compares the production token result with
full CPU logits and prints GPU buffer-creation API calls separately. Model
loading and request-state setup are excluded. Static findings include possible
cold paths and unresolved dispatches; their count is not a runtime allocation
count. Julia allocation measurements exclude Metal buffer storage. Set
`PLAMO_ALLOC_FULL=1` to print every static backtrace instead of grouped summaries.

For example, on M2 Max with Julia 1.13.3, Metal 1.12.0 and AllocCheck 0.2.8,
BF16 weights and the automatic Float32 prefill cache produced these warm host
allocation measurements (128-token prompt, request capacity 512):

| Operation | Julia host bytes/call |
| --- | ---: |
| Decode, token result, vector input | 98,096 |
| Decode, token result, tuple input (production) | 98,032 |
| Decode, full CPU logits | 534,384 |
| Prefill, 128 tokens, token result | 397,360 |

Making the split-attention launch types concrete reduced token decode from
103,408 to 98,032 host bytes/call. The allocation trace attributes most remaining
bytes to GPU kernel-launch call sites. Reusing prefill intermediates and score
buffers reduced warm prefill host allocation from 515,696 to 397,360 bytes/call.
The GPU profile observed one `newBufferWithLength:options:` call for warm decode
and four for warm prefill; the earlier allocating prefill path observed 246.
The first prefill also allocates its request-owned workspace, which is retained
for chunks of the same length and resized for a different length.
AllocCheck also reports potential allocation sites and unresolved dispatches in
Metal's launch/runtime paths; this is not an allocation-free inference path.

For comparisons, convert the same pinned checkpoint to an unquantized BF16
GGUF and build llama.cpp with Metal. For example, with llama.cpp commit
`fd27b2034905f8dadd8bd6d2cc44595f4929d391`:

```sh
python convert_hf_to_gguf.py "$MODEL_DIR" --outtype bf16 --outfile model-bf16.gguf
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build build --target llama-bench -j 8
build/bin/llama-bench -m model-bf16.gguf -p 128 -n 0 -r 3 -ub 128 -fa off
build/bin/llama-bench -m model-bf16.gguf -p 0 -n 64 -d 128 -r 3 -ub 128 -fa off
```

The checked llama-bench version accepts only reduced-precision KV caches through
its CLI. To reproduce a Float32 KV comparison with this backend, add `f32` to
`ggml_type_from_name` in `tools/llama-bench/llama-bench.cpp` and pass
`-ctk f32 -ctv f32`. These synthetic tests compare shapes and cache precision;
llama-bench and this script do not feed identical token IDs, and their prefill
implementations may perform different last-token work. Also compare actual
translation quality and workload-specific latency before drawing conclusions.

For a real short translation, export the exact PFML prompt and compare greedy
output and latency. Both commands use the same 21-token input, BF16 weights,
and Float32 KV cache; llama.cpp must not add a second BOS to the prompt:

```sh
PLAMO_BENCH_DEVICE=metal PLAMO_BENCH_PROMPT_FILE=/tmp/plamo-prompt.txt \
  JULIA_NUM_THREADS=1 julia --project=test/metal benchmark/translation.jl

build/bin/llama-completion -m model-bf16.gguf -f /tmp/plamo-prompt.txt \
  -n 32 -c 512 --temp 0 --seed 1 --no-display-prompt --no-conversation \
  --cache-type-k f32 --cache-type-v f32 \
  --override-kv tokenizer.ggml.add_bos_token=bool:false
```

The Julia timer excludes tokenization and model loading. `llama-completion`
reports prompt evaluation, decode, sampling, and total durations separately.
The two timing scopes are close but not identical, so repeat and report both
rather than treating a single run as a definitive performance ratio.
