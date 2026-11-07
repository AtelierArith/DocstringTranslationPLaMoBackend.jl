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
| Decode, token result, vector input | 93,296 |
| Decode, token result, tuple input (production) | 93,232 |
| Decode, full CPU logits | 534,384 |
| Prefill, 128 tokens, token result | 392,768 |

Making the split-attention launch types concrete reduced token decode from
103,408 to 98,032 host bytes/call. The allocation trace attributes most remaining
bytes to GPU kernel-launch call sites. Reusing prefill intermediates and score
buffers reduced warm prefill host allocation from 515,696 to 397,264 bytes/call.
A request-owned token-selection buffer further reduced production decode to
93,232 bytes and prefill to 392,768 bytes. One GPU kernel scans the vocabulary
into partial maxima; the host selects the final winner from these compact
partials (105 for the pinned vocabulary), preserving first-index, NaN and signed
zero ordering. Paired full-decode benchmarks found no material speed change.
The GPU profile now observed no buffer-creation API calls for warm decode and
three for warm prefill, down from one and four before token-buffer reuse. The
earlier allocating prefill path observed 246.
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
cmake --build build --target llama-bench llama-completion -j 8
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

## Measured comparison

On 2026-11-07, Apple M2 Max (38 GPU cores, 96 GB), macOS 26.7.1,
Julia 1.13.3 / Metal 1.12.0, and llama.cpp `fd27b203` produced the following
warm results. Both used the pinned unquantized BF16 checkpoint and Float32 KV
cache. Julia used one thread and production token output; llama.cpp used eight
CPU threads with Metal. Synthetic llama-bench runs explicitly disabled flash
attention; short translations kept llama-completion's default automatic setting.
Metal used its automatic Float32 prefill cache. Loading and compilation were excluded.

| Workload | Metal.jl | llama.cpp |
| --- | ---: | ---: |
| Prefill, 128 tokens (median, 7 runs) | 4,924 token/s | 4,974 token/s |
| Decode, 64 tokens after 128 cached tokens (median, 7 runs) | 161 token/s | 158 token/s |
| Short translation, 21 input / 7 generated tokens (median, 9 runs) | 54.7 ms | 64.2 ms |

Both translations produced `会議は10時に始まります。`. BF16 matrix-vector
products use packed UInt32 loads with exact BF16-to-Float32 reconstruction;
Apple GPU family 7 and newer use native SIMD reductions, with shuffle-based
reductions on older targets. Accumulation stays Float32.

These workloads reached comparable performance in this measurement; this does
not establish a speed advantage for every prompt length or GPU. The machine
also had other CPU work, and llama.cpp decode samples ranged from 102 to 173
token/s. Its arithmetic-mean throughput was 4,809 prefill / 152 decode token/s;
the table uses sample medians consistently for both implementations. Repeat on
your target machine using the commands above. The short-translation timers have
slightly different scopes, as described above.

## BF16, Q8_0 and Q4_0

Compare all supported formats through the actual Metal model loader:

```sh
PLAMO_BENCH_JSON=/tmp/plamo-storage.json JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=benchmark benchmark/compare_storage.jl
# Or benchmark one format with the existing tools:
PLAMO_BENCH_DEVICE=metal PLAMO_BENCH_STORAGE=q8_0 PLAMO_BENCH_OUTPUT=token \
  JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/inference.jl 128 64 7
PLAMO_BENCH_DEVICE=metal PLAMO_BENCH_STORAGE=q8_0 \
  JULIA_NUM_THREADS=1 julia --project=benchmark benchmark/translation.jl
PLAMO_ALLOC_STORAGE=q8_0 JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=benchmark benchmark/allocations_metal.jl
```

The comparison loads each model sequentially, warms inference once, and reports
seven samples by default (`PLAMO_BENCH_REPETITIONS`). Request-state construction
is outside the synthetic timer; its lazy prefill workspace is inside. Real
translation includes request/workspace setup. Eight translation examples are
printed for comparing output, not as a complete quality evaluation.

The opt-in quantizers use 32-value blocks and Float16 scales with Q8_0/Q4_0
reference rounding. The internal GPU layout uses separate scales and adjacent
Q4 nibbles; it is not a GGUF loader or a byte-compatible GGML block layout.
BF16 is the default. Quantization runs on the CPU at translator creation, without
Julia thread parallelism, and is not persisted. Activations, KV caches and dot
product accumulation stay Float32. Prefill's expanded cache contains the chosen
format's dequantized weights, not the original BF16 values.

On the same M2 Max environment as above, the production implementation measured:

| Median (7 samples) | BF16 | Q8_0 | Q4_0 |
| --- | ---: | ---: | ---: |
| Metal prefill, 128 tokens | 5,068 token/s | 5,211 token/s | 5,070 token/s |
| Metal decode, 64 after 128 cached tokens | 165 token/s | 218 token/s | 166 token/s |
| Metal short translation, 21 input / 7 output tokens | 56.0 ms | 45.3 ms | 56.3 ms |
| llama.cpp prefill | 5,389 token/s | 5,686 token/s | 5,102 token/s |
| llama.cpp decode | 181 token/s | 192 token/s | 283 token/s |
| llama.cpp short translation | 64.4 ms | 61.6 ms | 48.6 ms |

Q8_0 improved Metal decode by about 32% and short translation by about 19%
relative to BF16 in this run. Q4_0 currently reduces storage without improving
Metal decode; its decode performance remains below llama.cpp's Q4_0 kernel.
The earlier timer/workload caveats apply, and other CPU work was present.
llama.cpp used the same pinned checkpoint, `--pure` Q8_0/Q4_0 quantization,
Float32 KV and the earlier command settings. Payload and scale comparisons
matched all 66 matrices after accounting for the different internal layout.

All three produced `会議は10時に始まります。` for the short prompt. Q8_0's output
matched BF16 on all eight examples. Q4_0 matched six; the other two changed
wording, including `検証ステップをスキップするには...` to
`` `check=false` を設定してバリデーションステップをスキップします。 ``.
These string matches do not establish quality preservation for other inputs.

| Model memory and setup | BF16 | Q8_0 | Q4_0 |
| --- | ---: | ---: | ---: |
| GPU array payloads, including Float32 prefill cache | 3.745 GiB | 2.968 GiB | 2.553 GiB |
| Separate-process peak RSS | 5.290 GiB | 3.864 GiB | 3.648 GiB |
| Model setup in those separate processes | 2.78 s | 4.96 s | 5.09 s |

GPU payloads sum the model's arrays; they exclude request KV/workspace,
allocator padding and transient allocations. Peak RSS was measured with macOS
`/usr/bin/time -l` in separate runs of `compare_storage.jl <format>` with one
repetition. It includes compilation/mapped memory and is not a Metal-device
allocation measurement. Setup excludes package import but includes checkpoint
loading, quantization, cache creation and synchronization, with the checkpoint
already in the filesystem cache. First-inference compilation is separate.

Warm allocation diagnostics retained zero GPU buffer-creation API calls per
decode and three per 128-token prefill for both quantized formats. Julia host
allocation was 96,400 bytes/token for Q8_0 and 96,368 for Q4_0, compared with
93,232 for BF16. This remains an allocating host path.

For corresponding llama.cpp assets:

```sh
cmake --build build --target llama-quantize llama-bench llama-completion -j 8
build/bin/llama-quantize --pure model-bf16.gguf model-q8.gguf Q8_0 4
build/bin/llama-quantize --pure model-bf16.gguf model-q4.gguf Q4_0 4
# Repeat the earlier inference and translation commands for each model file.
```

The [recorded samples and outputs](results/metal-storage-m2max-2026-11-07.json)
include the model revision, conditions, llama.cpp comparison, array payloads,
separate-process RSS, setup times and allocation measurements.
