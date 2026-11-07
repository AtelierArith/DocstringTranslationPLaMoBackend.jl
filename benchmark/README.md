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
retains the official BF16 weight payload. The benchmark synchronizes when
reading logits back after every forward call.

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
