# Synthetic inference throughput; run after installing the model:
# CPU: JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 julia --project benchmark/inference.jl [prompt_tokens] [generated_tokens] [repetitions]
# Metal: PLAMO_BENCH_DEVICE=metal JULIA_NUM_THREADS=1 julia --project=<metal-environment> benchmark/inference.jl
using DocstringTranslationPLaMoBackend
using Statistics

if get(ENV, "PLAMO_BENCH_DEVICE", "cpu") == "metal"
    using Metal
end

const P = DocstringTranslationPLaMoBackend.PLaMo

function run_benchmark(prompt_tokens, generated_tokens, repetitions)
    prompt_tokens > 0 && generated_tokens > 0 && repetitions > 0 ||
        throw(ArgumentError("token counts and repetitions must be positive"))
    directory = DocstringTranslationPLaMoBackend.model_directory()
    DocstringTranslationPLaMoBackend.verify_model(directory)
    storage = Symbol(get(ENV, "PLAMO_BENCH_STORAGE", "bf16"))
    storage in (:bf16, :f32) ||
        throw(ArgumentError("PLAMO_BENCH_STORAGE must be bf16 or f32"))
    device = get(ENV, "PLAMO_BENCH_DEVICE", "cpu")
    device in ("cpu", "metal") ||
        throw(ArgumentError("PLAMO_BENCH_DEVICE must be cpu or metal"))
    if device == "metal"
        metal = Base.get_extension(
            DocstringTranslationPLaMoBackend,
            :DocstringTranslationPLaMoBackendMetalExt,
        )
        model = metal.MetalModel(
            directory;
            context_size = prompt_tokens + generated_tokens,
            weight_storage = storage,
        )
        make_state = () -> metal.MetalState(model, prompt_tokens + generated_tokens)
        forward = (ids, state) -> metal.forward_gpu!(model, ids, state)
    else
        model = P.Translator(
            directory;
            context_size = prompt_tokens + generated_tokens,
            weight_storage = storage,
        ).model
        make_state = () -> P.request_state(model, prompt_tokens, generated_tokens)
        forward = (ids, state) -> P.forward!(model, ids, state)
    end
    # A fixed valid token makes the arithmetic workload reproducible without
    # EOS or tokenizer-dependent early stopping.
    tokens = fill(0, prompt_tokens)
    function sample()
        state = make_state()
        prefill_start = time_ns()
        for first = 1:128:prompt_tokens
            forward(@view(tokens[first:min(first+127, end)]), state)
        end
        prefill_seconds = (time_ns() - prefill_start) / 1e9
        decode_start = time_ns()
        for _ = 1:generated_tokens
            forward([0], state)
        end
        decode_seconds = (time_ns() - decode_start) / 1e9
        return (prefill_seconds, decode_seconds)
    end
    sample() # compile and warm the exact path before measurement
    samples = [sample() for _ = 1:repetitions]
    for (i, (prefill, decode)) in enumerate(samples)
        println(
            "sample=$i prefill_tps=$(prompt_tokens / prefill) decode_tps=$(generated_tokens / decode)",
        )
    end
    println("median_prefill_tps=", median(prompt_tokens ./ first.(samples)))
    println("median_decode_tps=", median(generated_tokens ./ last.(samples)))
end

args = parse.(Int, ARGS)
length(args) <= 3 ||
    error("usage: inference.jl [prompt_tokens] [generated_tokens] [repetitions]")
run_benchmark((args..., (128, 64, 3)[(length(args)+1):end]...)...)
