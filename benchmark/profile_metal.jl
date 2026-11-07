# Warm GPU profiling. The integrated profiler changes command batching and
# omits MPS matrix multiplication: use inference.jl for throughput comparisons.
using DocstringTranslationPLaMoBackend, Metal

function profile_metal(phase)
    phase in ("prefill", "decode") || error("usage: profile_metal.jl [prefill|decode]")
    backend = DocstringTranslationPLaMoBackend
    extension = backend.MetalBackend
    directory = backend.model_directory()
    backend.verify_model(directory)
    hybrid = extension.use_hybrid_prefill(
        :bf16,
        :auto,
        Metal.free_memory(),
        Sys.free_memory(),
        extension.kv_cache_bytes(directory, 256),
    )
    println("phase=$phase hybrid_prefill=$hybrid")
    model =
        extension.MetalModel(directory; context_size = 256, weight_storage = :bf16, hybrid)
    state = extension.MetalState(model, 256)
    tokens = fill(0, 128)
    extension.forward_gpu!(model, tokens, state; token_only = true)
    for _ = 1:8
        extension.forward_gpu!(model, [0], state; token_only = true)
    end
    if phase == "prefill"
        state.position = 0
        result =
            Metal.@profile extension.forward_gpu!(model, tokens, state; token_only = true)
    else
        result = Metal.@profile for _ = 1:8
            extension.forward_gpu!(model, [0], state; token_only = true)
        end
    end
    show(stdout, result)
    println()
end

length(ARGS) <= 1 || error("usage: profile_metal.jl [prefill|decode]")
profile_metal(isempty(ARGS) ? "decode" : only(ARGS))
