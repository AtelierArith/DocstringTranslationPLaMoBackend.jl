# Compare the supported Metal storage paths in separate, sequential model loads.
using DocstringTranslationPLaMoBackend, Metal, Statistics

const B = DocstringTranslationPLaMoBackend
const P = B.PLaMo
const E = B.MetalBackend
const SOURCES = [
    "The meeting starts at ten.",
    "Return the sum of the elements in the array.",
    "Compute the eigenvalues and eigenvectors of a real symmetric matrix.",
    "Set `check=false` to skip the validation step.",
    "The function returns `nothing` when the input is empty.",
    "A scalar is broadcast across every element of the array.",
    "Throw an ArgumentError if the dimensions do not match.",
    "Allocate an uninitialized array with the specified element type and dimensions.",
]

payload(w::MtlArray) = length(w) * sizeof(eltype(w))
payload(w::E.QuantizedMatrix) = payload(w.data) + payload(w.scales)
function model_payload(model)
    weights = payload(model.embedding) + payload(model.head)
    prefill = 0
    other =
        payload(model.norm) +
        sum(payload, (model.local_cos, model.local_sin, model.full_cos, model.full_sin))
    for layer in model.layers
        weights += sum(payload, (layer.qkv, layer.out, layer.gate_up, layer.down))
        prefill += isnothing(layer.prefill) ? 0 : sum(payload, layer.prefill)
        other += sum(
            payload,
            (
                layer.qnorm,
                layer.knorm,
                layer.pre_attn,
                layer.post_attn,
                layer.pre_mlp,
                layer.post_mlp,
            ),
        )
    end
    return (;
        weight_bytes = weights,
        prefill_cache_bytes = prefill,
        other_model_bytes = other,
        model_array_bytes = weights+prefill+other,
    )
end

function sample(model, tokens)
    state = E.MetalState(model, 192) # setup is excluded; lazy prefill workspace is timed
    started = time_ns()
    E.forward_gpu!(model, tokens, state; token_only = true)
    prefilled = time_ns()
    for _ = 1:64
        E.forward_gpu!(model, (0,), state; token_only = true)
    end
    return (128e9/(prefilled-started), 64e9/(time_ns()-prefilled))
end

function run_storage(storage, repetitions)
    directory = B.model_directory()
    hybrid = E.use_hybrid_prefill(
        storage,
        :auto,
        Metal.free_memory(),
        Sys.free_memory(),
        E.kv_cache_bytes(directory, 512),
    )
    load_started = time_ns()
    model = E.MetalModel(directory; context_size = 512, weight_storage = storage, hybrid)
    Metal.synchronize()
    load_seconds = (time_ns() - load_started) / 1e9
    memory = model_payload(model)
    tokens = fill(0, 128)
    sample(model, tokens)
    samples = [sample(model, tokens) for _ = 1:repetitions]
    translator = E.MetalTranslator(model, P.Tokenizer(directory))
    request = B.translation_request(first(SOURCES), :ja, 96)
    P.translate(translator, request)
    translations = [P.translate(translator, request) for _ = 1:repetitions]
    quality = [
        (;
            source,
            text = result.text,
            finish = String(result.finish_reason),
            generated_tokens = result.generated_tokens,
        ) for source in SOURCES for
        result in (P.translate(translator, B.translation_request(source, :ja, 96)),)
    ]
    return (;
        storage = String(storage),
        hybrid_prefill = hybrid,
        load_seconds,
        memory...,
        prefill_tps = median(first.(samples)),
        decode_tps = median(last.(samples)),
        short_seconds = median(getproperty.(translations, :elapsed)),
        short_text = first(translations).text,
        prefill_samples = first.(samples),
        decode_samples = last.(samples),
        short_samples = getproperty.(translations, :elapsed),
        quality,
    )
end

function main()
    Metal.functional() || error("this comparison requires a working Metal GPU")
    storages = isempty(ARGS) ? [:bf16, :q8_0, :q4_0] : Symbol.(ARGS)
    all(in((:bf16, :q8_0, :q4_0, :f32)), storages) ||
        error("expected bf16, q8_0, q4_0 or f32")
    repetitions = parse(Int, get(ENV, "PLAMO_BENCH_REPETITIONS", "7"))
    repetitions > 0 || error("repetitions must be positive")
    B.configure_plamo!(; device = :metal, context_size = 512, max_tokens = 96)
    results = []
    for storage in storages
        GC.gc(true)
        Metal.synchronize()
        GC.gc(true)
        result = run_storage(storage, repetitions)
        push!(results, result)
        println(P.JSON.json(result))
        flush(stdout)
    end
    if haskey(ENV, "PLAMO_BENCH_JSON")
        write(
            ENV["PLAMO_BENCH_JSON"],
            P.JSON.json((;
                julia = string(VERSION),
                metal = string(pkgversion(Metal)),
                julia_threads = Threads.nthreads(),
                repetitions,
                results,
            )),
        )
    end
end

main()
