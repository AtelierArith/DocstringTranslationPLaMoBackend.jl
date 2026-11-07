# Real greedy translation timing; model load and tokenization are excluded.
using DocstringTranslationPLaMoBackend
using Statistics

if get(ENV, "PLAMO_BENCH_DEVICE", "cpu") == "metal"
    using Metal
end

const B = DocstringTranslationPLaMoBackend
const P = B.PLaMo

function main()
    device = Symbol(get(ENV, "PLAMO_BENCH_DEVICE", "cpu"))
    device in (:cpu, :metal) || throw(ArgumentError("device must be cpu or metal"))
    storage = Symbol(get(ENV, "PLAMO_BENCH_STORAGE", "bf16"))
    prefill = Symbol(get(ENV, "PLAMO_BENCH_PREFILL", "auto"))
    repetitions = parse(Int, get(ENV, "PLAMO_BENCH_REPETITIONS", "5"))
    repetitions > 0 || throw(ArgumentError("repetitions must be positive"))
    source = isempty(ARGS) ? "The meeting starts at ten." : join(ARGS, " ")
    B.configure_plamo!(;
        device,
        weight_storage = storage,
        prefill_storage = prefill,
        context_size = 512,
        max_tokens = 32,
    )
    request = B.translation_request(source, :ja, 32)
    if haskey(ENV, "PLAMO_BENCH_PROMPT_FILE")
        write(ENV["PLAMO_BENCH_PROMPT_FILE"], P.prompt(request))
    end
    translator = B.native_translator()
    if device === :metal
        println("hybrid_prefill=", !isnothing(first(translator.model.layers).prefill))
    end
    P.translate(translator, request) # compile and warm up
    results = [P.translate(translator, request) for _ = 1:repetitions]
    for (i, r) in enumerate(results)
        println((
            sample = i,
            text = r.text,
            finish = r.finish_reason,
            prompt_tokens = r.prompt_tokens,
            generated_tokens = r.generated_tokens,
            first_token_seconds = r.time_to_first_token,
            elapsed_seconds = r.elapsed,
        ))
    end
    println("median_elapsed_seconds=", median(getproperty.(results, :elapsed)))
end

main()
