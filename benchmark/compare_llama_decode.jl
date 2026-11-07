# Same token IDs, synchronized per token, sequential alternating runs.
using DocstringTranslationPLaMoBackend, Metal, Statistics
const B=DocstringTranslationPLaMoBackend
const E=Base.get_extension(B, :DocstringTranslationPLaMoBackendMetalExt)
const LIB = get(ENV, "PLAMO_LLAMA_BENCH_LIBRARY", "")
isempty(LIB) && error("set PLAMO_LLAMA_BENCH_LIBRARY to the compiled benchmark bridge")
length(ARGS) == 1 || error("usage: compare_llama_decode.jl model-q4_0.gguf")
const LLAMA_MODEL = abspath(only(ARGS))
const REPETITIONS = parse(Int, get(ENV, "PLAMO_BENCH_REPETITIONS", "11"))
REPETITIONS > 0 || error("repetitions must be positive")
function julia_sample(m, s)
    s.position=0
    E.forward_gpu!(m, fill(0, 128), s; token_only = true)
    started=time_ns()
    for _ = 1:64
        E.forward_gpu!(m, (0,), s; token_only = true)
    end
    return 64e9/(time_ns()-started)
end
function llama_sample(ptr)
    ccall((:q4_bench_prefill, LIB), Cint, (Ptr{Cvoid},), ptr)==0 || error("prefill")
    sec=ccall((:q4_bench_decode, LIB), Cdouble, (Ptr{Cvoid}, Cint, Cint), ptr, 64, 0)
    sec>0 || error("decode")
    return 64/sec
end
function main()
    results = []
    m=E.MetalModel(
        B.model_directory();
        context_size = 512,
        weight_storage = :q4_0,
        hybrid = true,
    )
    s=E.MetalState(m, 512)
    for (label, f32, fa) in (("default-f16-auto", 0, -1), ("f32-fa", 1, 1))
        ptr=ccall(
            (:q4_bench_create, LIB),
            Ptr{Cvoid},
            (Cstring, Cint, Cint),
            LLAMA_MODEL,
            f32,
            fa,
        )
        ptr==C_NULL && error("llama load")
        try
            julia_sample(m, s)
            llama_sample(ptr)
            a=Float64[]
            b=Float64[]
            for i = 1:REPETITIONS
                if isodd(i)
                    push!(a, julia_sample(m, s))
                    push!(b, llama_sample(ptr))
                else
                    push!(b, llama_sample(ptr))
                    push!(a, julia_sample(m, s))
                end
                println(label, " sample=", i, " julia=", a[end], " llama=", b[end])
                flush(stdout)
            end
            result = (;
                label,
                julia_samples = a,
                llama_samples = b,
                julia_median = median(a),
                llama_median = median(b),
            )
            push!(results, result)
            println(B.PLaMo.JSON.json(result))
            flush(stdout)
        finally
            ccall((:q4_bench_free, LIB), Cvoid, (Ptr{Cvoid},), ptr)
        end
    end
    if haskey(ENV, "PLAMO_BENCH_JSON")
        write(
            ENV["PLAMO_BENCH_JSON"],
            B.PLaMo.JSON.json((;
                julia = string(VERSION),
                metal = string(pkgversion(Metal)),
                julia_threads = Threads.nthreads(),
                repetitions = REPETITIONS,
                context = 512,
                prefill_tokens = 128,
                decode_tokens = 64,
                results,
            )),
        )
    end
end
main()
