using AllocCheck, DocstringTranslationPLaMoBackend, Metal, Profile, Statistics

const Backend = DocstringTranslationPLaMoBackend
const GPU = Base.get_extension(Backend, :DocstringTranslationPLaMoBackendMetalExt)

decode_vector!(model, state) = GPU.forward_gpu!(model, [0], state; token_only = true)
decode_token!(model, state) = GPU.forward_gpu!(model, (0,), state; token_only = true)
decode_logits!(model, state) = GPU.forward_gpu!(model, (0,), state; token_only = false)
prefill_token!(model, state, ids) = GPU.forward_gpu!(model, ids, state; token_only = true)

function measure_allocations(f::F, model, state, args...; position = 128) where {F}
    for _ = 1:3
        state.position = position
        f(model, state, args...)
    end
    samples = map(1:5) do _
        state.position = position
        @allocated f(model, state, args...)
    end
    return Int(median(samples))
end

function static_report(f, args)
    println("\nAnalyzing ", nameof(f), " with AllocCheck...")
    flush(stdout)
    findings = AllocCheck.check_allocs(f, typeof.(args))
    println("\nAllocCheck: ", nameof(f), " (", length(findings), " findings)")
    allocations = count(x -> x isa AllocCheck.AllocationSite, findings)
    runtime = count(x -> x isa AllocCheck.AllocatingRuntimeCall, findings)
    dispatches = count(x -> x isa AllocCheck.DynamicDispatch, findings)
    println(
        "potential allocation sites=$allocations runtime calls=$runtime dynamic dispatches=$dispatches",
    )
    groups = Dict{String,Int}()
    for finding in findings
        frame = findfirst(package_frame, finding.backtrace)
        location =
            isnothing(frame) ? "dependencies/runtime" :
            string(finding.backtrace[frame].file, ":", finding.backtrace[frame].line)
        label = string(nameof(typeof(finding)), " via ", location)
        groups[label] = get(groups, label, 0) + 1
        if get(ENV, "PLAMO_ALLOC_FULL", "0") == "1"
            show(stdout, finding)
            println()
        end
    end
    for (label, count) in sort!(collect(groups); by = last, rev = true)
        println(count, "\t", label)
    end
end

package_frame(frame) =
    occursin("/ext/", string(frame.file)) &&
    occursin("DocstringTranslationPLaMoBackend", string(frame.file))

function allocation_profile(model, state)
    state.position = 128
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1.0 decode_token!(model, state)
    groups = Dict{String,Tuple{Int,Int}}()
    for allocation in Profile.Allocs.fetch().allocs
        frame = findfirst(package_frame, allocation.stacktrace)
        label =
            isnothing(frame) ? "dependencies/runtime" :
            string(
                allocation.stacktrace[frame].file,
                ":",
                allocation.stacktrace[frame].line,
            )
        bytes, count = get(groups, label, (0, 0))
        groups[label] = (bytes + allocation.size, count + 1)
    end
    println("\nRuntime decode allocation stacks (bytes, count, package call site):")
    for (label, (bytes, count)) in sort!(collect(groups); by = x -> last(x)[1], rev = true)
        println(bytes, "\t", count, "\t", label)
    end
    Profile.Allocs.clear()
end

function buffer_profile(f::F, model, state, args...; position = 128) where {F}
    state.position = position
    activity = Metal.@profile f(model, state, args...)
    println("\nGPU buffer creation API calls for ", nameof(f), ":")
    for (name, count) in zip(activity.host.name, activity.host.calls)
        occursin("newBuffer", name) && println(name, " = ", count)
    end
end

function main()
    directory = Backend.model_directory()
    Backend.verify_model(directory)
    storage = Symbol(get(ENV, "PLAMO_ALLOC_STORAGE", "bf16"))
    storage in (:bf16, :f32, :q8_0, :q4_0) || error("unknown weight storage")
    hybrid = GPU.use_hybrid_prefill(
        storage,
        :auto,
        Metal.free_memory(),
        Sys.free_memory(),
        GPU.kv_cache_bytes(directory, 512),
    )
    model = GPU.MetalModel(directory; context_size = 512, weight_storage = storage, hybrid)
    state = GPU.MetalState(model, 512)
    ids = fill(0, 128)
    GPU.forward_gpu!(model, ids, state; token_only = true)
    println(
        "Julia ",
        VERSION,
        "; Metal ",
        pkgversion(Metal),
        "; AllocCheck ",
        pkgversion(AllocCheck),
        "; weight_storage=",
        storage,
        "; hybrid_prefill=",
        hybrid,
    )
    println("Warm Julia host allocations; model/state setup excluded:")
    for f in (decode_vector!, decode_token!, decode_logits!)
        println(nameof(f), " bytes/token=", measure_allocations(f, model, state))
    end
    println(
        "prefill_token! bytes/128-token-call=",
        measure_allocations(prefill_token!, model, state, ids; position = 0),
    )
    for f in (decode_vector!, decode_token!, decode_logits!)
        static_report(f, (model, state))
    end
    static_report(prefill_token!, (model, state, ids))
    allocation_profile(model, state)
    buffer_profile(decode_token!, model, state)
    buffer_profile(prefill_token!, model, state, ids; position = 0)
end

main()
