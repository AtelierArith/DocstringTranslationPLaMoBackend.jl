# Run in an environment containing both Metal and this package:
# julia --project=/path/to/metal-environment test/metal.jl
using Test, Random, Metal, DocstringTranslationPLaMoBackend

const B = DocstringTranslationPLaMoBackend
const P = B.PLaMo
const M = Base.get_extension(B, :DocstringTranslationPLaMoBackendMetalExt)

@testset "Metal inference primitives" begin
    @test !isnothing(M)
    gib = 1024^3
    @test M.use_hybrid_prefill(:bf16, :auto, 9gib, 9gib, 4096)
    @test !M.use_hybrid_prefill(:bf16, :auto, 8gib, 9gib, 4096)
    @test !M.use_hybrid_prefill(:bf16, :auto, 9gib, 8gib, 4096)
    @test M.use_hybrid_prefill(:bf16, :f32, 0, 0, 4096)
    @test !M.use_hybrid_prefill(:bf16, :bf16, typemax(Int), typemax(Int), 4096)
    @test !M.use_hybrid_prefill(:f32, :f32, typemax(Int), typemax(Int), 4096)
    calls = Bool[]
    fake_loader =
        (directory; context_size, weight_storage, hybrid) -> begin
            push!(calls, hybrid)
            hybrid && throw(OutOfMemoryError())
            return :fallback
        end
    options = (context_size = 128, weight_storage = :bf16, prefill_storage = :auto)
    @test_logs (:warn, r"falling back") M.load_model_with_fallback(
        fake_loader,
        "unused",
        options,
        true,
    ) == :fallback
    @test calls == [true, false]
    weak_buffer = Ref{WeakRef}()
    released_before_retry = Ref(false)
    allocation_loader =
        (directory; context_size, weight_storage, hybrid) -> begin
            if hybrid
                data = Metal.zeros(Float32, 1_000_000)
                weak_buffer[] = WeakRef(data)
                throw(OutOfMemoryError())
            end
            released_before_retry[] = isnothing(weak_buffer[].value)
            return :fallback
        end
    @test_logs (:warn, r"falling back") M.load_model_with_fallback(
        allocation_loader,
        "unused",
        options,
        true,
    ) == :fallback
    @test released_before_retry[]
    calls = Bool[]
    options = (context_size = 128, weight_storage = :bf16, prefill_storage = :f32)
    @test_throws OutOfMemoryError M.load_model_with_fallback(
        fake_loader,
        "unused",
        options,
        true,
    )
    @test calls == [true]
    broken_loader =
        (directory; context_size, weight_storage, hybrid) -> error("invalid checkpoint")
    options = (context_size = 128, weight_storage = :bf16, prefill_storage = :auto)
    @test_throws ErrorException M.load_model_with_fallback(
        broken_loader,
        "unused",
        options,
        true,
    )
    @test M.gpu_token(MtlArray(reshape(Float32[-2, 3, 3, 1], :, 1))) == 1
    @test M.gpu_token(MtlArray(reshape(Float32[-2, -3, 1], :, 1))) == 2
    @test M.gpu_token(MtlArray(reshape(Float32[1, NaN, 2], :, 1))) == 1
    tied = zeros(Float32, 107520, 1)
    tied[17] = tied[60000] = 1
    @test M.gpu_token(MtlArray(tied)) == 16
    Random.seed!(0xdeadbeef)
    x = randn(Float32, 8, 3)
    weight = randn(Float32, 8)
    @test Array(M.rms_norm_gpu(MtlArray(x), MtlArray(weight), 1.0f-5)) ≈
          P.rms_norm(x, weight, 1.0f-5) rtol=2.0f-5 atol=2.0f-5

    residual = randn(Float32, size(x))
    @test Array(
        M.rms_norm_gpu(
            MtlArray(x),
            MtlArray(weight),
            1.0f-5;
            residual = MtlArray(residual),
        ),
    ) ≈ P.rms_norm(x, weight, 1.0f-5) .+ residual rtol=2.0f-5 atol=2.0f-5

    x_wide = randn(Float32, 1537, 5)
    norm_wide = randn(Float32, 1537)
    @test Array(M.rms_norm_gpu(MtlArray(x_wide), MtlArray(norm_wide), 1.0f-5)) ≈
          P.rms_norm(x_wide, norm_wide, 1.0f-5) rtol=2.0f-5 atol=2.0f-5

    input = randn(Float32, 1537, 1)
    skip = randn(Float32, 1537, 1)
    post_weight = randn(Float32, 1537)
    pre_weight = randn(Float32, 1537)
    expected_residual = P.rms_norm(input, post_weight, 1.0f-5) .+ skip
    expected_norm = P.rms_norm(expected_residual, pre_weight, 1.0f-5)
    result_residual = MtlArray{Float32}(undef, size(input))
    result_norm = MtlArray{Float32}(undef, size(input))
    M.dual_rms_gpu!(
        result_residual,
        result_norm,
        MtlArray(input),
        MtlArray(skip),
        MtlArray(post_weight),
        MtlArray(pre_weight),
        1.0f-5,
    )
    @test Array(result_residual) ≈ expected_residual rtol=2.0f-5 atol=2.0f-5
    @test Array(result_norm) ≈ expected_norm rtol=2.0f-5 atol=2.0f-5

    for columns in (1, 3)
        host_weight = Metal.BFloat16.(randn(Float32, 19, 35))
        host_input = randn(Float32, 19, columns)
        expected_projection = transpose(Float32.(host_weight)) * host_input
        actual_projection = M.projection_gpu(MtlArray(host_weight), MtlArray(host_input))
        @test Array(actual_projection) ≈ expected_projection rtol=2.0f-5 atol=2.0f-5
    end

    for columns in (1, 3)
        host_weight = randn(Float32, 19, 35)
        host_input = randn(Float32, 19, columns)
        @test Array(M.projection_gpu(MtlArray(host_weight), MtlArray(host_input))) ≈
              transpose(host_weight) * host_input rtol=2.0f-5 atol=2.0f-5
    end

    host_weight = Metal.BFloat16.(randn(Float32, 128, 32))
    host_input = randn(Float32, 128, 1)
    @test Array(M.projection_gpu(MtlArray(host_weight), MtlArray(host_input))) ≈
          transpose(Float32.(host_weight)) * host_input rtol=2.0f-5 atol=2.0f-5

    host_weight = Metal.BFloat16.(randn(Float32, 512, 36))
    host_input = randn(Float32, 512, 1)
    @test Array(M.projection_gpu(MtlArray(host_weight), MtlArray(host_input))) ≈
          transpose(Float32.(host_weight)) * host_input rtol=2.0f-5 atol=2.0f-5

    host_weight = Metal.BFloat16.(randn(Float32, 512, 40))
    host_input = randn(Float32, 512, 1)
    @test Array(M.projection_gpu(MtlArray(host_weight), MtlArray(host_input))) ≈
          transpose(Float32.(host_weight)) * host_input rtol=2.0f-5 atol=2.0f-5

    q = randn(Float32, 8, 3)
    norm = randn(Float32, 4)
    co = randn(Float32, 2, 5)
    si = randn(Float32, 2, 5)
    expected = copy(q)
    P.rotate_heads!(expected, norm, 1.0f-5, 2, co, si, 2)
    actual = MtlArray(q)
    M.rotate_heads_gpu!(actual, MtlArray(norm), 1.0f-5, 2, MtlArray(co), MtlArray(si), 2)
    @test Array(actual) ≈ expected rtol=2.0f-5 atol=2.0f-5

    qkv = randn(Float32, 16, 3)
    knorm = randn(Float32, 4)
    expected_qkv = copy(qkv)
    @views P.rotate_heads!(expected_qkv[1:8, :], norm, 1.0f-5, 2, co, si, 2)
    @views P.rotate_heads!(expected_qkv[9:12, :], knorm, 1.0f-5, 1, co, si, 2)
    actual_qkv = MtlArray(qkv)
    M.rotate_qk_gpu!(
        actual_qkv,
        MtlArray(norm),
        MtlArray(knorm),
        1.0f-5,
        2,
        1,
        MtlArray(co),
        MtlArray(si),
        2,
    )
    @test Array(actual_qkv) ≈ expected_qkv rtol=2.0f-5 atol=2.0f-5
    cache_keys = MtlArray(zeros(Float32, 4, 5))
    cache_values = MtlArray(zeros(Float32, 4, 5))
    actual_qkv_cache = MtlArray(qkv)
    M.rotate_qk_gpu!(
        actual_qkv_cache,
        MtlArray(norm),
        MtlArray(knorm),
        1.0f-5,
        2,
        1,
        MtlArray(co),
        MtlArray(si),
        2;
        keys = cache_keys,
        values = cache_values,
    )
    @test Array(actual_qkv_cache) ≈ expected_qkv rtol=2.0f-5 atol=2.0f-5
    @test Array(cache_keys)[:, 3:5] ≈ expected_qkv[9:12, :] rtol=2.0f-5 atol=2.0f-5
    @test Array(cache_values)[:, 3:5] == qkv[13:16, :]

    # Decode uses a single-column view into the combined QKV buffer.
    wide = randn(Float32, 2304, 1)
    qview = @view wide[1:2048, :]
    reference = copy(qview)
    norm128 = randn(Float32, 128)
    co128 = randn(Float32, 64, 8)
    si128 = randn(Float32, 64, 8)
    P.rotate_heads!(reference, norm128, 1.0f-5, 16, co128, si128, 2)
    device_wide = MtlArray(wide)
    device_view = @view device_wide[1:2048, :]
    M.rotate_heads_gpu!(
        device_view,
        MtlArray(norm128),
        1.0f-5,
        16,
        MtlArray(co128),
        MtlArray(si128),
        2,
    )
    @test Array(device_view) ≈ reference rtol=2.0f-5 atol=2.0f-5

    # Single-KV-head prefill can flatten query heads into a matrix product.
    # Odd dimensions, offset caches and partial causal windows must agree.
    for dim in (7, 128), count in (3, 129), full in (true, false)
        position, start, heads = 5, 2, 3
        q = randn(Float32, dim * heads, count)
        k = randn(Float32, dim, position + count)
        v = randn(Float32, dim, position + count)
        reference = similar(q)
        P.grouped_attention!(reference, q, k, v, position, start, full, 4, dim, heads, 1)
        result = M.attention_gpu(
            MtlArray(q),
            MtlArray(k),
            MtlArray(v),
            position,
            start,
            full,
            4,
            dim,
            heads,
            1,
        )
        @test Array(result) ≈ reference rtol=3.0f-5 atol=3.0f-5
    end

    for full in (true, false), window in (2, 128)
        decode_q = randn(Float32, 128 * 4, 1)
        decode_k = randn(Float32, 128 * 2, 137)
        decode_v = randn(Float32, 128 * 2, 137)
        reference = similar(decode_q)
        P.grouped_attention!(
            reference,
            decode_q,
            decode_k,
            decode_v,
            136,
            1,
            full,
            window,
            128,
            4,
            2,
        )
        actual = M.single_attention_gpu(
            MtlArray(decode_q),
            MtlArray(decode_k),
            MtlArray(decode_v),
            136,
            1,
            full,
            window,
            128,
            4,
            2,
        )
        @test Array(actual) ≈ reference rtol=3.0f-5 atol=3.0f-5
    end

    # Exercise partial SIMD groups and split boundaries repeatedly. Reusing a
    # larger, poisoned workspace must not read unused or stale partitions.
    partial = MtlArray{Float32}(undef, 130, 4, 8)
    for nkeys in (1, 3, 31, 32, 33, 64, 65, 127, 128, 129)
        query_decode = randn(Float32, 128 * 4, 1)
        keys_decode = randn(Float32, 128 * 2, nkeys)
        values_decode = randn(Float32, 128 * 2, nkeys)
        reference = similar(query_decode)
        P.grouped_attention!(
            reference,
            query_decode,
            keys_decode,
            values_decode,
            nkeys - 1,
            1,
            true,
            256,
            128,
            4,
            2,
        )
        dq, dk, dv = MtlArray(query_decode), MtlArray(keys_decode), MtlArray(values_decode)
        for repetition = 1:10
            fill!(partial, NaN32)
            result = if repetition <= 5
                M.single_attention_gpu(dq, dk, dv, nkeys - 1, 1, true, 256, 128, 4, 2)
            else
                out = similar(dq)
                M.single_attention_gpu!(
                    out,
                    dq,
                    dk,
                    dv,
                    nkeys - 1,
                    1,
                    true,
                    256,
                    128,
                    4,
                    2;
                    partial,
                )
            end
            @test Array(result) ≈ reference rtol=3.0f-5 atol=3.0f-5
        end
    end

    for nkeys in (2048, 2049)
        query_decode = randn(Float32, 8, 1)
        keys_decode = randn(Float32, 4, nkeys)
        values_decode = randn(Float32, 4, nkeys)
        reference = similar(query_decode)
        P.grouped_attention!(
            reference,
            query_decode,
            keys_decode,
            values_decode,
            nkeys - 1,
            1,
            true,
            nkeys,
            4,
            2,
            1,
        )
        actual = M.attention_gpu(
            MtlArray(query_decode),
            MtlArray(keys_decode),
            MtlArray(values_decode),
            nkeys - 1,
            1,
            true,
            nkeys,
            4,
            2,
            1,
        )
        @test Array(actual) ≈ reference rtol=3.0f-5 atol=3.0f-5
    end

    query = randn(Float32, 8, 3)
    keys = randn(Float32, 4, 5)
    values = randn(Float32, 4, 5)
    for full in (true, false), window in (2, 5)
        output = similar(query)
        P.per_head_attention!(output, query, keys, values, 2, 1, full, window, 4, 2, 1)
        result = M.attention_gpu(
            MtlArray(query),
            MtlArray(keys),
            MtlArray(values),
            2,
            1,
            full,
            window,
            4,
            2,
            1,
        )
        @test Array(result) ≈ output rtol=2.0f-5 atol=2.0f-5
    end
end

if get(ENV, "PLAMO_METAL_REAL_MODEL", "0") == "1"
    @testset "Metal forward matches CPU" begin
        directory = B.model_directory()
        cpu = P.Model(directory; context_size = 256, weight_storage = :bf16)
        gpu = M.MetalModel(directory; context_size = 256, weight_storage = :bf16)
        cpu_state = P.ForwardState(cpu; capacity = 32)
        gpu_state = M.MetalState(gpu, 32)
        token_state = M.MetalState(gpu, 32)
        for ids in ([1, 2, 3, 4], [5], [6])
            expected = P.forward!(cpu, ids, cpu_state)
            actual = M.forward_gpu!(gpu, ids, gpu_state)
            @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
            @test argmax(actual) == argmax(expected)
            @test M.forward_gpu!(gpu, ids, token_state; token_only = true) ==
                  argmax(expected) - 1
        end
        # Exercise the prefill chunk boundary and a subsequent decode.
        cpu_state = P.ForwardState(cpu; capacity = 200)
        gpu_state = M.MetalState(gpu, 200)
        token_state = M.MetalState(gpu, 200)
        for ids in (collect(1:128), collect(129:140), [5])
            expected = P.forward!(cpu, ids, cpu_state)
            actual = M.forward_gpu!(gpu, ids, gpu_state)
            @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
            @test argmax(actual) == argmax(expected)
            @test M.forward_gpu!(gpu, ids, token_state; token_only = true) ==
                  argmax(expected) - 1
        end
        # A one-token tail of a 129-token prompt uses the decode workspace.
        cpu_state = P.ForwardState(cpu; capacity = 140)
        gpu_state = M.MetalState(gpu, 140)
        token_state = M.MetalState(gpu, 140)
        for ids in (collect(1:128), [129])
            expected = P.forward!(cpu, ids, cpu_state)
            actual = M.forward_gpu!(gpu, ids, gpu_state)
            @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
            @test argmax(actual) == argmax(expected)
            @test M.forward_gpu!(gpu, ids, token_state; token_only = true) ==
                  argmax(expected) - 1
        end
    end
end

if get(ENV, "PLAMO_METAL_REAL_MODEL", "0") == "1" && M.use_hybrid_prefill(
    :bf16,
    :auto,
    Metal.free_memory(),
    Sys.free_memory(),
    M.kv_cache_bytes(B.model_directory(), 256),
)
    @testset "Float32 prefill cache matches CPU" begin
        GC.gc(true)
        directory = B.model_directory()
        cpu = P.Model(directory; context_size = 256, weight_storage = :bf16)
        gpu = M.MetalModel(
            directory;
            context_size = 256,
            weight_storage = :bf16,
            hybrid = true,
        )
        @test !isnothing(first(gpu.layers).prefill)
        cpu_state = P.ForwardState(cpu; capacity = 160)
        gpu_state = M.MetalState(gpu, 160)
        token_state = M.MetalState(gpu, 160)
        for ids in (collect(1:128), [5], [6])
            expected = P.forward!(cpu, ids, cpu_state)
            actual = M.forward_gpu!(gpu, ids, gpu_state)
            @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
            @test argmax(actual) == argmax(expected)
            @test M.forward_gpu!(gpu, ids, token_state; token_only = true) ==
                  argmax(expected) - 1
        end
    end
elseif get(ENV, "PLAMO_METAL_REAL_MODEL", "0") == "1"
    @warn "Skipping Float32 prefill cache model test because available memory is below the auto threshold"
end

if get(ENV, "PLAMO_METAL_REAL_MODEL", "0") == "1"
    @testset "Metal translation through public configuration" begin
        try
            B.configure_plamo!(; device = :metal, context_size = 128, max_tokens = 32)
            @test strip(
                string(B.translate_with_plamo("The meeting starts at ten."; lang = :ja)),
            ) == "会議は10時に始まります。"
            @test B.NATIVE_TRANSLATOR[] isa B.AbstractMetalTranslator
        finally
            B.configure_plamo!()
        end
    end
end

if get(ENV, "PLAMO_METAL_F32_MODEL", "0") == "1"
    @testset "Float32 Metal forward matches CPU" begin
        GC.gc(true)
        directory = B.model_directory()
        cpu = P.Model(directory; context_size = 128, weight_storage = :f32)
        gpu = M.MetalModel(directory; context_size = 128, weight_storage = :f32)
        cpu_state = P.ForwardState(cpu; capacity = 32)
        gpu_state = M.MetalState(gpu, 32)
        token_state = M.MetalState(gpu, 32)
        for ids in ([1, 2, 3, 4], [5], [6])
            expected = P.forward!(cpu, ids, cpu_state)
            actual = M.forward_gpu!(gpu, ids, gpu_state)
            @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
            @test argmax(actual) == argmax(expected)
            @test M.forward_gpu!(gpu, ids, token_state; token_only = true) ==
                  argmax(expected) - 1
        end
    end
end

if get(ENV, "PLAMO_METAL_LONG_MODEL", "0") == "1"
    @testset "Long-context Metal decode matches CPU" begin
        directory = B.model_directory()
        cpu = P.Model(directory; context_size = 2052, weight_storage = :bf16)
        gpu = M.MetalModel(directory; context_size = 2052, weight_storage = :bf16)
        cpu_state = P.ForwardState(cpu; capacity = 2050)
        gpu_state = M.MetalState(gpu, 2050)
        token_state = M.MetalState(gpu, 2050)
        position = 2048
        rows = cpu.config.kv_heads * cpu.config.head_dim
        for index in eachindex(cpu.layers)
            host_keys = randn(Float32, rows, position)
            host_values = randn(Float32, rows, position)
            @views cpu_state.keys[index][:, 1:position] .= host_keys
            @views cpu_state.values[index][:, 1:position] .= host_values
            @views gpu_state.keys[index][:, 1:position] .= MtlArray(host_keys)
            @views gpu_state.values[index][:, 1:position] .= MtlArray(host_values)
            @views token_state.keys[index][:, 1:position] .= MtlArray(host_keys)
            @views token_state.values[index][:, 1:position] .= MtlArray(host_values)
        end
        cpu_state.position = position
        gpu_state.position = position
        token_state.position = position
        expected = P.forward!(cpu, [5], cpu_state)
        actual = M.forward_gpu!(gpu, [5], gpu_state)
        @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
        @test argmax(actual) == argmax(expected)
        @test M.forward_gpu!(gpu, [5], token_state; token_only = true) ==
              argmax(expected) - 1
    end
end
