# Run in an environment containing both Metal and this package:
# julia --project=/path/to/metal-environment test/metal.jl
using Test, Random, Metal, DocstringTranslationPLaMoBackend

const B = DocstringTranslationPLaMoBackend
const P = B.PLaMo
const M = Base.get_extension(B, :DocstringTranslationPLaMoBackendMetalExt)

@testset "Metal inference primitives" begin
    @test !isnothing(M)
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

    # Exercise SIMD groups with uneven numbers of keys repeatedly: all groups
    # must read the softmax maximum before its shared slot can be reused.
    for nkeys in (1, 3, 31, 33, 127, 128, 129)
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
        for _ = 1:10
            result = M.single_attention_gpu(dq, dk, dv, nkeys - 1, 1, true, 256, 128, 4, 2)
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
        for ids in ([1, 2, 3, 4], [5], [6])
            expected = P.forward!(cpu, ids, cpu_state)
            actual = M.forward_gpu!(gpu, ids, gpu_state)
            @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
            @test argmax(actual) == argmax(expected)
        end
        # Exercise the prefill chunk boundary and a subsequent decode.
        cpu_state = P.ForwardState(cpu; capacity = 200)
        gpu_state = M.MetalState(gpu, 200)
        for ids in (collect(1:128), collect(129:140), [5])
            expected = P.forward!(cpu, ids, cpu_state)
            actual = M.forward_gpu!(gpu, ids, gpu_state)
            @test actual ≈ expected rtol=2.0f-3 atol=2.0f-3
            @test argmax(actual) == argmax(expected)
        end
    end
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
