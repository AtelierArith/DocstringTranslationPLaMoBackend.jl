@testset "Quantized Metal blocks" begin
    q8_values = Float32[-127, 127, -1, 1, 0.5, -0.5, zeros(Float32, 26)...]
    q4_values = repeat(Float32.(-8:7), 2)
    for (bits, values) in ((8, q8_values), (4, q4_values))
        raw = reinterpret(UInt16, Metal.BFloat16.(reshape(values, 32, 1)))
        packed, scales = M.quantize_blocks(raw, Val(bits))
        @test scales[:, 1] == Float16[1]
        @test all(iszero, scales[:, 2])
        if bits == 8
            @test reinterpret(Int8, packed[:, 1]) ==
                  Int8[-127, 127, -1, 1, 1, -1, zeros(Int8, 26)...]
        else
            @test packed[:, 1] ==
                  repeat(UInt8[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe], 2)
        end
        weight = M.quantize_matrix(raw, Val(bits))
        expected =
            bits == 8 ? Float32[-127, 127, -1, 1, 1, -1, zeros(Float32, 26)...] : values
        @test size(weight) == (32, 1)
        @test vec(Array(M.dequantize_gpu(weight))) == expected
        @test vec(Array(M.embedding_gpu(weight, [0]))) == expected
    end
    for bits in (4, 8)
        @test_throws DimensionMismatch M.quantize_blocks(zeros(UInt16, 31, 2), Val(bits))
        bad = fill(reinterpret(UInt16, Metal.BFloat16(NaN32)), 32, 1)
        @test_throws ArgumentError M.quantize_blocks(bad, Val(bits))
    end
end

@testset "Quantized Metal projections and tails" begin
    for bits in (4, 8), width in (32, 64, 1536), columns in (1, 2, 17)
        # Integer weights use a unit block scale: the expected values are
        # independent of the GPU unpacking and dequantization implementations.
        values =
            bits == 8 ? Float32[-127, 127, -1, 1, zeros(Float32, 28)...] :
            repeat(Float32.(-8:7), 2)
        expected_weight = repeat(reshape(repeat(values, width ÷ 32), :, 1), 1, columns)
        for column = 1:columns, block = 1:(width÷32)
            factor = Float32(2.0^(-mod(block + column, 5)))
            expected_weight[((block-1)*32+1):(block*32), column] .*= factor
        end
        raw = reinterpret(UInt16, Metal.BFloat16.(expected_weight))
        weight = M.quantize_matrix(raw, Val(bits))
        for tokens in (1, 3)
            input = reshape(Float32.(sin.(1:(width*tokens))), width, tokens)
            actual = MtlArray{Float32}(undef, columns, tokens)
            M.projection_gpu!(actual, weight, MtlArray(input))
            @test Array(actual) ≈ transpose(expected_weight) * input rtol=3.0f-5 atol=3.0f-5
        end
    end
end

@testset "Quantized Metal prefill policy" begin
    for storage in (:q8_0, :q4_0)
        @test M.use_hybrid_prefill(storage, :auto, 9*1024^3, 9*1024^3, 4096)
        @test !M.use_hybrid_prefill(storage, :native, 9*1024^3, 9*1024^3, 4096)
        @test M.use_hybrid_prefill(storage, :f32, 0, 0, 4096)
    end
end

# Independent scalar CPU expansion for whole-model reference checks.
function quantized_cpu_weights(weight)
    data = Array(weight.data)
    scales = Float32.(Array(weight.scales))
    result = Matrix{Float32}(undef, size(weight))
    bits = weight.bits isa Val{8} ? 8 : 4
    for i in eachindex(result)
        value = if bits == 8
            Float32(reinterpret(Int8, data[i]))
        else
            Float32(Int((data[(i-1)÷2+1] >> (((i-1)%2)*4)) & 0x0f) - 8)
        end
        result[i] = value * scales[(i-1)÷32+1]
    end
    return result
end

function quantized_cpu_model(gpu)
    projection(w) = P.Projection(quantized_cpu_weights(w), nothing)
    layers = [
        P.DecoderLayer(
            projection(l.qkv),
            projection(l.out),
            projection(l.gate_up),
            projection(l.down),
            Array(l.qnorm),
            Array(l.knorm),
            Array(l.pre_attn),
            Array(l.post_attn),
            Array(l.pre_mlp),
            Array(l.post_mlp),
            l.full,
        ) for l in gpu.layers
    ]
    return P.Model(
        gpu.config,
        quantized_cpu_weights(gpu.embedding),
        projection(gpu.head),
        Array(gpu.norm),
        layers,
        Array(gpu.local_cos),
        Array(gpu.local_sin),
        Array(gpu.full_cos),
        Array(gpu.full_sin),
        gpu.capacity,
    )
end

if get(ENV, "PLAMO_METAL_QUANT_MODEL", "0") == "1"
    @testset "Quantized Metal real model" begin
        for storage in (:q8_0, :q4_0)
            GC.gc(true)
            gpu = M.MetalModel(
                B.model_directory();
                context_size = 2052,
                weight_storage = storage,
                hybrid = true,
            )
            cpu = quantized_cpu_model(gpu)
            state = M.MetalState(gpu, 160)
            tokens = M.MetalState(gpu, 160)
            reference = P.ForwardState(cpu; capacity = 160)
            for ids in ([1, 2, 3], [4], collect(5:132), [133])
                expected = P.forward!(cpu, ids, reference)
                actual = M.forward_gpu!(gpu, ids, state)
                @test actual ≈ expected rtol=3.0f-3 atol=3.0f-3
                @test argmax(actual) == argmax(expected)
                @test M.forward_gpu!(gpu, ids, tokens; token_only = true) ==
                      argmax(expected)-1
            end
            # The cache-disabled path expands only the current projection.
            layers = [
                M.MetalLayer(
                    l.qkv,
                    l.out,
                    l.gate_up,
                    l.down,
                    l.qnorm,
                    l.knorm,
                    l.pre_attn,
                    l.post_attn,
                    l.pre_mlp,
                    l.post_mlp,
                    l.full,
                    nothing,
                ) for l in gpu.layers
            ]
            native = M.MetalModel(
                gpu.config,
                gpu.embedding,
                gpu.head,
                gpu.norm,
                layers,
                gpu.local_cos,
                gpu.local_sin,
                gpu.full_cos,
                gpu.full_sin,
                gpu.capacity,
            )
            state = M.MetalState(native, 16)
            reference = P.ForwardState(cpu; capacity = 16)
            for ids in ([1, 2, 3], [4])
                @test M.forward_gpu!(native, ids, state) ≈ P.forward!(cpu, ids, reference) rtol=3.0f-3 atol=3.0f-3
            end
            tokenizer = P.Tokenizer(B.model_directory())
            request = B.translation_request("The meeting starts at ten.", :ja, 32)
            @test P.translate(M.MetalTranslator(gpu, tokenizer), request).text ==
                  "会議は10時に始まります。"
            # Decode immediately beyond the optimized attention range.
            state = M.MetalState(gpu, 2052)
            reference = P.ForwardState(cpu; capacity = 2052)
            for index in eachindex(cpu.layers)
                host_keys = randn(Float32, size(state.keys[index], 1), 2048)
                host_values = randn(Float32, size(state.values[index], 1), 2048)
                copyto!(@view(state.keys[index][:, 1:2048]), host_keys)
                copyto!(@view(state.values[index][:, 1:2048]), host_values)
                @views reference.keys[index][:, 1:2048] .= host_keys
                @views reference.values[index][:, 1:2048] .= host_values
            end
            state.position = reference.position = 2048
            @test M.forward_gpu!(gpu, (1,), state) ≈ P.forward!(cpu, [1], reference) rtol=3.0f-3 atol=3.0f-3
        end
    end
end

@testset "Quantized scale extremes" begin
    for bits in (4, 8)
        tiny = fill(UInt16(1), 32, 1) # Smallest positive BF16 subnormal.
        weight = M.quantize_matrix(tiny, Val(bits))
        @test all(iszero, Array(M.dequantize_gpu(weight)))
        huge = fill(reinterpret(UInt16, Metal.BFloat16(1.0f10)), 32, 1)
        @test_throws ArgumentError M.quantize_blocks(huge, Val(bits))
    end
end
