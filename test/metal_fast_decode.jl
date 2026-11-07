function decode_divide_kernel!(output, divisor)
    @inbounds output[1] = Int32(1) ÷ divisor[1]
    return
end

@testset "Decode GPU diagnostics" begin
    level = M.decode_debug_level()
    @test level == (Base.JLOptions().debug_level >= 2 ? 2 : 0)
    output = Metal.zeros(Int32, 1)
    divisor = MtlArray(Int32[0])
    @metal threads=1 debug_level=level decode_divide_kernel!(output, divisor)
    exception = try
        Metal.synchronize()
        nothing
    catch err
        err
    end
    @test exception isa Metal.KernelException
    if level == 0 && exception isa Metal.KernelException
        @test isempty(exception.name)
        @test isempty(exception.reason)
    elseif level == 2 && exception isa Metal.KernelException
        @test !isempty(exception.name)
        @test !isempty(exception.backtrace)
    end
    # A reported exception must not poison later inference launches.
    copyto!(divisor, Int32[1])
    @metal threads=1 debug_level=level decode_divide_kernel!(output, divisor)
    @test Array(output) == Int32[1]
end

@testset "Masked Q4 matvec" begin
    for width in (32, 64, 1536, 2048, 6144), columns in (4, 8, 12)
        source = repeat(reshape(repeat(Float32.(-8:7), 2*(width÷32)), :, 1), 1, columns)
        for column = 1:columns, block = 1:(width÷32)
            @views source[((block-1)*32+1):(block*32), column] .*=
                Float32(2.0^(-mod(column+block, 5)))
        end
        weight = M.quantize_matrix(reinterpret(UInt16, Metal.BFloat16.(source)), Val(4))
        input = reshape(Float32.(sin.(1:width)), :, 1)
        expected = transpose(source) * input
        actual = MtlArray{Float32}(undef, columns, 1)
        @metal threads=32 groups=columns÷4 M.q4_matvec_masked_kernel!(
            actual,
            weight.data,
            weight.scales,
            MtlArray(input),
            UInt32(width),
            UInt32(columns),
        )
        @test Array(actual) ≈ expected rtol=4.0f-5 atol=4.0f-5
        M.projection_gpu!(actual, weight, MtlArray(input))
        @test Array(actual) ≈ expected rtol=4.0f-5 atol=4.0f-5
    end
end

@testset "Vector decode normalization" begin
    for width in (128, 256, 1536, 6144), residual_enabled in (false, true)
        input = reshape(Float32.(cos.(1:width)), :, 1)
        residual = input .* 0.2f0
        post = Float32.(sin.(1:width))
        pre = post .+ 1.0f0
        expected = P.rms_norm(input, post, 1.0f-5) .+ residual
        output = MtlArray{Float32}(undef, width, 1)
        normalized = similar(output)
        M.dual_rms_gpu!(
            output,
            normalized,
            MtlArray(input),
            MtlArray(residual),
            MtlArray(post),
            MtlArray(pre),
            1.0f-5,
        )
        @test Array(output) ≈ expected rtol=3.0f-5 atol=3.0f-5
        @test Array(normalized) ≈ P.rms_norm(expected, pre, 1.0f-5) rtol=3.0f-5 atol=3.0f-5
        skip = residual_enabled ? MtlArray(residual) : nothing
        M.rms_norm_gpu!(output, MtlArray(input), MtlArray(post), 1.0f-5; residual = skip)
        @test Array(output) ≈ (
            P.rms_norm(input, post, 1.0f-5) .+
            (residual_enabled ? residual : zeros(Float32, size(input)))
        ) rtol=3.0f-5 atol=3.0f-5
    end
end

@testset "UInt32 fused QK rotation" begin
    Random.seed!(0xa91)
    width, qheads, kheads, position = 128, 4, 2, 7
    for count in (1, 3), padded in (false, true)
        rows = width * (qheads + 2kheads)
        qkv = randn(Float32, rows + (padded ? 8 : 0), count)
        qnorm, knorm = randn(Float32, width), randn(Float32, width)
        cosines = randn(Float32, width÷2 + (padded ? 4 : 0), 20)
        sines = randn(Float32, width÷2 + (padded ? 8 : 0), 20)
        keys = randn(Float32, width*kheads + (padded ? 4 : 0), 20)
        values = randn(Float32, width*kheads + (padded ? 8 : 0), 20)
        actual, expected = MtlArray(qkv), MtlArray(qkv)
        ak, av, ek, ev = MtlArray(keys), MtlArray(values), MtlArray(keys), MtlArray(values)
        qn, kn, co, si =
            MtlArray(qnorm), MtlArray(knorm), MtlArray(cosines), MtlArray(sines)
        @metal threads=128 groups=cld((qheads+kheads)*count, 4) M.rotate_qk_kernel!(
            expected,
            ek,
            ev,
            qn,
            kn,
            1.0f-5,
            co,
            si,
            position,
            width,
            qheads,
            kheads,
            count,
        )
        M.rotate_qk_gpu!(
            actual,
            qn,
            kn,
            1.0f-5,
            qheads,
            kheads,
            co,
            si,
            position;
            keys = ak,
            values = av,
        )
        @test Array(actual) ≈ Array(expected) rtol=2.0f-5 atol=2.0f-5
        @test Array(ak) ≈ Array(ek) rtol=2.0f-5 atol=2.0f-5
        @test Array(av) ≈ Array(ev) rtol=2.0f-5 atol=2.0f-5
    end
end

@testset "Decode fast-path fallbacks" begin
    for columns in (3, 4)
        width = 128
        raw = reinterpret(UInt16, Metal.BFloat16.(randn(Float32, width, columns)))
        weight = M.quantize_matrix(raw, Val(4))
        dense_weight = Array(M.dequantize_gpu(weight))
        input = randn(Float32, width, 1)
        backing = MtlArray(repeat(input; inner = (2, 1)))
        strided_input = @view backing[1:2:end, :]
        output_backing = Metal.zeros(Float32, 2columns, 1)
        strided_output = @view output_backing[1:2:end, :]
        M.projection_gpu!(strided_output, weight, strided_input)
        @test Array(strided_output) ≈ transpose(dense_weight)*input rtol=3.0f-5 atol=3.0f-5
        norm = randn(Float32, width)
        norm_output = similar(MtlArray(input))
        M.rms_norm_gpu!(norm_output, strided_input, MtlArray(norm), 1.0f-5)
        @test Array(norm_output) ≈ P.rms_norm(input, norm, 1.0f-5) rtol=3.0f-5 atol=3.0f-5
    end
    a = Metal.zeros(Float32, 128)
    @test M.vector_norm_eligible(128, a)
    @test !M.vector_norm_eligible(0, a)
    @test !M.vector_norm_eligible(Int(typemax(UInt32)), a)
    @test !M.vector_norm_eligible(Int(typemax(UInt32))+1, a)
    @test !M.vector_norm_eligible(128, @view a[1:2:end])
end

@testset "Attention query in combined QKV buffer" begin
    Random.seed!(0x237)
    heads, kv_heads, dim, nkeys = 4, 2, 128, 65
    query = randn(Float32, dim*(heads+2kv_heads), 1)
    keys = randn(Float32, dim*kv_heads, nkeys)
    values = randn(Float32, dim*kv_heads, nkeys)
    dq, dk, dv = MtlArray(query), MtlArray(keys), MtlArray(values)
    out = MtlArray{Float32}(undef, dim*heads, 1)
    partial = MtlArray{Float32}(undef, dim+2, heads, 3)
    @test M.attention128_eligible(
        partial,
        dq,
        dk,
        dv,
        nkeys-1,
        1,
        128,
        dim,
        heads,
        kv_heads,
    )
    M.single_attention_gpu!(
        out,
        dq,
        dk,
        dv,
        nkeys-1,
        1,
        true,
        128,
        dim,
        heads,
        kv_heads;
        partial,
    )
    expected = similar(Array(out))
    P.grouped_attention!(
        expected,
        view(query, 1:(dim*heads), :),
        keys,
        values,
        nkeys-1,
        1,
        true,
        128,
        dim,
        heads,
        kv_heads,
    )
    @test Array(out) ≈ expected rtol=3.0f-5 atol=3.0f-5
end


@testset "Register-resident dual normalization" begin
    width = 1536
    for (zero_input, alias_input) in ((false, false), (true, false), (false, true)),
        residual_scale in (0.0f0, 0.2f0, 2.0f0)
        input = zero_input ? zeros(Float32, width, 1) :
                reshape(Float32.(sin.(1:width)), :, 1)
        residual = reshape(Float32.(cos.(1:width)), :, 1) .* residual_scale
        post = Float32.(sin.(1:width)) .+ 0.5f0
        pre = Float32.(cos.(1:width)) .+ 0.5f0
        expected = P.rms_norm(input, post, 1.0f-5) .+ residual
        device_input = MtlArray(input)
        output = alias_input ? device_input : MtlArray{Float32}(undef, width, 1)
        normalized = similar(output)
        @metal debug_level=M.decode_debug_level() threads=512 groups=1 M.register_dual_rms1536_kernel!(
            output, normalized, device_input, MtlArray(residual),
            MtlArray(post), MtlArray(pre), 1.0f-5,
        )
        @test Array(output) ≈ expected rtol=3.0f-5 atol=3.0f-5
        @test Array(normalized) ≈ P.rms_norm(expected, pre, 1.0f-5) rtol=3.0f-5 atol=3.0f-5
    end
end
