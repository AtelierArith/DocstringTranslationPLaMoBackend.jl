@testset "Reusable Metal token selection" begin
    for count in (1, 31, 1023, 1024, 1025, 107520)
        values = -Float32.(1:count)
        values[end] = 3
        partial = MtlArray{Tuple{Float32,Int32}}(undef, cld(count, 1024))
        logits = MtlArray(reshape(values, :, 1))
        @test M.gpu_token!(logits, partial) == argmax(values) - 1
        values[1] = 3 # First index wins ties, including across partitions.
        copyto!(logits, reshape(values, :, 1))
        @test M.gpu_token!(logits, partial) == 0
        fill!(values, -Inf32)
        copyto!(logits, reshape(values, :, 1))
        @test M.gpu_token!(logits, partial) == 0
        values[end] = NaN32
        copyto!(logits, reshape(values, :, 1))
        @test M.gpu_token!(logits, partial) == argmax(values) - 1
        values[1] = NaN32
        copyto!(logits, reshape(values, :, 1))
        @test M.gpu_token!(logits, partial) == 0
    end
end

@testset "Metal token selection total float ordering" begin
    for values in
        (Float32[-0.0, 0.0], Float32[0.0, -0.0], Float32[-Inf, Inf, Inf], Float32[NaN, NaN])
        partial = MtlArray{Tuple{Float32,Int32}}(undef, 1)
        @test M.gpu_token!(MtlArray(reshape(values, :, 1)), partial) == argmax(values) - 1
    end
end
