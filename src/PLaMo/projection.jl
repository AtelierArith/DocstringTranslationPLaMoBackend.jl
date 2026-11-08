# F32 is the compatibility path. Opt-in BF16 uses only packed original bits;
# it neither retains F32 projection shadows nor rounds F32 source checkpoints.
struct Projection
    dense::Union{Nothing,Matrix{Float32}}
    bf16::Union{Nothing,PackedBF16}
    function Projection(
        dense::Union{Nothing,Matrix{Float32}},
        bf16::Union{Nothing,PackedBF16},
    )
        isnothing(dense)==isnothing(bf16) &&
            throw(ArgumentError("exactly one projection storage is required"))
        new(dense, bf16)
    end
end
function Projection(weight::Matrix{Float32}, source_dtype::AbstractString, storage::Symbol)
    storage==:f32 || throw(
        ArgumentError(
            "weight_storage=:bf16 requires original BF16 projection tensors; F32 weights are not rounded",
        ),
    )
    Projection(weight, nothing)
end
function Projection(weight::Matrix{UInt16}, source_dtype::AbstractString, storage::Symbol)
    source_dtype=="BF16" ||
        throw(ArgumentError("UInt16 projection payload must have BF16 source dtype"))
    storage==:f32 && return Projection(widen_f32(weight), nothing)
    storage==:bf16 || throw(ArgumentError("weight_storage must be :f32 or :bf16"))
    Projection(nothing, PackedBF16(weight))
end

# Explicit row-major reference kernel for regression tests and diagnostic
# benchmarks. Production BF16 inference uses only packed_project above/below.
function bf16_rows!(output, weight, input, first, last)
    width = size(weight, 1)
    @inbounds for row = first:last
        value = 0.0f0
        @simd for k = 1:width
            widened = reinterpret(Float32, UInt32(weight[k, row]) << 16)
            value = muladd(widened, input[k], value)
        end
        output[row] = value
    end
    return output
end

function bf16_gemv!(
    output::AbstractVector{Float32},
    weight::Matrix{UInt16},
    input::AbstractVector{Float32},
)
    Base.require_one_based_indexing(output, weight, input)
    size(weight, 1) == length(input) && size(weight, 2) == length(output) ||
        throw(DimensionMismatch("BF16 projection shapes differ"))
    (Base.mightalias(output, input) || Base.mightalias(output, weight)) &&
        throw(ArgumentError("BF16 projection output must not alias input or weights"))
    workers = min(Threads.nthreads(:default), max(1, length(output) ÷ 128))
    if workers == 1
        bf16_rows!(output, weight, input, 1, length(output))
    else
        @sync for worker = 1:workers
            first = fld((worker-1)*length(output), workers)+1
            last = fld(worker*length(output), workers)
            Threads.@spawn bf16_rows!(output, weight, input, first, last)
        end
    end
    return output
end


function project(weight::Projection, input::AbstractMatrix{Float32})
    if !isnothing(weight.bf16)
        return packed_project(weight.bf16, input)
    end
    dense=weight.dense::Matrix{Float32}
    if size(input, 2)==1
        output=Vector{Float32}(undef, size(dense, 2))
        mul!(output, transpose(dense), @view(input[:, 1]))
        return reshape(output, :, 1)
    end
    transpose(dense)*input
end

# Caller-owned output and blocked scratch are independent from input/weights.
function project!(
    output::AbstractMatrix{Float32},
    weight::Projection,
    input::AbstractMatrix{Float32},
    blocked::Vector{Float32},
)
    Base.require_one_based_indexing(output, input)
    rows, width=isnothing(weight.bf16) ? reverse(size(weight.dense)) :
                (weight.bf16.output, weight.bf16.input)
    size(input, 1)==width && size(output)==(rows, size(input, 2)) ||
        throw(DimensionMismatch("projection shapes differ"))
    (
        Base.mightalias(output, input) ||
        Base.mightalias(output, blocked) ||
        Base.mightalias(input, blocked)
    ) && throw(ArgumentError("projection buffers must not alias"))
    if isnothing(weight.bf16)
        Base.mightalias(output, weight.dense) &&
            throw(ArgumentError("projection output must not alias weights"))
        if size(input, 2)==1
            mul!(@view(output[:, 1]), transpose(weight.dense), @view(input[:, 1]))
        else
            mul!(output, transpose(weight.dense), input)
        end
        return output
    end
    bits=weight.bf16::PackedBF16
    length(bits.values)==cld(rows, 16)*width*16 ||
        throw(DimensionMismatch("packed BF16 buffer length differs"))
    (Base.mightalias(output, bits.values) || Base.mightalias(blocked, bits.values)) &&
        throw(ArgumentError("projection buffers must not alias weights"))
    size(input, 2)==1 &&
        !isnothing(bits.dictionary) &&
        return dictionary_project!(output, bits.dictionary, input, blocked)
    tokens=size(input, 2)
    blocks=cld(rows, 16)
    length(blocked)>=blocks*tokens*16 ||
        throw(DimensionMismatch("packed projection scratch is too small"))
    tokens==0 && return output
    workers=min(Threads.nthreads(:default), blocks)
    @sync for worker = 1:workers
        first=fld((worker-1)*blocks, workers)
        last=fld(worker*blocks, workers)-1
        Threads.@spawn block_forward!(blocked, bits.values, input, width, first, last)
    end
    @inbounds for token = 1:tokens, block = 0:(blocks-1)
        count=min(16, rows-block*16)
        source=block*tokens*16+(token-1)*16
        @simd for row = 1:count
            output[block*16+row, token]=blocked[source+row]
        end
    end
    return output
end
