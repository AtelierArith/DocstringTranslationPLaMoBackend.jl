# Q8_0/Q4_0 reference quantization values in an internal, non-GGML layout.
# Scales are separate Float16 arrays; Q4 packs adjacent elements together.
struct QuantizedMatrix{B,D,S} <: AbstractMatrix{Float32}
    bits::Val{B}
    data::D
    scales::S
    width::Int
    columns::Int
end
Base.size(w::QuantizedMatrix) = (w.width, w.columns)
Metal.Adapt.adapt_structure(to, w::QuantizedMatrix) = QuantizedMatrix(
    w.bits,
    Metal.Adapt.adapt(to, w.data),
    Metal.Adapt.adapt(to, w.scales),
    w.width,
    w.columns,
)

function quantize_blocks(raw::AbstractMatrix{UInt16}, ::Val{bits}) where {bits}
    bits in (4, 8) || throw(ArgumentError("quantization must use 4 or 8 bits"))
    width, columns = size(raw)
    width % 32 == 0 ||
        throw(DimensionMismatch("quantized matrix width must be a multiple of 32"))
    padded_columns = 2cld(columns, 2)
    data = zeros(UInt8, width * bits ÷ 8, padded_columns)
    scales = zeros(Float16, width ÷ 32, padded_columns)
    @inbounds for column = 1:columns, block = 1:(width÷32)
        first = (block - 1) * 32 + 1
        amax = signed_max = 0.0f0
        for i = first:(first+31)
            value = reinterpret(Float32, UInt32(raw[i, column]) << 16)
            isfinite(value) || throw(ArgumentError("quantized weights must be finite"))
            if abs(value) > amax
                amax, signed_max = abs(value), value
            end
        end
        scale = bits == 8 ? amax / 127.0f0 : signed_max / -8.0f0
        half_scale = Float16(scale)
        isfinite(half_scale) || throw(ArgumentError("weight scale exceeds Float16 range"))
        scales[block, column] = half_scale
        # A stored zero scale makes the block zero. Avoid overflowing the
        # Float32 reciprocal for finite BF16 subnormal inputs.
        inverse = iszero(half_scale) ? 0.0f0 : inv(scale)
        for i = first:(first+31)
            value = reinterpret(Float32, UInt32(raw[i, column]) << 16)
            if bits == 8
                q = clamp(round(Int32, value * inverse, RoundNearestTiesAway), -127, 127)
                data[i, column] = reinterpret(UInt8, Int8(q))
            else
                q = clamp(trunc(Int32, value * inverse + 8.5f0), 0, 15)
                data[(i-1)÷2+1, column] |= UInt8(q << (((i - 1) % 2) * 4))
            end
        end
    end
    return data, scales
end

function quantize_matrix(raw, bits)
    data, scales = quantize_blocks(raw, bits)
    return QuantizedMatrix(bits, MtlArray(data), MtlArray(scales), size(raw)...)
end

@inline function Base.getindex(w::QuantizedMatrix{bits}, i::Int) where {bits}
    if bits == 8
        @inbounds value = Float32(reinterpret(Int8, w.data[i]))
    else
        @inbounds byte = w.data[(i-1)÷2+1]
        value = Float32((byte >> (((i - 1) % 2) * 4)) & 0x0f) - 8.0f0
    end
    @inbounds return value * Float32(w.scales[(i-1)÷32+1])
end
@inline Base.getindex(w::QuantizedMatrix, row::Int, column::Int) = w[row+(column-1)*w.width]

function dequantize_kernel!(output, weight, count)
    i = Int(thread_position_in_grid_1d())
    i > count && return
    @inbounds output[i] = weight[i]
    return
end
function dequantize_gpu(weight::QuantizedMatrix)
    output = MtlArray{Float32}(undef, size(weight))
    @metal threads=256 groups=cld(length(output), 256) dequantize_kernel!(
        output,
        weight,
        length(output),
    )
    return output
end

@inline function quantized_eight(raw, ::Val{8}, ::Val{chunk}) where {chunk}
    words = P.SIMD.shufflevector(raw, Val((2chunk, 2chunk, 2chunk+1, 2chunk+1)))
    even = (words >> Vec{4,UInt32}((0, 16, 0, 16))) & Vec{4,UInt32}(255)
    odd = (words >> Vec{4,UInt32}((8, 24, 8, 24))) & Vec{4,UInt32}(255)
    # Integer values in these ranges are represented exactly by the mantissa.
    even_f =
        reinterpret(
            Vec{4,Float32},
            (even ⊻ Vec{4,UInt32}(128)) | Vec{4,UInt32}(0x4b000000),
        ) - 8388736.0f0
    odd_f =
        reinterpret(
            Vec{4,Float32},
            (odd ⊻ Vec{4,UInt32}(128)) | Vec{4,UInt32}(0x4b000000),
        ) - 8388736.0f0
    return even_f, odd_f
end
@inline function quantized_eight(raw, ::Val{4}, ::Val{chunk}) where {chunk}
    words = Vec{4,UInt32}(raw[chunk+1])
    even = (words >> Vec{4,UInt32}((0, 8, 16, 24))) & Vec{4,UInt32}(15)
    odd = (words >> Vec{4,UInt32}((4, 12, 20, 28))) & Vec{4,UInt32}(15)
    return reinterpret(Vec{4,Float32}, even | Vec{4,UInt32}(0x4b000000)) - 8388616.0f0,
    reinterpret(Vec{4,Float32}, odd | Vec{4,UInt32}(0x4b000000)) - 8388616.0f0
end

function quantized_matvec_kernel!(
    output,
    data,
    scales,
    input,
    width,
    columns,
    bits::Val{B},
) where {B}
    lane = Int(thread_index_in_simdgroup())
    first_row = (Int(threadgroup_position_in_grid_1d()) - 1) * 2 + 1
    a0 = a1 = 0.0f0
    for k = ((lane-1)*32+1):1024:(width-31)
        if B == 8
            p0 = reinterpret(
                Core.LLVMPtr{UInt32,Metal.AS.Device},
                pointer(data, k + (first_row - 1) * width),
            )
            p1 = reinterpret(
                Core.LLVMPtr{UInt32,Metal.AS.Device},
                pointer(data, k + first_row * width),
            )
            raw0, raw1 = vload(Vec{8,UInt32}, p0), vload(Vec{8,UInt32}, p1)
        else
            p0 = reinterpret(
                Core.LLVMPtr{UInt32,Metal.AS.Device},
                pointer(data, (k - 1) ÷ 2 + 1 + (first_row - 1) * (width ÷ 2)),
            )
            p1 = reinterpret(
                Core.LLVMPtr{UInt32,Metal.AS.Device},
                pointer(data, (k - 1) ÷ 2 + 1 + first_row * (width ÷ 2)),
            )
            raw0, raw1 = vload(Vec{4,UInt32}, p0), vload(Vec{4,UInt32}, p1)
        end
        d0 = d1 = Vec{4,Float32}(0.0f0)
        Base.Cartesian.@nexprs 4 chunk -> begin
            xv = vload(Vec{8,Float32}, pointer(input, k + (chunk - 1) * 8))
            xe = P.SIMD.shufflevector(xv, Val((0, 2, 4, 6)))
            xo = P.SIMD.shufflevector(xv, Val((1, 3, 5, 7)))
            qe0, qo0 = quantized_eight(raw0, bits, Val(chunk - 1))
            qe1, qo1 = quantized_eight(raw1, bits, Val(chunk - 1))
            d0 = muladd(qe0, xe, d0)
            d0 = muladd(qo0, xo, d0)
            d1 = muladd(qe1, xe, d1)
            d1 = muladd(qo1, xo, d1)
        end
        @inbounds a0 += sum(d0) * Float32(scales[(k-1)÷32+1+(first_row-1)*(width÷32)])
        @inbounds a1 += sum(d1) * Float32(scales[(k-1)÷32+1+first_row*(width÷32)])
    end
    a0, a1 = simd_sum_f32(a0), simd_sum_f32(a1)
    if lane == 1
        @inbounds output[first_row] = a0
        first_row < columns && (@inbounds output[first_row+1] = a1)
    end
    return
end

function projection_gpu!(result, weight::QuantizedMatrix, input)
    if size(input, 2) == 1 && input isa MtlArray
        @metal threads=32 groups=cld(size(weight, 2), 2) quantized_matvec_kernel!(
            result,
            weight.data,
            weight.scales,
            input,
            size(weight, 1),
            size(weight, 2),
            weight.bits,
        )
    else
        # Low-memory prefill fallback: expand only the current projection.
        expanded = dequantize_gpu(weight)
        mul!(result, transpose(expanded), input)
    end
    return result
end
