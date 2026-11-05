# Original BF16 values packed by sixteen output rows. The quarter swap lets
# one 256-bit load reconstruct two F32 vectors through lane-local unpack.
struct PackedBF16
    input::Int
    output::Int
    values::Vector{UInt16}
    dictionary::Union{Nothing,DictionaryBF16}
    function PackedBF16(input::Int, output::Int, values::Vector{UInt16})
        input>0 && output>0 ||
            throw(ArgumentError("projection dimensions must be positive"))
        expected=Base.checked_mul(Base.checked_mul(cld(output, 16), input), 16)
        length(values)==expected ||
            throw(DimensionMismatch("packed BF16 buffer length differs"))
        dictionary=length(values)*2>=16*1024^2 && has_byte_shuffle() ?
                   DictionaryBF16(values, input, output) : nothing
        new(input, output, values, dictionary)
    end
end

function pack16(bits::Matrix{UInt16})
    input, output=size(bits)
    blocks=cld(output, 16)
    packed=zeros(UInt16, blocks*input*16)
    Threads.@threads :dynamic for block = 0:(blocks-1)
        @inbounds for k = 1:input, row = 1:16
            source=block*16+(row<=4 ? row : row<=8 ? row+4 : row<=12 ? row-4 : row)
            if source<=output
                packed[block*input*16+(k-1)*16+row]=bits[k, source]
            end
        end
    end
    packed
end
Base.@propagate_inbounds function load_bf16x16(packed, index)
    raw=vload(Vec{16,UInt16}, packed, index)
    zero=Vec{16,UInt16}(0)
    lo=ENDIAN_BOM==0x04030201 ?
       shufflevector(
        zero,
        raw,
        Val((0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27)),
    ) :
       shufflevector(
        raw,
        zero,
        Val((0, 16, 1, 17, 2, 18, 3, 19, 8, 24, 9, 25, 10, 26, 11, 27)),
    )
    hi=ENDIAN_BOM==0x04030201 ?
       shufflevector(
        zero,
        raw,
        Val((4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31)),
    ) :
       shufflevector(
        raw,
        zero,
        Val((4, 20, 5, 21, 6, 22, 7, 23, 12, 28, 13, 29, 14, 30, 15, 31)),
    )
    return reinterpret(Vec{8,Float32}, lo), reinterpret(Vec{8,Float32}, hi)
end
function tile!(output, packed, x, input, block, token, ::Val{COUNT}) where {COUNT}
    z=Vec{8,Float32}(0.0f0)
    a0=a1=b0=b1=c0=c1=d0=d1=z
    @inbounds for k = 1:input
        index=block*input*16+(k-1)*16+1
        w0, w1=load_bf16x16(packed, index)
        a=Vec{8,Float32}(x[k, token])
        a0=muladd(w0, a, a0)
        a1=muladd(w1, a, a1)
        if COUNT>1
            b=Vec{8,Float32}(x[k, token+1])
            b0=muladd(w0, b, b0)
            b1=muladd(w1, b, b1)
        end
        if COUNT>2
            c=Vec{8,Float32}(x[k, token+2])
            c0=muladd(w0, c, c0)
            c1=muladd(w1, c, c1)
        end
        if COUNT>3
            d=Vec{8,Float32}(x[k, token+3])
            d0=muladd(w0, d, d0)
            d1=muladd(w1, d, d1)
        end
    end
    offset=block*size(x, 2)*16+(token-1)*16+1
    @inbounds begin
        vstore(a0, output, offset)
        vstore(a1, output, offset+8)
        if COUNT>1
            vstore(b0, output, offset+16)
            vstore(b1, output, offset+24)
        end
        if COUNT>2
            vstore(c0, output, offset+32)
            vstore(c1, output, offset+40)
        end
        if COUNT>3
            vstore(d0, output, offset+48)
            vstore(d1, output, offset+56)
        end
    end
    nothing
end
# Decode-only unroll: eight independent F32 accumulators hide FMA latency.
function tile!(output, packed, x, input, block, token, ::Val{1})
    z=Vec{8,Float32}(0.0f0)
    a0=a1=b0=b1=c0=c1=d0=d1=z
    k=1
    @inbounds while k+3<=input
        index=block*input*16+(k-1)*16+1
        av=Vec{8,Float32}(x[k, token])
        bv=Vec{8,Float32}(x[k+1, token])
        cv=Vec{8,Float32}(x[k+2, token])
        dv=Vec{8,Float32}(x[k+3, token])
        aw0, aw1=load_bf16x16(packed, index)
        bw0, bw1=load_bf16x16(packed, index+16)
        cw0, cw1=load_bf16x16(packed, index+32)
        dw0, dw1=load_bf16x16(packed, index+48)
        a0=muladd(aw0, av, a0)
        a1=muladd(aw1, av, a1)
        b0=muladd(bw0, bv, b0)
        b1=muladd(bw1, bv, b1)
        c0=muladd(cw0, cv, c0)
        c1=muladd(cw1, cv, c1)
        d0=muladd(dw0, dv, d0)
        d1=muladd(dw1, dv, d1)
        k+=4
    end
    lo=(a0+b0)+(c0+d0)
    hi=(a1+b1)+(c1+d1)
    @inbounds for tail = k:input
        index=block*input*16+(tail-1)*16+1
        value=Vec{8,Float32}(x[tail, token])
        w0, w1=load_bf16x16(packed, index)
        lo=muladd(w0, value, lo)
        hi=muladd(w1, value, hi)
    end
    offset=block*size(x, 2)*16+(token-1)*16+1
    @inbounds begin
        vstore(lo, output, offset)
        vstore(hi, output, offset+8)
    end
    nothing
end
function block_forward!(out, packed, x, input, first, last)
    tokens=size(x, 2)
    for block = first:last
        token=1
        while token+3<=tokens
            tile!(out, packed, x, input, block, token, Val(4))
            token+=4
        end
        remaining=tokens-token+1
        if remaining==1
            tile!(out, packed, x, input, block, token, Val(1))
        elseif remaining==2
            tile!(out, packed, x, input, block, token, Val(2))
        elseif remaining==3
            tile!(out, packed, x, input, block, token, Val(3))
        end
    end
    nothing
end
function packed_forward(packed, input, rows, x::AbstractMatrix{Float32})
    size(x, 1)==input || throw(DimensionMismatch())
    blocks=cld(rows, 16)
    tokens=size(x, 2)
    tokens==0 && return Matrix{Float32}(undef, rows, 0)
    blocked=Vector{Float32}(undef, blocks*tokens*16)
    workers=min(Threads.nthreads(:default), blocks)
    @sync for worker = 1:workers
        first=fld((worker-1)*blocks, workers)
        last=fld(worker*blocks, workers)-1
        Threads.@spawn block_forward!(blocked, packed, x, input, first, last)
    end
    if tokens==1
        resize!(blocked, rows)
        return reshape(blocked, rows, 1)
    end
    output=Matrix{Float32}(undef, rows, tokens)
    for token = 1:tokens, block = 0:(blocks-1)
        count=min(16, rows-block*16)
        copyto!(
            output,
            (token-1)*rows+block*16+1,
            blocked,
            block*tokens*16+(token-1)*16+1,
            count,
        )
    end
    output
end

function PackedBF16(bits::Matrix{UInt16})
    input, output=size(bits)
    input>0 && output>0 || throw(ArgumentError("projection dimensions must be positive"))
    PackedBF16(input, output, pack16(bits))
end
function packed_project(weight::PackedBF16, input::AbstractMatrix{Float32})
    Base.require_one_based_indexing(input)
    length(weight.values)==cld(weight.output, 16)*weight.input*16 ||
        throw(DimensionMismatch("packed BF16 buffer length differs"))
    size(input, 2)==1 &&
        !isnothing(weight.dictionary) &&
        return dictionary_project(weight.dictionary, input)
    packed_forward(weight.values, weight.input, weight.output, input)
end
function bf16_source_bits(weight::PackedBF16)
    length(weight.values)==cld(weight.output, 16)*weight.input*16 ||
        throw(DimensionMismatch("packed BF16 buffer length differs"))
    source=Matrix{UInt16}(undef, weight.input, weight.output)
    @inbounds for row = 1:weight.output, feature = 1:weight.input
        block, slot=divrem(row-1, 16)
        slot+=1
        slot=slot<=4 ? slot : slot<=8 ? slot+4 : slot<=12 ? slot-4 : slot
        source[feature, row]=weight.values[block*weight.input*16+(feature-1)*16+slot]
    end
    source
end
