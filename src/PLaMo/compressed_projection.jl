# Lossless high-byte dictionaries retain every original BF16 source bit.
# Packed original weights remain available for prompt projections.
struct DictionaryBlock
    offset::Int
    length::Int
    raw::Bool
end
struct DictionaryBF16
    input::Int
    output::Int
    per_row::Int
    data::Vector{UInt8}
    blocks::Vector{DictionaryBlock}
    expected_bytes::Int
    function DictionaryBF16(
        input::Int,
        output::Int,
        per_row::Int,
        data::Vector{UInt8},
        blocks::Vector{DictionaryBlock},
    )
        input>0 && output>0 && per_row==cld(input, 64) ||
            throw(ArgumentError("invalid dictionary dimensions"))
        length(blocks)==cld(output, 16)*per_row ||
            throw(DimensionMismatch("dictionary block count differs"))
        offset=1
        for tile = 0:(cld(output, 16)-1), part = 0:(per_row-1)
            block=blocks[tile*per_row+part+1]
            count=min(64, input-part*64)
            block.offset==offset && block.length==count ||
                throw(ArgumentError("invalid dictionary block offsets"))
            offset+=block.raw ? count*32 : 16+count*24
        end
        offset==length(data)+1 ||
            throw(DimensionMismatch("dictionary payload length differs"))
        new(input, output, per_row, data, blocks, length(data))
    end
end
function DictionaryBF16(values::Vector{UInt16}, width::Int, rows::Int)
    width>0 && rows>0 ||
        throw(ArgumentError("dictionary projection dimensions must be positive"))
    length(values)==cld(rows, 16)*width*16 ||
        throw(DimensionMismatch("packed source length differs"))
    data=UInt8[]
    sizehint!(data, length(values)*2)
    blocks=DictionaryBlock[]
    sizehint!(blocks, cld(rows, 16)*cld(width, 64))
    mapping=fill(UInt8(0xff), 256)
    table=zeros(UInt8, 16)
    @inbounds for tile = 0:(cld(rows, 16)-1), first = 0:64:(width-1)
        count=min(64, width-first)
        base=tile*width*16+first*16
        fill!(mapping, 0xff)
        fill!(table, 0)
        entries=0
        for index = 1:(count*16)
            high=Int(values[base+index]>>8)+1
            if mapping[high]==0xff
                entries+=1
                entries>16 && break
                mapping[high]=UInt8(entries-1)
                table[entries]=UInt8(high-1)
            end
        end
        raw=entries>16
        push!(blocks, DictionaryBlock(length(data)+1, count, raw))
        if raw
            for index = 1:(count*16)
                word=values[base+index]
                push!(data, UInt8(word&0xff), UInt8(word>>8))
            end
        else
            append!(data, table)
            for feature = 0:(count-1)
                for lane = 1:16
                    push!(data, UInt8(values[base+feature*16+lane]&0xff))
                end
                for lane = 1:2:16
                    a=Int(values[base+feature*16+lane]>>8)+1
                    b=Int(values[base+feature*16+lane+1]>>8)+1
                    push!(data, mapping[a]|(mapping[b]<<4))
                end
            end
        end
    end
    sizehint!(data, length(data); shrink = true)
    DictionaryBF16(width, rows, cld(width, 64), data, blocks)
end
function dictionary_word(c::DictionaryBF16, tile, feature, lane)
    part, local_feature=divrem(feature, 64)
    block=c.blocks[tile*c.per_row+part+1]
    if block.raw
        offset=block.offset+(local_feature*16+lane)*2
        return UInt16(c.data[offset])|(UInt16(c.data[offset+1])<<8)
    end
    offset=block.offset+16+local_feature*24
    code=c.data[offset+16+lane÷2]
    index=iseven(lane) ? code&0x0f : code>>4
    UInt16(c.data[offset+lane])|(UInt16(c.data[block.offset+Int(index)])<<8)
end
has_byte_shuffle() =
    Sys.ARCH==:x86_64 &&
    Base.BinaryPlatforms.CPUID.test_cpu_feature(Base.BinaryPlatforms.CPUID.JL_X86_ssse3)
@inline function byte_lookup(table::Vec{16,UInt8}, index::Vec{16,UInt8}, ::Val{true})
    result=Base.llvmcall(
        (
            """
declare <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8>, <16 x i8>)
define <16 x i8> @entry(<16 x i8> %table, <16 x i8> %index) alwaysinline {
    %value = call <16 x i8> @llvm.x86.ssse3.pshuf.b.128(<16 x i8> %table, <16 x i8> %index)
    ret <16 x i8> %value
}
""",
            "entry",
        ),
        NTuple{16,VecElement{UInt8}},
        Tuple{NTuple{16,VecElement{UInt8}},NTuple{16,VecElement{UInt8}}},
        table.data,
        index.data,
    )
    Vec{16,UInt8}(result)
end
@inline function byte_lookup(table::Vec{16,UInt8}, index::Vec{16,UInt8}, ::Val{false})
    Vec(ntuple(i->table[Int(index[i])+1], Val(16)))
end
Base.@propagate_inbounds function dictionary_raw(data, block, table, feature, fast)
    if block.raw
        offset=block.offset+feature*32
        GC.@preserve data begin
            raw=vload(Vec{16,UInt16}, Ptr{UInt16}(pointer(data, offset)))
        end
        return ENDIAN_BOM==0x04030201 ? raw : (raw<<8)|(raw>>8)
    end
    offset=block.offset+16+feature*24
    code=vload(Vec{8,UInt8}, data, offset+16)
    low_index=code&Vec{8,UInt8}(0x0f)
    high_index=code>>4
    index=shufflevector(
        low_index,
        high_index,
        Val((0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15)),
    )
    high=byte_lookup(table, index, fast)
    low=vload(Vec{16,UInt8}, data, offset)
    mask=Val((
        0,
        16,
        1,
        17,
        2,
        18,
        3,
        19,
        4,
        20,
        5,
        21,
        6,
        22,
        7,
        23,
        8,
        24,
        9,
        25,
        10,
        26,
        11,
        27,
        12,
        28,
        13,
        29,
        14,
        30,
        15,
        31,
    ))
    raw=ENDIAN_BOM==0x04030201 ? shufflevector(low, high, mask) :
        shufflevector(high, low, mask)
    reinterpret(Vec{16,UInt16}, raw)
end
@inline function dictionary_expand(raw)
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
    reinterpret(Vec{8,Float32}, lo), reinterpret(Vec{8,Float32}, hi)
end
function dictionary_tile!(output, c, x, tile, fast)
    z=Vec{8,Float32}(0.0f0)
    a0=a1=b0=b1=c0=c1=d0=d1=z
    global_feature=0
    @inbounds for part = 1:c.per_row
        block=c.blocks[tile*c.per_row+part]
        table=block.raw ? Vec{16,UInt8}(0) : vload(Vec{16,UInt8}, c.data, block.offset)
        feature=0
        while feature+4<=block.length
            w0, w1=dictionary_expand(dictionary_raw(c.data, block, table, feature, fast))
            value=Vec{8,Float32}(x[global_feature+feature+1, 1])
            a0=muladd(w0, value, a0)
            a1=muladd(w1, value, a1)
            w0, w1=dictionary_expand(dictionary_raw(c.data, block, table, feature+1, fast))
            value=Vec{8,Float32}(x[global_feature+feature+2, 1])
            b0=muladd(w0, value, b0)
            b1=muladd(w1, value, b1)
            w0, w1=dictionary_expand(dictionary_raw(c.data, block, table, feature+2, fast))
            value=Vec{8,Float32}(x[global_feature+feature+3, 1])
            c0=muladd(w0, value, c0)
            c1=muladd(w1, value, c1)
            w0, w1=dictionary_expand(dictionary_raw(c.data, block, table, feature+3, fast))
            value=Vec{8,Float32}(x[global_feature+feature+4, 1])
            d0=muladd(w0, value, d0)
            d1=muladd(w1, value, d1)
            feature+=4
        end
        global_feature+=block.length
    end
    lo=(a0+b0)+(c0+d0)
    hi=(a1+b1)+(c1+d1)
    if c.input%4!=0
        block=c.blocks[(tile+1)*c.per_row]
        table=block.raw ? Vec{16,UInt8}(0) : vload(Vec{16,UInt8}, c.data, block.offset)
        @inbounds for feature = (fld(block.length, 4)*4):(block.length-1)
            w0, w1=dictionary_expand(dictionary_raw(c.data, block, table, feature, fast))
            value=Vec{8,Float32}(x[c.input-block.length+feature+1, 1])
            lo=muladd(w0, value, lo)
            hi=muladd(w1, value, hi)
        end
    end
    @inbounds begin
        vstore(lo, output, tile*16+1)
        vstore(hi, output, tile*16+9)
    end
    nothing
end
function dictionary_run!(output, c, x, fast)
    tiles=cld(c.output, 16)
    workers=min(Threads.nthreads(:default), tiles)
    @sync for worker = 1:workers
        first=fld((worker-1)*tiles, workers)
        last=fld(worker*tiles, workers)-1
        Threads.@spawn for tile = first:last
            dictionary_tile!(output, c, x, tile, fast)
        end
    end
    output
end
function dictionary_parallel(c, x, fast)
    output=Vector{Float32}(undef, cld(c.output, 16)*16)
    dictionary_run!(output, c, x, fast)
    resize!(output, c.output)
    reshape(output, :, 1)
end
function dictionary_project(c, x; fast = has_byte_shuffle())
    Base.require_one_based_indexing(x)
    size(x)==(c.input, 1) || throw(DimensionMismatch())
    check_dictionary(c)
    fast && !has_byte_shuffle() && throw(ArgumentError("SSSE3 is unavailable"))
    fast ? dictionary_parallel(c, x, Val(true)) : dictionary_parallel(c, x, Val(false))
end

function dictionary_project!(output, c, x, blocked; fast = has_byte_shuffle())
    Base.require_one_based_indexing(output, x)
    size(x)==(c.input, 1) && size(output)==(c.output, 1) ||
        throw(DimensionMismatch("dictionary projection shapes differ"))
    length(blocked)>=cld(c.output, 16)*16 ||
        throw(DimensionMismatch("dictionary scratch is too small"))
    check_dictionary(c)
    (
        Base.mightalias(output, x) ||
        Base.mightalias(output, blocked) ||
        Base.mightalias(x, blocked) ||
        Base.mightalias(output, c.data) ||
        Base.mightalias(blocked, c.data)
    ) && throw(ArgumentError("dictionary projection buffers must not alias"))
    fast && !has_byte_shuffle() && throw(ArgumentError("SSSE3 is unavailable"))
    fast ? dictionary_run!(blocked, c, x, Val(true)) :
    dictionary_run!(blocked, c, x, Val(false))
    @inbounds @simd for row = 1:c.output
        output[row, 1]=blocked[row]
    end
    output
end

function check_dictionary(c::DictionaryBF16)
    length(c.data)==c.expected_bytes && length(c.blocks)==cld(c.output, 16)*c.per_row ||
        throw(DimensionMismatch("dictionary buffers have been resized"))
    nothing
end
