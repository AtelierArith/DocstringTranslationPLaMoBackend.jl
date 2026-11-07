# Four output rows share each activation load. Unsigned 32-bit indices avoid
# expensive 64-bit address arithmetic; the host checks their range.
function q4_matvec_masked_kernel!(out, data, scales, x, width, cols)
    lane=UInt32(thread_index_in_simdgroup())
    base=(UInt32(threadgroup_position_in_grid_1d())-UInt32(1))*UInt32(4)+UInt32(1)
    a0=0.0f0
    a1=0.0f0
    a2=0.0f0
    a3=0.0f0
    for k = ((lane-UInt32(1))*UInt32(16)+UInt32(1)):UInt32(512):(width-UInt32(15))
        x0=vload(Vec{8,Float32}, pointer(x, k+UInt32(0)))
        x1=vload(Vec{8,Float32}, pointer(x, k+UInt32(8)))
        y0=Vec{4,Float32}((x0[1], x0[5], x1[1], x1[5]))
        y1=Vec{4,Float32}((x0[2], x0[6], x1[2], x1[6]))
        y2=Vec{4,Float32}((x0[3], x0[7], x1[3], x1[7]))
        y3=Vec{4,Float32}((x0[4], x0[8], x1[4], x1[8]))
        sumx=sum((y0+y1)+(y2+y3))
        y1*=Float32(0.0625)
        y2*=Float32(0.00390625)
        y3*=Float32(0.000244140625)
        p=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(
                data,
                (k-UInt32(1))÷UInt32(2)+UInt32(1)+(base+UInt32(0)-UInt32(1))*(
                    width÷UInt32(2)
                ),
            ),
        )
        raw=vload(Vec{4,UInt16}, p)
        d=Vec{4,Float32}(0.0f0)
        masked=raw&Vec{4,UInt16}(0x000f)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y0, d)
        masked=raw&Vec{4,UInt16}(0x00f0)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y1, d)
        masked=raw&Vec{4,UInt16}(0x0f00)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y2, d)
        masked=raw&Vec{4,UInt16}(0xf000)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y3, d)
        @inbounds scale=Float32(
            scales[(k-UInt32(1))÷UInt32(32)+UInt32(1)+(base+UInt32(0)-UInt32(
                1,
            ))*(width÷UInt32(32))],
        )
        a0+=(sum(d)-8.0f0*sumx)*scale
        p=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(
                data,
                (k-UInt32(1))÷UInt32(2)+UInt32(1)+(base+UInt32(1)-UInt32(1))*(
                    width÷UInt32(2)
                ),
            ),
        )
        raw=vload(Vec{4,UInt16}, p)
        d=Vec{4,Float32}(0.0f0)
        masked=raw&Vec{4,UInt16}(0x000f)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y0, d)
        masked=raw&Vec{4,UInt16}(0x00f0)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y1, d)
        masked=raw&Vec{4,UInt16}(0x0f00)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y2, d)
        masked=raw&Vec{4,UInt16}(0xf000)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y3, d)
        @inbounds scale=Float32(
            scales[(k-UInt32(1))÷UInt32(32)+UInt32(1)+(base+UInt32(1)-UInt32(
                1,
            ))*(width÷UInt32(32))],
        )
        a1+=(sum(d)-8.0f0*sumx)*scale
        p=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(
                data,
                (k-UInt32(1))÷UInt32(2)+UInt32(1)+(base+UInt32(2)-UInt32(1))*(
                    width÷UInt32(2)
                ),
            ),
        )
        raw=vload(Vec{4,UInt16}, p)
        d=Vec{4,Float32}(0.0f0)
        masked=raw&Vec{4,UInt16}(0x000f)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y0, d)
        masked=raw&Vec{4,UInt16}(0x00f0)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y1, d)
        masked=raw&Vec{4,UInt16}(0x0f00)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y2, d)
        masked=raw&Vec{4,UInt16}(0xf000)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y3, d)
        @inbounds scale=Float32(
            scales[(k-UInt32(1))÷UInt32(32)+UInt32(1)+(base+UInt32(2)-UInt32(
                1,
            ))*(width÷UInt32(32))],
        )
        a2+=(sum(d)-8.0f0*sumx)*scale
        p=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(
                data,
                (k-UInt32(1))÷UInt32(2)+UInt32(1)+(base+UInt32(3)-UInt32(1))*(
                    width÷UInt32(2)
                ),
            ),
        )
        raw=vload(Vec{4,UInt16}, p)
        d=Vec{4,Float32}(0.0f0)
        masked=raw&Vec{4,UInt16}(0x000f)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y0, d)
        masked=raw&Vec{4,UInt16}(0x00f0)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y1, d)
        masked=raw&Vec{4,UInt16}(0x0f00)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y2, d)
        masked=raw&Vec{4,UInt16}(0xf000)
        q=convert(Vec{4,Float32}, convert(Vec{4,Int32}, masked))
        d=muladd(q, y3, d)
        @inbounds scale=Float32(
            scales[(k-UInt32(1))÷UInt32(32)+UInt32(1)+(base+UInt32(3)-UInt32(
                1,
            ))*(width÷UInt32(32))],
        )
        a3+=(sum(d)-8.0f0*sumx)*scale
    end
    a0=simd_sum_f32(a0)
    a1=simd_sum_f32(a1)
    a2=simd_sum_f32(a2)
    a3=simd_sum_f32(a3)
    if lane==1
        @inbounds out[base+UInt32(0)]=a0
        @inbounds out[base+UInt32(1)]=a1
        @inbounds out[base+UInt32(2)]=a2
        @inbounds out[base+UInt32(3)]=a3
    end
    return
end
function projection_gpu!(result, weight::QuantizedMatrix{4}, input)
    width, columns = size(weight)
    if size(input, 2) == 1 &&
       input isa MtlArray{Float32} &&
       result isa MtlArray{Float32} &&
       width >= 32 &&
       columns > 0 &&
       columns % 4 == 0 &&
       length(input) == width &&
       length(result) == columns &&
       length(weight.data) <= typemax(UInt32) &&
       length(weight.scales) <= typemax(UInt32) &&
       width <= typemax(UInt32)
        @metal threads=32 groups=columns÷4 q4_matvec_masked_kernel!(
            result,
            weight.data,
            weight.scales,
            input,
            UInt32(width),
            UInt32(columns),
        )
        return result
    end
    return invoke(projection_gpu!, Tuple{Any,QuantizedMatrix,Any}, result, weight, input)
end

function u32_vector_dual_rms_kernel!(
    output,
    normalized,
    input,
    residual,
    post_weight,
    pre_weight,
    eps,
    ::Val{W},
    ::Val{nsg},
) where {nsg,W}
    width=UInt32(W)
    lane=UInt32(thread_index_in_simdgroup())
    tid=UInt32(thread_index_in_threadgroup())
    simd=UInt32(simdgroup_index_in_threadgroup())
    partial=MtlThreadGroupArray(Float32, nsg)
    square=Vec{4,Float32}(0.0f0)
    for i = ((tid-UInt32(1))*UInt32(4)+UInt32(1)):UInt32(128*nsg):(width-UInt32(3))
        xv=vload(Vec{4,Float32}, pointer(input, i))
        square=muladd(xv, xv, square)
    end
    total=simd_sum_f32(sum(square))
    if nsg>UInt32(1)
        lane==UInt32(1) && (@inbounds partial[simd]=total)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        total=lane<=nsg ? (@inbounds partial[lane]) : 0.0f0
        total=simd_sum_f32(total)
    end
    scale=inv(sqrt(simd_shuffle(total, UInt32(1))/width+eps))
    nsg>UInt32(1) && threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    square=Vec{4,Float32}(0.0f0)
    for i = ((tid-UInt32(1))*UInt32(4)+UInt32(1)):UInt32(128*nsg):(width-UInt32(3))
        xv=vload(Vec{4,Float32}, pointer(input, i))
        pw=vload(Vec{4,Float32}, pointer(post_weight, i))
        rv=vload(Vec{4,Float32}, pointer(residual, i))
        value=(xv*scale)*pw+rv
        P.SIMD.vstore(value, pointer(output, i))
        square=muladd(value, value, square)
    end
    total=simd_sum_f32(sum(square))
    if nsg>UInt32(1)
        lane==UInt32(1) && (@inbounds partial[simd]=total)
        threadgroup_barrier(Metal.MemoryFlagThreadGroup)
        total=lane<=nsg ? (@inbounds partial[lane]) : 0.0f0
        total=simd_sum_f32(total)
    end
    scale2=inv(sqrt(simd_shuffle(total, UInt32(1))/width+eps))
    for i = ((tid-UInt32(1))*UInt32(4)+UInt32(1)):UInt32(128*nsg):(width-UInt32(3))
        xv=vload(Vec{4,Float32}, pointer(output, i))
        pw=vload(Vec{4,Float32}, pointer(pre_weight, i))
        P.SIMD.vstore((xv*scale2)*pw, pointer(normalized, i))
    end
    return
end


function u32_first_rms_kernel!(out, input, weight, residual, eps, ::Val{W}) where {W}
    width=UInt32(W)
    tid=UInt32(thread_index_in_threadgroup())
    lane=UInt32(thread_index_in_simdgroup())
    simd=UInt32(simdgroup_index_in_threadgroup())
    partial=MtlThreadGroupArray(Float32, 8)
    square=Vec{4,Float32}(0.0f0)
    for i = ((tid-UInt32(1))*UInt32(4)+UInt32(1)):UInt32(1024):(width-UInt32(3))
        x=vload(Vec{4,Float32}, pointer(input, i))
        square=muladd(x, x, square)
    end
    total=simd_sum_f32(sum(square))
    lane==UInt32(1) && (@inbounds partial[simd]=total)
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    total=lane<=UInt32(8) ? (@inbounds partial[lane]) : 0.0f0
    total=simd_sum_f32(total)
    scale=inv(sqrt(simd_shuffle(total, 1)/Float32(width)+eps))
    for i = ((tid-UInt32(1))*UInt32(4)+UInt32(1)):UInt32(1024):(width-UInt32(3))
        x=vload(Vec{4,Float32}, pointer(input, i))
        w=vload(Vec{4,Float32}, pointer(weight, i))
        value=(x*scale)*w
        if !isnothing(residual)
            value+=vload(Vec{4,Float32}, pointer(residual, i))
        end
        P.SIMD.vstore(value, pointer(out, i))
    end
    return
end

function attention128_kernel!(
    output,
    query,
    keys,
    values,
    position,
    start,
    full,
    window,
    groups,
    nkeys,
    heads,
    ::Val{split},
) where {split}
    lane=UInt32(thread_index_in_simdgroup())
    simd=UInt32(simdgroup_index_in_threadgroup())
    tid=UInt32(thread_index_in_threadgroup())
    group=UInt32(threadgroup_position_in_grid_1d())-UInt32(1)
    head=split ? group%heads : group
    part=split ? group÷heads : UInt32(0)
    if split
        nkeys=min(UInt32(32), nkeys-part*UInt32(32))
        start+=part*UInt32(32)
    end
    kv=head÷groups
    kstride=UInt32(size(keys, 1))
    vstride=UInt32(size(values, 1))
    qbase=head*UInt32(128)+lane
    @inbounds begin
        q0=query[qbase]
        q1=query[qbase+UInt32(32)]
        q2=query[qbase+UInt32(64)]
        q3=query[qbase+UInt32(96)]
    end
    scores=MtlThreadGroupArray(Float32, 32)
    for key = simd:UInt32(4):nkeys
        kp=start+key-UInt32(1)
        idx=(kp-UInt32(1))*kstride+kv*UInt32(128)+lane
        @inbounds begin
            d=muladd(q0, keys[idx], 0.0f0)
            d=muladd(q1, keys[idx+UInt32(32)], d)
            d=muladd(q2, keys[idx+UInt32(64)], d)
            d=muladd(q3, keys[idx+UInt32(96)], d)
        end
        d=simd_sum_f32(d)
        if lane==UInt32(1)
            @inbounds scores[key]=!full && kp+window<=position+UInt32(1) ? -Inf32 :
                                  d/sqrt(128.0f0)
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if simd==UInt32(1)
        value=lane<=nkeys ? (@inbounds scores[lane]) : -Inf32
        maximum_value=simd_shuffle(simd_max_f32(value), 1)
        probability=lane<=nkeys && maximum_value != -Inf32 ? exp(value-maximum_value) :
                    0.0f0
        total=simd_shuffle(simd_sum_f32(probability), 1)
        if lane<=nkeys
            @inbounds scores[lane]=split ? probability : probability/total
        end
        if split && lane==UInt32(1)
            @inbounds output[UInt32(129)+head*UInt32(130)+part*heads*UInt32(130)]=maximum_value
            @inbounds output[UInt32(130)+head*UInt32(130)+part*heads*UInt32(130)]=total
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    result=0.0f0
    for key = UInt32(1):nkeys
        idx=(start+key-UInt32(2))*vstride+kv*UInt32(128)+tid
        @inbounds result=muladd(scores[key], values[idx], result)
    end
    if split
        @inbounds output[tid+head*UInt32(130)+part*heads*UInt32(130)]=result
    else
        @inbounds output[head*UInt32(128)+tid]=result
    end
    return
end

# Raw SIMD loads require dense, one-token Float32 storage.
function vector_norm_eligible(width, arrays...)
    return 128 <= width <= typemax(UInt32) - 1024 &&
           width % 128 == 0 &&
           all(x -> x isa MtlArray{Float32} && length(x) == width, arrays)
end

function attention128_eligible(
    storage,
    query,
    keys,
    values,
    position,
    start,
    window,
    dim,
    heads,
    kv_heads,
)
    return dim == 128 &&
           heads > 0 &&
           kv_heads > 0 &&
           heads % kv_heads == 0 &&
           1 <= start <= position + 1 < typemax(UInt32) &&
           0 <= window < typemax(UInt32) - position - 1 &&
           size(keys, 1) == 128 * kv_heads &&
           size(values, 1) == 128 * kv_heads &&
           size(query, 1) >= 128 * heads &&
           size(query, 2) == 1 &&
           (
               (
                   ndims(storage) == 3 &&
                   size(storage, 1) == 130 &&
                   size(storage, 2) == heads
               ) || (ndims(storage) <= 2 && length(storage) == 128 * heads)
           ) &&
           all(
               x -> x isa MtlArray{Float32} && length(x) <= typemax(UInt32),
               (storage, query, keys, values),
           )
end

function u32_rotate_qk_kernel!(
    qkv,
    keys,
    values,
    qnorm,
    knorm,
    eps,
    cosines,
    sines,
    position,
    ::Val{W},
    ::Val{Q},
    ::Val{K},
    ::Val{C},
) where {W,Q,K,C}
    position, width, qheads, kheads, count=UInt32(position),
    UInt32(W),
    UInt32(Q),
    UInt32(K),
    UInt32(C)
    qstride=UInt32(size(qkv, 1))
    kstride=isnothing(keys) ? UInt32(0) : UInt32(size(keys, 1))
    vstride=isnothing(values) ? UInt32(0) : UInt32(size(values, 1))
    cstride=UInt32(size(cosines, 1))
    sstride=UInt32(size(sines, 1))
    lane = UInt32(thread_index_in_simdgroup())
    group =
        (UInt32(threadgroup_position_in_grid_1d()) - UInt32(1)) * UInt32(4) +
        UInt32(simdgroup_index_in_threadgroup())
    heads = qheads + kheads
    group > heads * count && return
    head = C == 1 ? group - UInt32(1) : (group - UInt32(1)) % heads
    token = C == 1 ? UInt32(1) : (group - UInt32(1)) ÷ heads + UInt32(1)
    base = head * width
    square = 0.0f0
    for i = lane:UInt32(32):width
        @inbounds square = muladd(
            qkv[base+i+(token-UInt32(1))*qstride],
            qkv[base+i+(token-UInt32(1))*qstride],
            square,
        )
    end
    square = simd_sum_f32(square)
    scale = inv(sqrt(simd_shuffle(square, UInt32(1)) / width + eps))
    half = width ÷ UInt32(2)
    for i = lane:UInt32(32):half
        @inbounds begin
            n1 = head < qheads ? qnorm[i] : knorm[i]
            n2 = head < qheads ? qnorm[half+i] : knorm[half+i]
            a = qkv[base+i+(token-UInt32(1))*qstride] * scale * n1
            b = qkv[base+half+i+(token-UInt32(1))*qstride] * scale * n2
            co = cosines[i+(position+token-UInt32(1))*cstride]
            si = sines[i+(position+token-UInt32(1))*sstride]
            first_value = a * co - b * si
            second_value = b * co + a * si
            qkv[base+i+(token-UInt32(1))*qstride] = first_value
            qkv[base+half+i+(token-UInt32(1))*qstride] = second_value
            if head >= qheads && !isnothing(keys)
                kvbase = (head - qheads) * width
                valuebase = (qheads + kheads) * width + kvbase
                keys[kvbase+i+(position+token-UInt32(1))*kstride] = first_value
                keys[kvbase+half+i+(position+token-UInt32(1))*kstride] = second_value
                values[kvbase+i+(position+token-UInt32(1))*vstride] =
                    qkv[valuebase+i+(token-UInt32(1))*qstride]
                values[kvbase+half+i+(position+token-UInt32(1))*vstride] =
                    qkv[valuebase+half+i+(token-UInt32(1))*qstride]
            end
        end
    end
    return
end


function u32_merge_attention_kernel!(output, partial, dim, heads, parts)
    dim, heads, parts = UInt32(dim), UInt32(heads), UInt32(parts)
    tid = UInt32(thread_position_in_grid_1d())
    tid > dim * heads && return
    feature = (tid - UInt32(1)) % dim + UInt32(1)
    head = (tid - UInt32(1)) ÷ dim + UInt32(1)
    maximum_value = -Inf32
    for part = UInt32(1):parts
        @inbounds maximum_value = max(
            maximum_value,
            partial[dim+UInt32(1)+(head-UInt32(1))*(dim+UInt32(2))+(part-UInt32(1))*heads*(dim+UInt32(
                2,
            ))],
        )
    end
    numerator = denominator = 0.0f0
    for part = UInt32(1):parts
        @inbounds begin
            factor = exp(
                partial[dim+UInt32(1)+(head-UInt32(1))*(dim+UInt32(2))+(part-UInt32(1))*heads*(dim+UInt32(2))] - maximum_value,
            )
            numerator = muladd(
                factor,
                partial[feature+(head-UInt32(1))*(dim+UInt32(2))+(part-UInt32(1))*heads*(dim+UInt32(
                    2,
                ))],
                numerator,
            )
            denominator = muladd(
                factor,
                partial[dim+UInt32(2)+(head-UInt32(1))*(dim+UInt32(2))+(part-UInt32(1))*heads*(dim+UInt32(2))],
                denominator,
            )
        end
    end
    @inbounds output[tid] = numerator / denominator
    return
end


function rotate32_eligible(
    qkv,
    qnorm,
    knorm,
    cosines,
    sines,
    position,
    qheads,
    kheads,
    keys,
    values,
)
    width = length(qnorm)
    arrays = (qkv, qnorm, knorm, cosines, sines)
    return width == 128 &&
           size(qkv, 2) == 1 &&
           length(knorm) == width &&
           qheads > 0 &&
           kheads > 0 &&
           0 <= position <= typemax(UInt32) - size(qkv, 2) &&
           (qheads + kheads) * size(qkv, 2) <= typemax(UInt32) - 4 &&
           all(x -> x isa MtlArray{Float32} && length(x) <= typemax(UInt32), arrays) &&
           (
               isnothing(keys) || (
                   keys isa MtlArray{Float32} &&
                   values isa MtlArray{Float32} &&
                   length(keys) <= typemax(UInt32) &&
                   length(values) <= typemax(UInt32)
               )
           )
end
