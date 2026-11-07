module DocstringTranslationPLaMoBackendMetalExt

using DocstringTranslationPLaMoBackend
using Metal
using LinearAlgebra


# Apple GPU family 7 introduced native SIMD reductions. The family is a GPU
# compile-time constant, so older targets retain only the shuffle fallback.
# Only lane 1 is guaranteed to contain the reduction on the fallback path.
@inline function simd_sum_f32(x::Float32)
    if Metal.apple_family() >= 7
        return ccall("extern air.simd_sum.f32", llvmcall, Float32, (Float32,), x)
    end
    for offset in (16, 8, 4, 2, 1)
        x += simd_shuffle_down(x, offset)
    end
    return x
end
@inline function simd_max_f32(x::Float32)
    if Metal.apple_family() >= 7
        return ccall("extern air.simd_max.f32", llvmcall, Float32, (Float32,), x)
    end
    for offset in (16, 8, 4, 2, 1)
        x = max(x, simd_shuffle_down(x, offset))
    end
    return x
end

const B = DocstringTranslationPLaMoBackend
const P = B.PLaMo

function rms_kernel!(output, input, weight, residual, eps, width, count)
    lane = Int(thread_index_in_simdgroup())
    token =
        (Int(threadgroup_position_in_grid_1d()) - 1) * 4 +
        Int(simdgroup_index_in_threadgroup())
    token > count && return
    square = 0.0f0
    for i = lane:32:width
        @inbounds square = muladd(input[i, token], input[i, token], square)
    end
    square = simd_sum_f32(square)
    scale = inv(sqrt(simd_shuffle(square, 1) / width + eps))
    for i = lane:32:width
        @inbounds begin
            value = (input[i, token] * scale) * weight[i]
            output[i, token] = isnothing(residual) ? value : value + residual[i, token]
        end
    end
    return
end

function rms_norm_gpu!(output, x, weight, eps; residual = nothing)
    if size(x, 2) == 1 &&
       vector_norm_eligible(size(x, 1), output, x, weight) &&
       (isnothing(residual) || vector_norm_eligible(size(x, 1), residual))
        @metal threads=256 groups=1 u32_first_rms_kernel!(
            output,
            x,
            weight,
            residual,
            eps,
            Val(size(x, 1)),
        )
        return output
    end
    @metal threads=128 groups=cld(size(x, 2), 4) rms_kernel!(
        output,
        x,
        weight,
        residual,
        eps,
        size(x, 1),
        size(x, 2),
    )
    return output
end

function rms_norm_gpu(x, weight, eps; residual = nothing)
    output = MtlArray{Float32}(undef, size(x))
    return rms_norm_gpu!(output, x, weight, eps; residual)
end

# One threadgroup normalizes a residual branch, adds the skip connection,
# and prepares the normalized input for the next projection.
function dual_rms_kernel!(
    output,
    normalized,
    input,
    residual,
    post_weight,
    pre_weight,
    eps,
    width,
)
    token = Int(threadgroup_position_in_grid_1d())
    lane = Int(thread_index_in_simdgroup())
    tid = Int(thread_index_in_threadgroup())
    simd = Int(simdgroup_index_in_threadgroup())
    partial = MtlThreadGroupArray(Float32, 8)
    square = 0.0f0
    for i = tid:256:width
        @inbounds square = muladd(input[i, token], input[i, token], square)
    end
    square = simd_sum_f32(square)
    lane == 1 && (@inbounds partial[simd] = square)
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    square = lane <= 8 ? (@inbounds partial[lane]) : 0.0f0
    square = simd_sum_f32(square)
    scale = inv(sqrt(simd_shuffle(square, 1) / width + eps))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    square2 = 0.0f0
    for i = tid:256:width
        @inbounds begin
            value = (input[i, token] * scale) * post_weight[i] + residual[i, token]
            output[i, token] = value
            square2 = muladd(value, value, square2)
        end
    end
    square2 = simd_sum_f32(square2)
    lane == 1 && (@inbounds partial[simd] = square2)
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    square2 = lane <= 8 ? (@inbounds partial[lane]) : 0.0f0
    square2 = simd_sum_f32(square2)
    scale2 = inv(sqrt(simd_shuffle(square2, 1) / width + eps))
    for i = tid:256:width
        @inbounds normalized[i, token] = (output[i, token] * scale2) * pre_weight[i]
    end
    return
end

function dual_rms_gpu!(output, normalized, input, residual, post_weight, pre_weight, eps)
    if size(input, 2) == 1 && vector_norm_eligible(
        size(input, 1),
        output,
        normalized,
        input,
        residual,
        post_weight,
        pre_weight,
    )
        @metal threads=256 groups=1 u32_vector_dual_rms_kernel!(
            output,
            normalized,
            input,
            residual,
            post_weight,
            pre_weight,
            eps,
            Val(size(input, 1)),
            Val(8),
        )
        return output, normalized
    end
    @metal threads=256 groups=size(input, 2) dual_rms_kernel!(
        output,
        normalized,
        input,
        residual,
        post_weight,
        pre_weight,
        eps,
        size(input, 1),
    )
    return output, normalized
end

# One SIMD group owns a head, so all input values have been read before any
# lane writes a rotated pair. This also avoids GPU reductions over QKV views.
function rotate_kernel!(x, norm, eps, cosines, sines, position, width, heads, count)
    lane = Int(thread_index_in_simdgroup())
    group =
        (Int(threadgroup_position_in_grid_1d()) - 1) * 4 +
        Int(simdgroup_index_in_threadgroup())
    group > heads * count && return
    head = (group - 1) % heads
    token = (group - 1) ÷ heads + 1
    base = head * width
    square = 0.0f0
    for i = lane:32:width
        @inbounds square = muladd(x[base+i, token], x[base+i, token], square)
    end
    square = simd_sum_f32(square)
    scale = inv(sqrt(simd_shuffle(square, 1) / width + eps))
    half = width ÷ 2
    for i = lane:32:half
        @inbounds begin
            a = x[base+i, token] * scale * norm[i]
            b = x[base+half+i, token] * scale * norm[half+i]
            co = cosines[i, position+token]
            si = sines[i, position+token]
            x[base+i, token] = a * co - b * si
            x[base+half+i, token] = b * co + a * si
        end
    end
    return
end

function rotate_heads_gpu!(x, norm, eps, heads, cosines, sines, position)
    width = length(norm)
    count = size(x, 2)
    @metal threads=128 groups=cld(heads * count, 4) rotate_kernel!(
        x,
        norm,
        eps,
        cosines,
        sines,
        position,
        width,
        heads,
        count,
    )
    return x
end

function rotate_qk_kernel!(
    qkv,
    keys,
    values,
    qnorm,
    knorm,
    eps,
    cosines,
    sines,
    position,
    width,
    qheads,
    kheads,
    count,
)
    lane = Int(thread_index_in_simdgroup())
    group =
        (Int(threadgroup_position_in_grid_1d()) - 1) * 4 +
        Int(simdgroup_index_in_threadgroup())
    heads = qheads + kheads
    group > heads * count && return
    head = (group - 1) % heads
    token = (group - 1) ÷ heads + 1
    base = head * width
    square = 0.0f0
    for i = lane:32:width
        @inbounds square = muladd(qkv[base+i, token], qkv[base+i, token], square)
    end
    square = simd_sum_f32(square)
    scale = inv(sqrt(simd_shuffle(square, 1) / width + eps))
    half = width ÷ 2
    for i = lane:32:half
        @inbounds begin
            n1 = head < qheads ? qnorm[i] : knorm[i]
            n2 = head < qheads ? qnorm[half+i] : knorm[half+i]
            a = qkv[base+i, token] * scale * n1
            b = qkv[base+half+i, token] * scale * n2
            co = cosines[i, position+token]
            si = sines[i, position+token]
            first_value = a * co - b * si
            second_value = b * co + a * si
            qkv[base+i, token] = first_value
            qkv[base+half+i, token] = second_value
            if head >= qheads && !isnothing(keys)
                kvbase = (head - qheads) * width
                valuebase = (qheads + kheads) * width + kvbase
                keys[kvbase+i, position+token] = first_value
                keys[kvbase+half+i, position+token] = second_value
                values[kvbase+i, position+token] = qkv[valuebase+i, token]
                values[kvbase+half+i, position+token] = qkv[valuebase+half+i, token]
            end
        end
    end
    return
end

function rotate_qk_gpu!(
    qkv,
    qnorm,
    knorm,
    eps,
    qheads,
    kheads,
    cosines,
    sines,
    position;
    keys = nothing,
    values = nothing,
)
    isnothing(keys) == isnothing(values) ||
        throw(ArgumentError("keys and values must be provided together"))
    count = size(qkv, 2)
    if rotate32_eligible(
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
        @metal threads=128 groups=cld((qheads + kheads) * count, 4) u32_rotate_qk_kernel!(
            qkv,
            keys,
            values,
            qnorm,
            knorm,
            eps,
            cosines,
            sines,
            UInt32(position),
            Val(length(qnorm)),
            Val(qheads),
            Val(kheads),
            Val(count),
        )
        return qkv
    end
    @metal threads=128 groups=cld((qheads + kheads) * count, 4) rotate_qk_kernel!(
        qkv,
        keys,
        values,
        qnorm,
        knorm,
        eps,
        cosines,
        sines,
        position,
        length(qnorm),
        qheads,
        kheads,
        count,
    )
    return qkv
end

function score_kernel!(
    scores,
    query,
    keys,
    position,
    start,
    full,
    window,
    dim,
    heads,
    groups,
    nkeys,
    nqueries,
)
    id = Int(thread_position_in_grid_1d())
    id > nkeys * heads * nqueries && return
    key = (id - 1) % nkeys + 1
    head = ((id - 1) ÷ nkeys) % heads
    token = (id - 1) ÷ (nkeys * heads) + 1
    kp = start + key - 1
    qp = position + token
    if kp > qp || (!full && kp + window <= qp)
        @inbounds scores[key, head+1, token] = -Inf32
        return
    end
    kv = head ÷ groups
    value = 0.0f0
    for feature = 1:dim
        @inbounds value =
            muladd(query[head*dim+feature, token], keys[kv*dim+feature, kp], value)
    end
    @inbounds scores[key, head+1, token] = value / sqrt(Float32(dim))
    return
end

function softmax_kernel!(scores, heads, nkeys, nqueries)
    id = Int(thread_position_in_grid_1d())
    id > heads * nqueries && return
    head = (id - 1) % heads + 1
    token = (id - 1) ÷ heads + 1
    maximum_value = -Inf32
    for key = 1:nkeys
        @inbounds maximum_value = max(maximum_value, scores[key, head, token])
    end
    total = 0.0f0
    for key = 1:nkeys
        @inbounds begin
            v = exp(scores[key, head, token] - maximum_value)
            scores[key, head, token] = v
            total += v
        end
    end
    for key = 1:nkeys
        @inbounds scores[key, head, token] /= total
    end
    return
end

function weighted_sum_kernel!(
    output,
    scores,
    values,
    start,
    dim,
    heads,
    groups,
    nkeys,
    nqueries,
)
    id = Int(thread_position_in_grid_1d())
    id > dim * heads * nqueries && return
    feature = (id - 1) % dim + 1
    head = ((id - 1) ÷ dim) % heads
    token = (id - 1) ÷ (dim * heads) + 1
    kv = head ÷ groups
    value = 0.0f0
    for key = 1:nkeys
        @inbounds value =
            muladd(scores[key, head+1, token], values[kv*dim+feature, start+key-1], value)
    end
    @inbounds output[head*dim+feature, token] = value
    return
end

# Each threadgroup owns a query head and, for longer contexts, a 32-key
# partition. Partial softmax results are merged with their maxima for stability.
function single_attention_kernel!(
    output,
    query,
    keys,
    values,
    position,
    start,
    full,
    window,
    dim,
    groups,
    nkeys,
    heads,
    ::Val{split},
) where {split}
    tid = Int(thread_index_in_threadgroup())
    lane = Int(thread_index_in_simdgroup())
    simd = Int(simdgroup_index_in_threadgroup())
    group = Int(threadgroup_position_in_grid_1d()) - 1
    head = split ? group % heads : group
    part = split ? group ÷ heads + 1 : 1
    if split
        nkeys = min(32, nkeys - (part - 1) * 32)
        start += (part - 1) * 32
    end
    kv = head ÷ groups
    scores = MtlThreadGroupArray(Float32, split ? 32 : 2048)
    scratch = MtlThreadGroupArray(Float32, 4)
    for key = simd:4:nkeys
        kp = start + key - 1
        dot = 0.0f0
        for feature = lane:32:dim
            @inbounds dot =
                muladd(query[head*dim+feature, 1], keys[kv*dim+feature, kp], dot)
        end
        dot = simd_sum_f32(dot)
        if lane == 1
            @inbounds scores[key] =
                !full && kp + window <= position + 1 ? -Inf32 : dot / sqrt(Float32(dim))
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    maximum_value = -Inf32
    for key = tid:128:nkeys
        @inbounds maximum_value = max(maximum_value, scores[key])
    end
    maximum_value = simd_max_f32(maximum_value)
    lane == 1 && (@inbounds scratch[simd] = maximum_value)
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if simd == 1
        maxpart = lane <= 4 ? (@inbounds scratch[lane]) : -Inf32
        maxpart = simd_max_f32(maxpart)
        lane == 1 && (@inbounds scratch[1] = maxpart)
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    @inbounds maximum_value = scratch[1]
    # Every SIMD group must read the maximum before any group reuses scratch.
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    total = 0.0f0
    for key = tid:128:nkeys
        @inbounds begin
            value = maximum_value == -Inf32 ? 0.0f0 : exp(scores[key] - maximum_value)
            scores[key] = value
            total += value
        end
    end
    total = simd_sum_f32(total)
    lane == 1 && (@inbounds scratch[simd] = total)
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if simd == 1
        subtotal = lane <= 4 ? (@inbounds scratch[lane]) : 0.0f0
        subtotal = simd_sum_f32(subtotal)
        lane == 1 && (@inbounds scratch[1] = subtotal)
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    @inbounds total = scratch[1]
    if split && tid == 1
        @inbounds begin
            output[dim+1, head+1, part] = maximum_value
            output[dim+2, head+1, part] = total
        end
    end
    for feature = tid:128:dim
        value = 0.0f0
        for key = 1:nkeys
            @inbounds value =
                muladd(scores[key], values[kv*dim+feature, start+key-1], value)
        end
        @inbounds if split
            output[feature, head+1, part] = value
        else
            output[head*dim+feature, 1] = value / total
        end
    end
    return
end

function merge_attention_kernel!(output, partial, dim, heads, parts)
    tid = Int(thread_position_in_grid_1d())
    tid > dim * heads && return
    feature = (tid - 1) % dim + 1
    head = (tid - 1) ÷ dim + 1
    maximum_value = -Inf32
    for part = 1:parts
        @inbounds maximum_value = max(maximum_value, partial[dim+1, head, part])
    end
    numerator = denominator = 0.0f0
    for part = 1:parts
        @inbounds begin
            factor = exp(partial[dim+1, head, part] - maximum_value)
            numerator = muladd(factor, partial[feature, head, part], numerator)
            denominator = muladd(factor, partial[dim+2, head, part], denominator)
        end
    end
    @inbounds output[tid] = numerator / denominator
    return
end

function launch_single_attention!(
    storage,
    query,
    keys,
    values,
    position,
    start,
    full,
    window,
    dim,
    heads,
    kv_heads,
    nkeys,
    parts,
    ::Val{split},
) where {split}
    if attention128_eligible(
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
        @metal threads=128 groups=heads*parts attention128_kernel!(
            storage,
            query,
            keys,
            values,
            UInt32(position),
            UInt32(start),
            full,
            UInt32(window),
            UInt32(heads ÷ kv_heads),
            UInt32(nkeys),
            UInt32(heads),
            Val(split),
        )
        return nothing
    end
    @metal threads=128 groups=heads*parts single_attention_kernel!(
        storage,
        query,
        keys,
        values,
        position,
        start,
        full,
        window,
        dim,
        heads ÷ kv_heads,
        nkeys,
        heads,
        Val(split),
    )
    return nothing
end

function single_attention_gpu!(
    output,
    query,
    keys,
    values,
    position,
    start,
    full,
    window,
    dim,
    heads,
    kv_heads;
    partial = nothing,
)
    nkeys = position + 2 - start
    1 <= nkeys <= 2048 || throw(ArgumentError("single attention supports 1..2048 keys"))
    if nkeys > 32
        parts = cld(nkeys, 32)
        storage =
            isnothing(partial) ? MtlArray{Float32}(undef, dim+2, heads, parts) : partial
        launch_single_attention!(
            storage,
            query,
            keys,
            values,
            position,
            start,
            full,
            window,
            dim,
            heads,
            kv_heads,
            nkeys,
            parts,
            Val(true),
        )
        if attention128_eligible(
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
           ) &&
           output isa MtlArray{Float32} &&
           length(output) <= typemax(UInt32)
            @metal threads=256 groups=cld(dim*heads, 256) u32_merge_attention_kernel!(
                output,
                storage,
                UInt32(dim),
                UInt32(heads),
                UInt32(parts),
            )
        else
            @metal threads=256 groups=cld(dim*heads, 256) merge_attention_kernel!(
                output,
                storage,
                dim,
                heads,
                parts,
            )
        end
    else
        launch_single_attention!(
            output,
            query,
            keys,
            values,
            position,
            start,
            full,
            window,
            dim,
            heads,
            kv_heads,
            nkeys,
            1,
            Val(false),
        )
    end
    return output
end

function single_attention_gpu(
    query,
    keys,
    values,
    position,
    start,
    full,
    window,
    dim,
    heads,
    kv_heads,
)
    output = MtlArray{Float32}(undef, dim * heads, 1)
    return single_attention_gpu!(
        output,
        query,
        keys,
        values,
        position,
        start,
        full,
        window,
        dim,
        heads,
        kv_heads,
    )
end

# A SIMD group owns each head/query column; lanes keep their own score slots
# through all three passes, so no threadgroup-memory synchronization is needed.
function masked_softmax_kernel!(
    scores,
    position,
    start,
    full,
    window,
    heads,
    nkeys,
    nqueries,
    scale,
)
    lane = Int(thread_index_in_simdgroup())
    index =
        (Int(threadgroup_position_in_grid_1d()) - 1) * 4 +
        Int(simdgroup_index_in_threadgroup())
    index > heads * nqueries && return
    head = (index - 1) % heads + 1
    token = (index - 1) ÷ heads + 1
    qp = position + token
    maximum_value = -Inf32
    for key = lane:32:nkeys
        kp = start + key - 1
        @inbounds value =
            kp > qp || (!full && kp + window <= qp) ? -Inf32 :
            scores[key, head, token] * scale
        maximum_value = max(maximum_value, value)
    end
    maximum_value = simd_max_f32(maximum_value)
    maximum_value = simd_shuffle(maximum_value, 1)
    total = 0.0f0
    for key = lane:32:nkeys
        kp = start + key - 1
        @inbounds begin
            value =
                kp > qp || (!full && kp + window <= qp) ? 0.0f0 :
                exp(scores[key, head, token] * scale - maximum_value)
            scores[key, head, token] = value
            total += value
        end
    end
    total = simd_sum_f32(total)
    total = simd_shuffle(total, 1)
    for key = lane:32:nkeys
        @inbounds scores[key, head, token] /= total
    end
    return
end

# Contiguous K/V slices share storage with the cache. Callers can also reuse
# query, score and output buffers across all layers of a prefill chunk.
function matrix_attention_gpu!(
    output,
    query_buffer,
    scores2,
    query,
    keys,
    values,
    position,
    start,
    full,
    window,
    dim,
    heads,
)
    nqueries = size(query, 2)
    nkeys = position + nqueries - start + 1
    copyto!(query_buffer, query)
    q = reshape(query_buffer, dim, heads*nqueries)
    k = @view keys[:, start:(position+nqueries)]
    v = @view values[:, start:(position+nqueries)]
    mul!(scores2, transpose(k), q)
    scores = reshape(scores2, nkeys, heads, nqueries)
    @metal threads=128 groups=cld(heads*nqueries, 4) masked_softmax_kernel!(
        scores,
        position,
        start,
        full,
        window,
        heads,
        nkeys,
        nqueries,
        inv(sqrt(Float32(dim))),
    )
    mul!(reshape(output, dim, heads*nqueries), v, scores2)
    return output
end

function attention_gpu(
    query,
    keys,
    values,
    position,
    start,
    full,
    window,
    dim,
    heads,
    kv_heads,
)
    nqueries = size(query, 2)
    nkeys = position + nqueries - start + 1
    if nqueries >= 32 && kv_heads == 1
        query_buffer = similar(query)
        scores = MtlArray{Float32}(undef, nkeys, heads*nqueries)
        output = MtlArray{Float32}(undef, dim*heads, nqueries)
        return matrix_attention_gpu!(
            output,
            query_buffer,
            scores,
            query,
            keys,
            values,
            position,
            start,
            full,
            window,
            dim,
            heads,
        )
    end
    if nqueries == 1 && nkeys <= 2048
        return single_attention_gpu(
            query,
            keys,
            values,
            position,
            start,
            full,
            window,
            dim,
            heads,
            kv_heads,
        )
    end
    groups = heads ÷ kv_heads
    scores = MtlArray{Float32}(undef, nkeys, heads, nqueries)
    output = MtlArray{Float32}(undef, dim * heads, nqueries)
    @metal threads=256 groups=cld(nkeys * heads * nqueries, 256) score_kernel!(
        scores,
        query,
        keys,
        position,
        start,
        full,
        window,
        dim,
        heads,
        groups,
        nkeys,
        nqueries,
    )
    @metal threads=256 groups=cld(heads * nqueries, 256) softmax_kernel!(
        scores,
        heads,
        nkeys,
        nqueries,
    )
    @metal threads=256 groups=cld(dim * heads * nqueries, 256) weighted_sum_kernel!(
        output,
        scores,
        values,
        start,
        dim,
        heads,
        groups,
        nkeys,
        nqueries,
    )
    return output
end

include("metal_quantization.jl")
include("metal_decode.jl")
include("metal_token.jl")
include("metal_model.jl")

end
