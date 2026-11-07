module DocstringTranslationPLaMoBackendMetalExt

using DocstringTranslationPLaMoBackend
using Metal
using LinearAlgebra

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
    for offset in (16, 8, 4, 2, 1)
        square += simd_shuffle_down(square, offset)
    end
    scale = inv(sqrt(simd_shuffle(square, 1) / width + eps))
    for i = lane:32:width
        @inbounds begin
            value = (input[i, token] * scale) * weight[i]
            output[i, token] = isnothing(residual) ? value : value + residual[i, token]
        end
    end
    return
end

function rms_norm_gpu(x, weight, eps; residual = nothing)
    output = MtlArray{Float32}(undef, size(x))
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
    for offset in (16, 8, 4, 2, 1)
        square += simd_shuffle_down(square, offset)
    end
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
    for offset in (16, 8, 4, 2, 1)
        square += simd_shuffle_down(square, offset)
    end
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
            qkv[base+i, token] = a * co - b * si
            qkv[base+half+i, token] = b * co + a * si
        end
    end
    return
end

function rotate_qk_gpu!(qkv, qnorm, knorm, eps, qheads, kheads, cosines, sines, position)
    count = size(qkv, 2)
    @metal threads=128 groups=cld((qheads + kheads) * count, 4) rotate_qk_kernel!(
        qkv,
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

# One threadgroup per query head. Four SIMD groups calculate keys in parallel;
# softmax and the value product stay in threadgroup memory during decoding.
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
)
    tid = Int(thread_index_in_threadgroup())
    lane = Int(thread_index_in_simdgroup())
    simd = Int(simdgroup_index_in_threadgroup())
    head = Int(threadgroup_position_in_grid_1d()) - 1
    kv = head ÷ groups
    scores = MtlThreadGroupArray(Float32, 2048)
    scratch = MtlThreadGroupArray(Float32, 4)
    for key = simd:4:nkeys
        kp = start + key - 1
        dot = 0.0f0
        for feature = lane:32:dim
            @inbounds dot =
                muladd(query[head*dim+feature, 1], keys[kv*dim+feature, kp], dot)
        end
        for offset in (16, 8, 4, 2, 1)
            dot += simd_shuffle_down(dot, offset)
        end
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
    for offset in (16, 8, 4, 2, 1)
        maximum_value = max(maximum_value, simd_shuffle_down(maximum_value, offset))
    end
    lane == 1 && (@inbounds scratch[simd] = maximum_value)
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if simd == 1
        maxpart = lane <= 4 ? (@inbounds scratch[lane]) : -Inf32
        for offset in (16, 8, 4, 2, 1)
            maxpart = max(maxpart, simd_shuffle_down(maxpart, offset))
        end
        lane == 1 && (@inbounds scratch[1] = maxpart)
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    @inbounds maximum_value = scratch[1]
    # Every SIMD group must read the maximum before any group reuses scratch.
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    total = 0.0f0
    for key = tid:128:nkeys
        @inbounds begin
            value = exp(scores[key] - maximum_value)
            scores[key] = value
            total += value
        end
    end
    for offset in (16, 8, 4, 2, 1)
        total += simd_shuffle_down(total, offset)
    end
    lane == 1 && (@inbounds scratch[simd] = total)
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if simd == 1
        subtotal = lane <= 4 ? (@inbounds scratch[lane]) : 0.0f0
        for offset in (16, 8, 4, 2, 1)
            subtotal += simd_shuffle_down(subtotal, offset)
        end
        lane == 1 && (@inbounds scratch[1] = subtotal)
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    @inbounds total = scratch[1]
    for feature = tid:128:dim
        value = 0.0f0
        for key = 1:nkeys
            @inbounds value =
                muladd(scores[key], values[kv*dim+feature, start+key-1], value)
        end
        @inbounds output[head*dim+feature, 1] = value / total
    end
    return
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
    nkeys = position + 2 - start
    1 <= nkeys <= 2048 || throw(ArgumentError("single attention supports 1..2048 keys"))
    output = MtlArray{Float32}(undef, dim * heads, 1)
    @metal threads=128 groups=heads single_attention_kernel!(
        output,
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
    )
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

include("metal_model.jl")

end
