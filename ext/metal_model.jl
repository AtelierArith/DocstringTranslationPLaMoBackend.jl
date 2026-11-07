# Separate GPU-owned model; the vendored CPU implementation stays unchanged.
struct MetalLayer{W,V,F}
    qkv::W
    out::W
    gate_up::W
    down::W
    qnorm::V
    knorm::V
    pre_attn::V
    post_attn::V
    pre_mlp::V
    post_mlp::V
    full::Bool
    prefill::F
end

struct MetalModel{W,V,L}
    config::P.ModelConfig
    embedding::W
    head::W
    norm::V
    layers::Vector{L}
    local_cos::MtlMatrix{Float32}
    local_sin::MtlMatrix{Float32}
    full_cos::MtlMatrix{Float32}
    full_sin::MtlMatrix{Float32}
    capacity::Int
end

expanded_prefill_weight(weight::QuantizedMatrix) = dequantize_gpu(weight)
expanded_prefill_weight(weight) = Float32.(weight)

function MetalModel(
    directory;
    context_size::Int = 4096,
    weight_storage::Symbol = :bf16,
    hybrid::Bool = false,
)
    1 <= context_size <= 32768 || throw(ArgumentError("context_size must be 1..32768"))
    weight_storage in (:bf16, :f32, :q8_0, :q4_0) ||
        throw(ArgumentError("weight_storage must be :bf16, :f32, :q8_0 or :q4_0"))
    c = P.ModelConfig(P.JSON.parsefile(joinpath(directory, "config.json")))
    dtypes = Dict{String,String}()
    weights = P.checkpoint_weights(directory; dtypes, keep_bf16 = weight_storage !== :f32)
    function takeweight(name, shape)
        key = name * ".weight"
        value = pop!(weights, key, nothing)
        isnothing(value) && throw(ArgumentError("missing weight $name"))
        size(value) == shape ||
            throw(DimensionMismatch("$name has size $(size(value)), expected $shape"))
        return value
    end
    function matrix(name, input, output)
        value = takeweight(name, (input, output))
        if weight_storage === :bf16
            eltype(value) === UInt16 || throw(ArgumentError("$name is not BF16"))
            return MtlArray(reinterpret(Metal.BFloat16, value))
        elseif weight_storage in (:q8_0, :q4_0)
            eltype(value) === UInt16 || throw(ArgumentError("$name is not BF16"))
            return quantize_matrix(value, weight_storage === :q8_0 ? Val(8) : Val(4))
        end
        return MtlArray(value)
    end
    function norm(name, size, offset = 1.0f0)
        return MtlArray(P.widen_f32(takeweight(name, (size,))) .+ Float32(offset))
    end
    qdim, kdim = c.heads * c.head_dim, c.kv_heads * c.head_dim
    layers = map(0:(c.layers-1)) do i
        prefix = "model.layers.layers.$i"
        qkv = matrix(prefix * ".mixer.qkv_proj", c.hidden, qdim + 2kdim)
        out = matrix(prefix * ".mixer.o_proj", qdim, c.hidden)
        gate_up = matrix(prefix * ".mlp.gate_up_proj", c.hidden, 2c.intermediate)
        down = matrix(prefix * ".mlp.down_proj", c.intermediate, c.hidden)
        prefill =
            hybrid && weight_storage !== :f32 ?
            map(expanded_prefill_weight, (qkv, out, gate_up, down)) : nothing
        MetalLayer(
            qkv,
            out,
            gate_up,
            down,
            norm(prefix * ".mixer.q_norm", c.head_dim),
            norm(prefix * ".mixer.k_norm", c.head_dim),
            norm(prefix * ".pre_mixer_norm", c.hidden),
            norm(prefix * ".post_mixer_norm", c.hidden, 0.2f0),
            norm(prefix * ".pre_mlp_norm", c.hidden),
            norm(prefix * ".post_mlp_norm", c.hidden, 1 / 5^1.5),
            (i + 1) % c.pattern == 0,
            prefill,
        )
    end
    lc, ls = P.rope_tables(c, false, context_size)
    fc, fs = P.rope_tables(c, true, context_size)
    MetalModel(
        c,
        matrix("model.embed_tokens", c.hidden, c.vocabulary),
        matrix("lm_head", c.hidden, c.vocabulary),
        norm("model.norm", c.hidden),
        layers,
        MtlArray(lc),
        MtlArray(ls),
        MtlArray(fc),
        MtlArray(fs),
        context_size,
    )
end

struct DecodeWorkspace{A,S,T}
    hidden::A
    residual::A
    normalized::A
    projected::A
    qkv::A
    attended::A
    gate_up::A
    gate::A
    logits::A
    attention_partial::S
    token_partial::T
end
function DecodeWorkspace(c, capacity)
    allocate(rows) = MtlArray{Float32}(undef, rows, 1)
    qdim, kdim = c.heads * c.head_dim, c.kv_heads * c.head_dim
    DecodeWorkspace(
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(qdim + 2kdim),
        allocate(qdim),
        allocate(2c.intermediate),
        allocate(c.intermediate),
        allocate(c.vocabulary),
        MtlArray{Float32}(undef, c.head_dim+2, c.heads, cld(min(capacity, 2048), 32)),
        MtlArray{Tuple{Float32,Int32}}(undef, cld(c.vocabulary, 1024)),
    )
end

struct PrefillWorkspace{A}
    hidden::A
    residual::A
    normalized::A
    projected::A
    qkv::A
    attended::A
    query::A
    gate_up::A
    gate::A
end

function PrefillWorkspace(c, count)
    allocate(rows) = MtlArray{Float32}(undef, rows, count)
    qdim, kdim = c.heads * c.head_dim, c.kv_heads * c.head_dim
    PrefillWorkspace(
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(qdim + 2kdim),
        allocate(qdim),
        allocate(qdim),
        allocate(2c.intermediate),
        allocate(c.intermediate),
    )
end

mutable struct MetalState{A,W}
    keys::Vector{A}
    values::Vector{A}
    position::Int
    capacity::Int
    workspace::W
    prefill::Union{Nothing,PrefillWorkspace{A}}
end
function MetalState(m::MetalModel, capacity::Int)
    1 <= capacity <= m.capacity || throw(ArgumentError("invalid cache capacity"))
    rows = m.config.kv_heads * m.config.head_dim
    MetalState(
        [MtlArray{Float32}(undef, rows, capacity) for _ in m.layers],
        [MtlArray{Float32}(undef, rows, capacity) for _ in m.layers],
        0,
        capacity,
        DecodeWorkspace(m.config, capacity),
        nothing,
    )
end

function embedding_kernel!(output, weight, ids, hidden, tokens)
    id = Int(thread_position_in_grid_1d())
    id > hidden * tokens && return
    feature = (id - 1) % hidden + 1
    token = (id - 1) ÷ hidden + 1
    @inbounds output[feature, token] = Float32(weight[feature, ids[token]+1])
    return
end

function embedding_gpu(weight, ids)
    hidden = size(weight, 1)
    tokens = length(ids)
    output = MtlArray{Float32}(undef, hidden, tokens)
    dev_ids = MtlArray(Int32.(ids))
    @metal threads=256 groups=cld(hidden * tokens, 256) embedding_kernel!(
        output,
        weight,
        dev_ids,
        hidden,
        tokens,
    )
    return output
end

# BF16 weights stay packed on the GPU; accumulate products in Float32.
# Four output rows share each activation load within a SIMD group.
function bf16_matvec_kernel!(output, weight, input, width, columns)
    lane = Int(thread_index_in_simdgroup())
    base =
        (Int(threadgroup_position_in_grid_1d()) - 1) * 16 +
        (Int(simdgroup_index_in_threadgroup()) - 1) * 4 +
        1
    base > columns && return
    a = b = c = d = 0.0f0
    for k = lane:32:width
        @inbounds begin
            value = input[k, 1]
            a = muladd(Float32(weight[k, base]), value, a)
            if base + 1 <= columns
                b = muladd(Float32(weight[k, base+1]), value, b)
            end
            if base + 2 <= columns
                c = muladd(Float32(weight[k, base+2]), value, c)
            end
            if base + 3 <= columns
                d = muladd(Float32(weight[k, base+3]), value, d)
            end
        end
    end
    a = simd_sum_f32(a)
    b = simd_sum_f32(b)
    c = simd_sum_f32(c)
    d = simd_sum_f32(d)
    if lane == 1
        @inbounds begin
            output[base, 1] = a
            base + 1 <= columns && (output[base+1, 1] = b)
            base + 2 <= columns && (output[base+2, 1] = c)
            base + 3 <= columns && (output[base+3, 1] = d)
        end
    end
    return
end

const Vec = P.SIMD.Vec
const vload = P.SIMD.vload

# Packed, aligned four-element loads make BF16 decode more bandwidth efficient.
function bf16_matvec_vector_kernel!(output, weight, input, width, columns)
    lane = Int(thread_index_in_simdgroup())
    base =
        (Int(threadgroup_position_in_grid_1d()) - 1) * 16 +
        (Int(simdgroup_index_in_threadgroup()) - 1) * 4 +
        1
    base > columns && return
    a = b = c = d = 0.0f0
    for block = 0:(width÷128-1)
        k = block * 128 + (lane - 1) * 4 + 1
        xv = vload(Vec{4,Float32}, pointer(input, k))
        for row = 0:3
            ptr = reinterpret(
                Core.LLVMPtr{UInt16,Metal.AS.Device},
                pointer(weight, k + (base + row - 1) * width),
            )
            raw = vload(Vec{4,UInt16}, ptr)
            vals = reinterpret(Vec{4,Float32}, Vec{4,UInt32}(raw) << 16)
            product = sum(vals * xv)
            if row == 0
                a += product
            elseif row == 1
                b += product
            elseif row == 2
                c += product
            else
                d += product
            end
        end
    end
    a = simd_sum_f32(a)
    b = simd_sum_f32(b)
    c = simd_sum_f32(c)
    d = simd_sum_f32(d)
    if lane == 1
        @inbounds begin
            output[base, 1] = a
            output[base+1, 1] = b
            output[base+2, 1] = c
            output[base+3, 1] = d
        end
    end
    return
end

function bf16_matvec_group_kernel!(out, weight, x, width, cols)
    lane = Int(thread_index_in_simdgroup())
    simd = Int(simdgroup_index_in_threadgroup())
    tid = Int(thread_index_in_threadgroup())
    base = (Int(threadgroup_position_in_grid_1d()) - 1) * 4 + 1
    partial = MtlThreadGroupArray(Float32, 16)
    a0=0.0f0
    a1=0.0f0
    a2=0.0f0
    a3=0.0f0
    for k = ((simd-1)*128+(lane-1)*4+1):512:width
        xv=vload(Vec{4,Float32}, pointer(x, k))
        ptr0=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(weight, k+(base+0-1)*width),
        )
        val0=reinterpret(Vec{4,Float32}, Vec{4,UInt32}(vload(Vec{4,UInt16}, ptr0))<<16)
        a0 += sum(val0*xv)
        ptr1=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(weight, k+(base+1-1)*width),
        )
        val1=reinterpret(Vec{4,Float32}, Vec{4,UInt32}(vload(Vec{4,UInt16}, ptr1))<<16)
        a1 += sum(val1*xv)
        ptr2=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(weight, k+(base+2-1)*width),
        )
        val2=reinterpret(Vec{4,Float32}, Vec{4,UInt32}(vload(Vec{4,UInt16}, ptr2))<<16)
        a2 += sum(val2*xv)
        ptr3=reinterpret(
            Core.LLVMPtr{UInt16,Metal.AS.Device},
            pointer(weight, k+(base+3-1)*width),
        )
        val3=reinterpret(Vec{4,Float32}, Vec{4,UInt32}(vload(Vec{4,UInt16}, ptr3))<<16)
        a3 += sum(val3*xv)
    end
    a0 = simd_sum_f32(a0)
    a1 = simd_sum_f32(a1)
    a2 = simd_sum_f32(a2)
    a3 = simd_sum_f32(a3)
    if lane==1
        @inbounds begin
            partial[(simd-1)*4+1] = a0
            partial[(simd-1)*4+2] = a1
            partial[(simd-1)*4+3] = a2
            partial[(simd-1)*4+4] = a3
        end
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if tid <= 4
        value=0.0f0
        for group = 0:3
            @inbounds value+=partial[group*4+tid]
        end
        @inbounds out[base+tid-1] = value
    end
    return
end

# Load eight BF16 values as four UInt32 words, then widen each half exactly.
# Four SIMD groups split the work for each pair of Float32 output rows.
function bf16_packed_matvec_kernel!(out, weight, x, width, cols)
    lane=Int(thread_index_in_simdgroup())
    simd=Int(simdgroup_index_in_threadgroup())
    tid=Int(thread_index_in_threadgroup())
    base=(Int(threadgroup_position_in_grid_1d())-1)*2+1
    partial=MtlThreadGroupArray(Float32, 8)
    a0=0.0f0
    a1=0.0f0
    for k = ((simd-1)*256+(lane-1)*8+1):1024:width
        xv=vload(Vec{8,Float32}, pointer(x, k))
        xe=P.SIMD.shufflevector(xv, Val((0, 2, 4, 6)))
        xo=P.SIMD.shufflevector(xv, Val((1, 3, 5, 7)))
        ptr0=reinterpret(
            Core.LLVMPtr{UInt32,Metal.AS.Device},
            pointer(weight, k+(base+0-1)*width),
        )
        raw0=vload(Vec{4,UInt32}, ptr0)
        low0=reinterpret(Vec{4,Float32}, raw0<<16)
        high0=reinterpret(Vec{4,Float32}, raw0&Vec{4,UInt32}(0xffff0000))
        a0+=sum(low0*xe)+sum(high0*xo)
        ptr1=reinterpret(
            Core.LLVMPtr{UInt32,Metal.AS.Device},
            pointer(weight, k+(base+1-1)*width),
        )
        raw1=vload(Vec{4,UInt32}, ptr1)
        low1=reinterpret(Vec{4,Float32}, raw1<<16)
        high1=reinterpret(Vec{4,Float32}, raw1&Vec{4,UInt32}(0xffff0000))
        a1+=sum(low1*xe)+sum(high1*xo)
    end
    a0 = simd_sum_f32(a0)
    a1 = simd_sum_f32(a1)
    if lane==1
        @inbounds partial[(simd-1)*2+1]=a0
        @inbounds partial[(simd-1)*2+2]=a1
    end
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if tid<=2
        value=0.0f0
        for group = 0:3
            @inbounds value+=partial[group*2+tid]
        end
        @inbounds out[base+tid-1]=value
    end
    return
end
function projection_gpu!(result, weight, input)
    columns = size(weight, 2)
    if eltype(weight) === Metal.BFloat16 &&
       size(input, 2) == 1 &&
       size(weight, 1) >= 1024 &&
       size(weight, 1) % 256 == 0 &&
       columns % 2 == 0 &&
       weight isa MtlArray &&
       input isa MtlArray
        @metal threads=128 groups=cld(columns, 2) bf16_packed_matvec_kernel!(
            result,
            weight,
            input,
            size(weight, 1),
            columns,
        )
    elseif eltype(weight) === Metal.BFloat16 &&
           size(input, 2) == 1 &&
           size(weight, 1) % 512 == 0 &&
           columns % 4 == 0 &&
           columns <= 12288 &&
           input isa MtlArray
        @metal threads=128 groups=cld(columns, 4) bf16_matvec_group_kernel!(
            result,
            weight,
            input,
            size(weight, 1),
            columns,
        )
    elseif eltype(weight) === Metal.BFloat16 &&
           size(input, 2) == 1 &&
           size(weight, 1) % 128 == 0 &&
           columns % 16 == 0 &&
           input isa MtlArray
        @metal threads=128 groups=cld(columns, 16) bf16_matvec_vector_kernel!(
            result,
            weight,
            input,
            size(weight, 1),
            columns,
        )
    elseif eltype(weight) === Metal.BFloat16 && size(input, 2) == 1
        @metal threads=128 groups=cld(columns, 16) bf16_matvec_kernel!(
            result,
            weight,
            input,
            size(weight, 1),
            columns,
        )
    else
        mul!(result, transpose(weight), input)
    end
    return result
end

function projection_gpu(weight, input)
    result = MtlArray{Float32}(undef, size(weight, 2), size(input, 2))
    return projection_gpu!(result, weight, input)
end

gpu_token(logits) = Int(argmax(vec(logits))) - 1

function forward_gpu!(m::MetalModel, ids, state::MetalState; token_only = false)
    isempty(ids) && throw(ArgumentError("token sequence must not be empty"))
    state.position + length(ids) <= state.capacity ||
        throw(ArgumentError("sequence exceeds cache capacity"))
    all(id -> 0 <= id < m.config.vocabulary, ids) ||
        throw(ArgumentError("token id outside vocabulary"))
    length(ids) == 1 && return forward_decode_gpu!(m, only(ids), state; token_only)
    c = m.config
    n = length(ids)
    pos = state.position
    ws = state.prefill
    if isnothing(ws) || size(ws.hidden, 2) != n
        ws = PrefillWorkspace(c, n)
        state.prefill = ws
    end
    qdim = c.heads * c.head_dim
    dev_ids = MtlArray(Int32.(ids))
    @metal threads=256 groups=cld(c.hidden*n, 256) embedding_kernel!(
        ws.hidden,
        m.embedding,
        dev_ids,
        c.hidden,
        n,
    )
    c.scale_embedding && (ws.hidden .*= sqrt(Float32(c.hidden)))
    rms_norm_gpu!(ws.normalized, ws.hidden, first(m.layers).pre_attn, c.eps)
    full_scores = local_scores = nothing
    for (index, layer) in enumerate(m.layers)
        prefill = layer.prefill
        wq, wo, wg, wd =
            isnothing(prefill) ? (layer.qkv, layer.out, layer.gate_up, layer.down) : prefill
        projection_gpu!(ws.qkv, wq, ws.normalized)
        q = @view ws.qkv[1:qdim, :]
        co, si = layer.full ? (m.full_cos, m.full_sin) : (m.local_cos, m.local_sin)
        rotate_qk_gpu!(
            ws.qkv,
            layer.qnorm,
            layer.knorm,
            c.eps,
            c.heads,
            c.kv_heads,
            co,
            si,
            pos;
            keys = state.keys[index],
            values = state.values[index],
        )
        start = layer.full ? 1 : max(1, pos + 2 - c.window)
        attended = if n >= 32 && c.kv_heads == 1
            nkeys = pos + n - start + 1
            scores = if layer.full
                if isnothing(full_scores)
                    full_scores = MtlArray{Float32}(undef, nkeys, c.heads*n)
                end
                full_scores
            else
                if isnothing(local_scores)
                    local_scores = MtlArray{Float32}(undef, nkeys, c.heads*n)
                end
                local_scores
            end
            matrix_attention_gpu!(
                ws.attended,
                ws.query,
                scores,
                q,
                state.keys[index],
                state.values[index],
                pos,
                start,
                layer.full,
                c.window,
                c.head_dim,
                c.heads,
            )
        else
            attention_gpu(
                q,
                state.keys[index],
                state.values[index],
                pos,
                start,
                layer.full,
                c.window,
                c.head_dim,
                c.heads,
                c.kv_heads,
            )
        end
        projection_gpu!(ws.projected, wo, attended)
        dual_rms_gpu!(
            ws.residual,
            ws.normalized,
            ws.projected,
            ws.hidden,
            layer.post_attn,
            layer.pre_mlp,
            c.eps,
        )
        projection_gpu!(ws.gate_up, wg, ws.normalized)
        gate = @view ws.gate_up[1:c.intermediate, :]
        up = @view ws.gate_up[(c.intermediate+1):end, :]
        @. ws.gate = (gate / (1.0f0 + exp(-gate))) * up
        projection_gpu!(ws.projected, wd, ws.gate)
        next_norm = index < length(m.layers) ? m.layers[index+1].pre_attn : m.norm
        dual_rms_gpu!(
            ws.hidden,
            ws.normalized,
            ws.projected,
            ws.residual,
            layer.post_mlp,
            next_norm,
            c.eps,
        )
    end
    state.position += n
    hidden = @view ws.normalized[:, end:end]
    projection_gpu!(state.workspace.logits, m.head, hidden)
    return token_only ? gpu_token!(state.workspace.logits, state.workspace.token_partial) :
           vec(Array(state.workspace.logits))
end

function embedding_decode_kernel!(output, weight, token, hidden)
    i = Int(thread_position_in_grid_1d())
    i > hidden && return
    @inbounds output[i, 1] = Float32(weight[i, token+1])
    return
end


function activate_gate_kernel!(out, gate_up, intermediate)
    i = Int(thread_position_in_grid_1d())
    i > intermediate && return
    @inbounds begin
        gate = gate_up[i, 1]
        out[i, 1] = (gate / (1.0f0 + exp(-gate))) * gate_up[intermediate+i, 1]
    end
    return
end

function forward_decode_gpu!(
    m::MetalModel,
    token::Integer,
    state::MetalState;
    token_only = false,
)
    c = m.config
    pos = state.position
    ws = state.workspace
    qdim, kdim = c.heads * c.head_dim, c.kv_heads * c.head_dim
    @metal threads=256 groups=cld(c.hidden, 256) embedding_decode_kernel!(
        ws.hidden,
        m.embedding,
        token,
        c.hidden,
    )
    c.scale_embedding && (ws.hidden .*= sqrt(Float32(c.hidden)))
    rms_norm_gpu!(ws.normalized, ws.hidden, first(m.layers).pre_attn, c.eps)
    for (index, layer) in enumerate(m.layers)
        projection_gpu!(ws.qkv, layer.qkv, ws.normalized)
        co, si = layer.full ? (m.full_cos, m.full_sin) : (m.local_cos, m.local_sin)
        rotate_qk_gpu!(
            ws.qkv,
            layer.qnorm,
            layer.knorm,
            c.eps,
            c.heads,
            c.kv_heads,
            co,
            si,
            pos;
            keys = state.keys[index],
            values = state.values[index],
        )
        start = layer.full ? 1 : max(1, pos + 2 - c.window)
        nkeys = pos + 2 - start
        if nkeys <= 2048
            single_attention_gpu!(
                ws.attended,
                ws.qkv,
                state.keys[index],
                state.values[index],
                pos,
                start,
                layer.full,
                c.window,
                c.head_dim,
                c.heads,
                c.kv_heads;
                partial = ws.attention_partial,
            )
        else
            query = @view ws.qkv[1:qdim, :]
            attended = attention_gpu(
                query,
                state.keys[index],
                state.values[index],
                pos,
                start,
                layer.full,
                c.window,
                c.head_dim,
                c.heads,
                c.kv_heads,
            )
            ws.attended .= attended
        end
        projection_gpu!(ws.projected, layer.out, ws.attended)
        dual_rms_gpu!(
            ws.residual,
            ws.normalized,
            ws.projected,
            ws.hidden,
            layer.post_attn,
            layer.pre_mlp,
            c.eps,
        )
        projection_gpu!(ws.gate_up, layer.gate_up, ws.normalized)
        @metal threads=256 groups=cld(c.intermediate, 256) activate_gate_kernel!(
            ws.gate,
            ws.gate_up,
            c.intermediate,
        )
        projection_gpu!(ws.projected, layer.down, ws.gate)
        next_norm = index < length(m.layers) ? m.layers[index+1].pre_attn : m.norm
        dual_rms_gpu!(
            ws.hidden,
            ws.normalized,
            ws.projected,
            ws.residual,
            layer.post_mlp,
            next_norm,
            c.eps,
        )
    end
    state.position += 1
    projection_gpu!(ws.logits, m.head, ws.normalized)
    return token_only ? gpu_token!(ws.logits, ws.token_partial) : vec(Array(ws.logits))
end

struct MetalTranslator{M} <: B.AbstractMetalTranslator
    model::M
    tokenizer::P.Tokenizer
end

B.metal_available() = Metal.functional()
function kv_cache_bytes(directory, context_size)
    c = P.ModelConfig(P.JSON.parsefile(joinpath(directory, "config.json")))
    return 2 * c.layers * c.kv_heads * c.head_dim * context_size * sizeof(Float32)
end

function use_hybrid_prefill(
    weight_storage,
    prefill_storage,
    gpu_free,
    system_free,
    cache_bytes,
)
    weight_storage in (:bf16, :q8_0, :q4_0) || return false
    prefill_storage === :f32 && return true
    prefill_storage === :auto || return false
    # Includes the BF16 model, expanded prefill weights, host-side loading,
    # temporary buffers and a margin for the rest of the application.
    required = 8 * 1024^3 + cache_bytes
    return gpu_free >= required && system_free >= required
end

function load_model_with_fallback(loader, directory, options, hybrid)
    model = try
        loader(
            directory;
            context_size = options.context_size,
            weight_storage = options.weight_storage,
            hybrid,
        )
    catch err
        if err isa OutOfMemoryError && hybrid && options.prefill_storage === :auto
            @warn "Float32 prefill cache could not be allocated; falling back to BF16 prefill"
            nothing
        else
            rethrow()
        end
    end
    if isnothing(model)
        GC.gc(true)
        Metal.synchronize()
        GC.gc(true)
        return loader(
            directory;
            context_size = options.context_size,
            weight_storage = options.weight_storage,
            hybrid = false,
        )
    end
    return model
end

function B.metal_translator(directory::AbstractString, options)
    Metal.functional() ||
        throw(ArgumentError("Metal.jl cannot access a supported Apple GPU"))
    tokenizer = P.Tokenizer(directory)
    cache_bytes = kv_cache_bytes(directory, options.context_size)
    hybrid = use_hybrid_prefill(
        options.weight_storage,
        options.prefill_storage,
        Metal.free_memory(),
        Sys.free_memory(),
        cache_bytes,
    )
    model = load_model_with_fallback(MetalModel, directory, options, hybrid)
    length(tokenizer.pieces) == model.config.vocabulary ||
        throw(ArgumentError("vocabulary/model size mismatch"))
    return MetalTranslator(model, tokenizer)
end

function P.translate(
    translator::MetalTranslator,
    request::P.TranslationRequest;
    callback = Returns(true),
)
    tokens = P.encode(translator.tokenizer, P.prompt(request))
    length(tokens) + request.max_tokens <= translator.model.capacity ||
        throw(ArgumentError("prompt plus output limit exceeds context capacity"))
    started = time_ns()
    state = MetalState(translator.model, length(tokens) + request.max_tokens)
    token = 0
    for first = 1:128:length(tokens)
        token = forward_gpu!(
            translator.model,
            @view(tokens[first:min(first+127, end)]),
            state;
            token_only = true,
        )
    end
    stops = (
        translator.model.config.eos,
        translator.tokenizer.specials["<|plamo:tag|>"],
        translator.tokenizer.specials["<|plamo:bos|>"],
    )
    bytes = UInt8[]
    generated = 0
    first = 0.0
    finish = :length
    for step = 1:request.max_tokens
        step == 1 && (first = (time_ns() - started) / 1e9)
        if token in stops
            finish = :stop
            break
        end
        append!(bytes, translator.tokenizer.pieces[token+1])
        generated += 1
        if !callback(P.valid_utf8_prefix(bytes))
            finish = :cancelled
            break
        end
        step < request.max_tokens &&
            (token = forward_gpu!(translator.model, (token,), state; token_only = true))
    end
    return P.Translation(
        P.decode_bytes(bytes),
        finish,
        length(tokens),
        generated,
        first,
        (time_ns() - started) / 1e9,
    )
end
