# Separate GPU-owned model; the vendored CPU implementation stays unchanged.
struct MetalLayer{W,V}
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

function MetalModel(directory; context_size::Int = 4096, weight_storage::Symbol = :bf16)
    1 <= context_size <= 32768 || throw(ArgumentError("context_size must be 1..32768"))
    weight_storage in (:bf16, :f32) ||
        throw(ArgumentError("weight_storage must be :bf16 or :f32"))
    c = P.ModelConfig(P.JSON.parsefile(joinpath(directory, "config.json")))
    dtypes = Dict{String,String}()
    weights = P.checkpoint_weights(directory; dtypes, keep_bf16 = weight_storage === :bf16)
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
        end
        return MtlArray(value)
    end
    function norm(name, size, offset = 1.0f0)
        return MtlArray(P.widen_f32(takeweight(name, (size,))) .+ Float32(offset))
    end
    qdim, kdim = c.heads * c.head_dim, c.kv_heads * c.head_dim
    layers = map(0:(c.layers-1)) do i
        prefix = "model.layers.layers.$i"
        MetalLayer(
            matrix(prefix * ".mixer.qkv_proj", c.hidden, qdim + 2kdim),
            matrix(prefix * ".mixer.o_proj", qdim, c.hidden),
            matrix(prefix * ".mlp.gate_up_proj", c.hidden, 2c.intermediate),
            matrix(prefix * ".mlp.down_proj", c.intermediate, c.hidden),
            norm(prefix * ".mixer.q_norm", c.head_dim),
            norm(prefix * ".mixer.k_norm", c.head_dim),
            norm(prefix * ".pre_mixer_norm", c.hidden),
            norm(prefix * ".post_mixer_norm", c.hidden, 0.2f0),
            norm(prefix * ".pre_mlp_norm", c.hidden),
            norm(prefix * ".post_mlp_norm", c.hidden, 1 / 5^1.5),
            (i + 1) % c.pattern == 0,
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

mutable struct MetalState{A}
    keys::Vector{A}
    values::Vector{A}
    position::Int
    capacity::Int
end
function MetalState(m::MetalModel, capacity::Int)
    1 <= capacity <= m.capacity || throw(ArgumentError("invalid cache capacity"))
    rows = m.config.kv_heads * m.config.head_dim
    MetalState(
        [MtlArray{Float32}(undef, rows, capacity) for _ in m.layers],
        [MtlArray{Float32}(undef, rows, capacity) for _ in m.layers],
        0,
        capacity,
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
    for offset in (16, 8, 4, 2, 1)
        a += simd_shuffle_down(a, offset)
        b += simd_shuffle_down(b, offset)
        c += simd_shuffle_down(c, offset)
        d += simd_shuffle_down(d, offset)
    end
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
    for offset in (16, 8, 4, 2, 1)
        a += simd_shuffle_down(a, offset)
        b += simd_shuffle_down(b, offset)
        c += simd_shuffle_down(c, offset)
        d += simd_shuffle_down(d, offset)
    end
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

function projection_gpu(weight, input)
    columns = size(weight, 2)
    result = MtlArray{Float32}(undef, columns, size(input, 2))
    if eltype(weight) === Metal.BFloat16 &&
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

function forward_gpu!(m::MetalModel, ids, state::MetalState)
    isempty(ids) && throw(ArgumentError("token sequence must not be empty"))
    state.position + length(ids) <= state.capacity ||
        throw(ArgumentError("sequence exceeds cache capacity"))
    all(id -> 0 <= id < m.config.vocabulary, ids) ||
        throw(ArgumentError("token id outside vocabulary"))
    c = m.config
    n = length(ids)
    pos = state.position
    qdim, kdim = c.heads * c.head_dim, c.kv_heads * c.head_dim
    x = embedding_gpu(m.embedding, ids)
    c.scale_embedding && (x .*= sqrt(Float32(c.hidden)))
    for (index, layer) in enumerate(m.layers)
        qkv = projection_gpu(layer.qkv, rms_norm_gpu(x, layer.pre_attn, c.eps))
        q = @view qkv[1:qdim, :]
        k = @view qkv[(qdim+1):(qdim+kdim), :]
        v = @view qkv[(qdim+kdim+1):end, :]
        co, si = layer.full ? (m.full_cos, m.full_sin) : (m.local_cos, m.local_sin)
        rotate_qk_gpu!(
            qkv,
            layer.qnorm,
            layer.knorm,
            c.eps,
            c.heads,
            c.kv_heads,
            co,
            si,
            pos,
        )
        @views state.keys[index][:, (pos+1):(pos+n)] .= k
        @views state.values[index][:, (pos+1):(pos+n)] .= v
        start = layer.full ? 1 : max(1, pos + 2 - c.window)
        attended = attention_gpu(
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
        output = projection_gpu(layer.out, attended)
        output = rms_norm_gpu(output, layer.post_attn, c.eps; residual = x)
        gate_up = projection_gpu(layer.gate_up, rms_norm_gpu(output, layer.pre_mlp, c.eps))
        gate = @view gate_up[1:c.intermediate, :]
        up = @view gate_up[(c.intermediate+1):end, :]
        activated = (gate ./ (1.0f0 .+ exp.(-gate))) .* up
        mlp = projection_gpu(layer.down, activated)
        x = rms_norm_gpu(mlp, layer.post_mlp, c.eps; residual = output)
    end
    state.position += n
    hidden = copy(@view x[:, end:end])
    return vec(Array(projection_gpu(m.head, rms_norm_gpu(hidden, m.norm, c.eps))))
end

struct MetalTranslator{M} <: B.AbstractMetalTranslator
    model::M
    tokenizer::P.Tokenizer
end

B.metal_available() = Metal.functional()
function B.metal_translator(directory::AbstractString, options)
    Metal.functional() ||
        throw(ArgumentError("Metal.jl cannot access a supported Apple GPU"))
    tokenizer = P.Tokenizer(directory)
    model = MetalModel(
        directory;
        context_size = options.context_size,
        weight_storage = options.weight_storage,
    )
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
    logits = Float32[]
    for first = 1:128:length(tokens)
        logits =
            forward_gpu!(translator.model, @view(tokens[first:min(first+127, end)]), state)
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
        token = argmax(logits) - 1
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
            (logits = forward_gpu!(translator.model, [token], state))
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
