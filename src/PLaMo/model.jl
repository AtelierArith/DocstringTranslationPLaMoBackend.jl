struct ModelConfig
    hidden::Int
    intermediate::Int
    layers::Int
    heads::Int
    kv_heads::Int
    head_dim::Int
    vocabulary::Int
    eps::Float32
    pattern::Int
    window::Int
    rope_theta::Float64
    local_theta::Float64
    factor::Float64
    original_length::Int
    scale_embedding::Bool
    eos::Int
end
function ModelConfig(c)
    c["model_type"] == "plamo3" || throw(ArgumentError("expected a PLaMo 3 checkpoint"))
    get(c, "tie_word_embeddings", false) &&
        throw(ArgumentError("tied embeddings are not supported"))
    isnothing(get(c, "image_feature_size", nothing)) ||
        throw(ArgumentError("image models are not supported"))
    cfg = ModelConfig(
        c["hidden_size"],
        c["intermediate_size"],
        c["num_hidden_layers"],
        c["num_attention_heads"],
        c["num_key_value_heads"],
        c["head_dim"],
        c["vocab_size"],
        c["rms_norm_eps"],
        c["sliding_window_pattern"],
        c["window_size"],
        c["rope_theta"],
        c["rope_local_theta"],
        c["rope_scaling_factor"],
        c["initial_context_length"],
        c["scale_embedding"],
        c["eos_token_id"],
    )
    all(
        >(0),
        (
            cfg.hidden,
            cfg.intermediate,
            cfg.layers,
            cfg.heads,
            cfg.kv_heads,
            cfg.head_dim,
            cfg.vocabulary,
            cfg.pattern,
            cfg.window,
            cfg.original_length,
        ),
    ) || throw(ArgumentError("model dimensions must be positive"))
    iseven(cfg.head_dim) && cfg.heads % cfg.kv_heads == 0 ||
        throw(ArgumentError("invalid attention dimensions"))
    all(isfinite, (cfg.eps, cfg.rope_theta, cfg.local_theta, cfg.factor)) &&
    cfg.eps > 0 &&
    cfg.rope_theta > 1 &&
    cfg.local_theta > 1 &&
    cfg.factor >= 1 || throw(ArgumentError("invalid normalization/RoPE parameters"))
    return cfg
end

struct DecoderLayer
    qkv::Projection
    out::Projection
    gate_up::Projection
    down::Projection
    qnorm::Vector{Float32}
    knorm::Vector{Float32}
    pre_attn::Vector{Float32}
    post_attn::Vector{Float32}
    pre_mlp::Vector{Float32}
    post_mlp::Vector{Float32}
    full::Bool
end
struct Model
    config::ModelConfig
    embedding::Union{Matrix{Float32},Matrix{UInt16}}
    head::Projection
    norm::Vector{Float32}
    layers::Vector{DecoderLayer}
    local_cos::Matrix{Float32}
    local_sin::Matrix{Float32}
    full_cos::Matrix{Float32}
    full_sin::Matrix{Float32}
    capacity::Int
end

function rope_tables(c, full, capacity)
    half = c.head_dim ÷ 2
    base = full ? c.rope_theta : c.local_theta
    factor = full ? c.factor : 1.0
    frequencies = base .^ (Float32.(0:2:(c.head_dim-1)) ./ c.head_dim)
    inverse = inv.(frequencies)
    if factor != 1
        low = half * log(c.original_length / (32*2*pi)) / log(base)
        high = half * log(c.original_length / (2*pi)) / log(base)
        mask = 1 .- clamp.((Float32.(0:(half-1)) .- low) ./ (high-low), 0, 1)
        inverse = (1 .- mask) ./ (factor .* frequencies) .+ mask ./ frequencies
    end
    angles = Float32.(inverse) .* permutedims(Float32.(0:(capacity-1)))
    scale = Float32(1 + 0.1log(factor))
    return cos.(angles) .* scale, sin.(angles) .* scale
end

function Model(
    directory::AbstractString;
    context_size::Int = 4096,
    weight_storage::Symbol = :f32,
)
    1 <= context_size <= 32768 || throw(ArgumentError("context_size must be 1..32768"))
    c = ModelConfig(JSON.parsefile(joinpath(directory, "config.json")))
    weight_storage in (:f32, :bf16) ||
        throw(ArgumentError("weight_storage must be :f32 or :bf16"))
    dtypes = Dict{String,String}()
    weights = checkpoint_weights(directory; dtypes, keep_bf16 = weight_storage==:bf16)
    function getweight(name, shape)
        weight = get(weights, name*".weight", nothing)
        isnothing(weight) && throw(ArgumentError("missing weight $name"))
        size(weight) == shape ||
            throw(DimensionMismatch("$name has size $(size(weight)), expected $shape"))
        return weight
    end
    matrix(name, input, output) = getweight(name, (input, output))
    function projection(name, input, output)
        value=Projection(
            matrix(name, input, output),
            dtypes[name*".weight"],
            weight_storage,
        )
        # The projection owns its dense F32 or packed original BF16 payload;
        # do not retain an additional row-major array through this dictionary.
        delete!(weights, name*".weight")
        return value
    end
    norm(name, size, offset = 1.0f0) =
        widen_f32(getweight(name, (size,))) .+ Float32(offset)
    qdim, kdim = c.heads*c.head_dim, c.kv_heads*c.head_dim
    layers = DecoderLayer[]
    for i = 0:(c.layers-1)
        prefix = "model.layers.layers.$i"
        push!(
            layers,
            DecoderLayer(
                projection(prefix*".mixer.qkv_proj", c.hidden, qdim+2kdim),
                projection(prefix*".mixer.o_proj", qdim, c.hidden),
                projection(prefix*".mlp.gate_up_proj", c.hidden, 2c.intermediate),
                projection(prefix*".mlp.down_proj", c.intermediate, c.hidden),
                norm(prefix*".mixer.q_norm", c.head_dim),
                norm(prefix*".mixer.k_norm", c.head_dim),
                norm(prefix*".pre_mixer_norm", c.hidden),
                norm(prefix*".post_mixer_norm", c.hidden, 0.2f0),
                norm(prefix*".pre_mlp_norm", c.hidden),
                norm(prefix*".post_mlp_norm", c.hidden, 1/5^1.5),
                (i+1)%c.pattern==0,
            ),
        )
    end
    lc, ls=rope_tables(c, false, context_size)
    fc, fs=rope_tables(c, true, context_size)
    return Model(
        c,
        matrix("model.embed_tokens", c.hidden, c.vocabulary),
        projection("lm_head", c.hidden, c.vocabulary),
        norm("model.norm", c.hidden),
        layers,
        lc,
        ls,
        fc,
        fs,
        context_size,
    )
end

# Request-owned activation buffers; weights and the model stay immutable.
struct ForwardWorkspace
    hidden_a::Matrix{Float32}
    hidden_b::Matrix{Float32}
    normalized::Matrix{Float32}
    qkv::Matrix{Float32}
    attended::Matrix{Float32}
    gate_up::Matrix{Float32}
    blocked::Vector{Float32}
end
function ForwardWorkspace(c::ModelConfig, columns::Int)
    columns>0 || throw(ArgumentError("workspace columns must be positive"))
    qdim=c.heads*c.head_dim
    kdim=c.kv_heads*c.head_dim
    allocate(rows) = Matrix{Float32}(undef, rows, columns)
    maximum_rows=max(c.hidden, qdim+2kdim, 2c.intermediate)
    return ForwardWorkspace(
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(c.hidden),
        allocate(qdim+2kdim),
        allocate(qdim),
        allocate(2c.intermediate),
        Vector{Float32}(undef, cld(maximum_rows, 16)*columns*16),
    )
end

function workspace_layer_forward(
    layer,
    x,
    c,
    state,
    index,
    cosines,
    sines,
    workspace;
    last_query_only = false,
)
    n=size(x, 2)
    pos=state.position
    qdim, kdim=c.heads*c.head_dim, c.kv_heads*c.head_dim
    normalized=@view workspace.normalized[:, 1:n]
    rms_norm!(normalized, x, layer.pre_attn, c.eps)
    qkv=@view workspace.qkv[:, 1:n]
    project!(qkv, layer.qkv, normalized, workspace.blocked)
    q=@view qkv[1:qdim, :]
    k=@view qkv[(qdim+1):(qdim+kdim), :]
    v=@view qkv[(qdim+kdim+1):end, :]
    rotate_heads!(q, layer.qnorm, c.eps, c.heads, cosines, sines, pos)
    rotate_heads!(k, layer.knorm, c.eps, c.kv_heads, cosines, sines, pos)
    @views state.keys[index][:, (pos+1):(pos+n)].=k
    @views state.values[index][:, (pos+1):(pos+n)].=v
    first_query=last_query_only ? n : 1
    query=@view qkv[1:qdim, first_query:n]
    residual=@view x[:, first_query:n]
    query_position=pos+first_query-1
    start=layer.full ? 1 : max(1, query_position+2-c.window)
    query_count=n-first_query+1
    attended=@view workspace.attended[:, 1:query_count]
    attention!(
        attended,
        query,
        state.keys[index],
        state.values[index],
        query_position,
        start,
        layer.full,
        c.window,
        c.head_dim,
        c.heads,
        c.kv_heads,
    )
    output_parent=isodd(index) ? workspace.hidden_b : workspace.hidden_a
    output=@view output_parent[:, 1:query_count]
    project!(output, layer.out, attended, workspace.blocked)
    rms_norm!(output, output, layer.post_attn, c.eps)
    output .+= residual
    normalized=@view workspace.normalized[:, 1:query_count]
    rms_norm!(normalized, output, layer.pre_mlp, c.eps)
    gate_up=@view workspace.gate_up[:, 1:query_count]
    project!(gate_up, layer.gate_up, normalized, workspace.blocked)
    gate=@view gate_up[1:c.intermediate, :]
    up=@view gate_up[(c.intermediate+1):end, :]
    @. gate=(gate/(1.0f0+exp(-gate)))*up
    project!(normalized, layer.down, gate, workspace.blocked)
    rms_norm!(normalized, normalized, layer.post_mlp, c.eps)
    output .+= normalized
    return output
end

mutable struct ForwardState
    keys::Vector{Matrix{Float32}}
    values::Vector{Matrix{Float32}}
    position::Int
    capacity::Int
    workspace::Union{Nothing,ForwardWorkspace}
end
# Only [1:position] may be read. Each layer writes new K/V entries before
# attention sees them, so untouched capacity does not need zero initialization.
function ForwardState(m::Model; capacity::Int = m.capacity, workspace_columns::Int = 0)
    1<=capacity<=m.capacity ||
        throw(ArgumentError("cache capacity must be within the model context capacity"))
    0<=workspace_columns<=capacity ||
        throw(ArgumentError("workspace columns exceed the cache capacity"))
    return ForwardState(
        [
            Matrix{Float32}(undef, m.config.kv_heads*m.config.head_dim, capacity) for
            _ in m.layers
        ],
        [
            Matrix{Float32}(undef, m.config.kv_heads*m.config.head_dim, capacity) for
            _ in m.layers
        ],
        0,
        capacity,
        workspace_columns==0 ? nothing : ForwardWorkspace(m.config, workspace_columns),
    )
end

function request_state(model::Model, prompt_tokens::Int, max_tokens::Int)
    columns=isnothing(model.head.bf16) ? 0 : min(prompt_tokens, 128)
    ForwardState(model; capacity = prompt_tokens+max_tokens, workspace_columns = columns)
end

function rms_norm!(output, input, weight, eps)
    width=size(input, 1)
    for j in axes(input, 2)
        square=0.0f0
        @inbounds @simd for i = 1:width
            square += input[i, j]^2
        end
        scale=inv(sqrt(square/width+eps))
        @inbounds @simd for i = 1:width
            output[i, j]=(input[i, j]*scale)*weight[i]
        end
    end
    return output
end
rms_norm(input, weight, eps) = rms_norm!(similar(input), input, weight, eps)

function rotate_heads!(x, norm, eps, heads, cosines, sines, position)
    width=length(norm)
    half=width÷2
    for token in axes(x, 2), head = 0:(heads-1)
        rows=(head*width+1):((head+1)*width)
        v=@view x[rows, token:token]
        rms_norm!(v, v, norm, eps)
        @inbounds @simd for i = 1:half
            a, b=v[i, 1], v[half+i, 1]
            co, si=cosines[i, position+token], sines[i, position+token]
            v[i, 1]=a*co-b*si
            v[half+i, 1]=b*co+a*si
        end
    end
    return x
end

function layer_forward(layer, x, c, state, index, cosines, sines; last_query_only = false)
    n=size(x, 2)
    pos=state.position
    qdim, kdim=c.heads*c.head_dim, c.kv_heads*c.head_dim
    qkv=project(layer.qkv, rms_norm(x, layer.pre_attn, c.eps))
    q=@view qkv[1:qdim, :]
    k=@view qkv[(qdim+1):(qdim+kdim), :]
    v=@view qkv[(qdim+kdim+1):end, :]
    rotate_heads!(q, layer.qnorm, c.eps, c.heads, cosines, sines, pos)
    rotate_heads!(k, layer.knorm, c.eps, c.kv_heads, cosines, sines, pos)
    @views state.keys[index][:, (pos+1):(pos+n)] .= k
    @views state.values[index][:, (pos+1):(pos+n)] .= v
    # All K/V entries keep their original RoPE positions. Only the final
    # decoder layer may discard earlier query outputs when logits are last-only.
    first_query=last_query_only ? n : 1
    query=@view qkv[1:qdim, first_query:n]
    residual=@view x[:, first_query:n]
    query_position=pos+first_query-1
    start=layer.full ? 1 : max(1, query_position+2-c.window)
    attended=Matrix{Float32}(undef, qdim, n-first_query+1)
    attention!(
        attended,
        query,
        state.keys[index],
        state.values[index],
        query_position,
        start,
        layer.full,
        c.window,
        c.head_dim,
        c.heads,
        c.kv_heads,
    )
    output=project(layer.out, attended)
    rms_norm!(output, output, layer.post_attn, c.eps)
    output .+= residual
    gate_up=project(layer.gate_up, rms_norm(output, layer.pre_mlp, c.eps))
    gate=@view gate_up[1:c.intermediate, :]
    up=@view gate_up[(c.intermediate+1):end, :]
    gate .= (gate ./ (1.0f0 .+ exp.(-gate))) .* up
    mlp=project(layer.down, gate)
    rms_norm!(mlp, mlp, layer.post_mlp, c.eps)
    output .+= mlp
    return output
end

function forward!(
    m::Model,
    ids::AbstractVector{<:Integer},
    state::ForwardState;
    last_only = true,
)
    isempty(ids) && throw(ArgumentError("token sequence must not be empty"))
    state.position+length(ids)<=m.capacity ||
        throw(ArgumentError("sequence exceeds context capacity"))
    state.position+length(ids)<=state.capacity ||
        throw(ArgumentError("sequence exceeds this request's cache capacity"))
    all(id->0<=id<m.config.vocabulary, ids) ||
        throw(ArgumentError("token id outside vocabulary"))
    workspace=state.workspace
    if isnothing(workspace)
        x=embedding_lookup(m.embedding, ids)
    else
        length(ids)<=size(workspace.hidden_a, 2) ||
            throw(ArgumentError("token block exceeds request workspace capacity"))
        x=@view workspace.hidden_a[:, 1:length(ids)]
        embedding_lookup!(x, m.embedding, ids)
    end
    m.config.scale_embedding && (x .*= sqrt(Float32(m.config.hidden)))
    for (index, layer) in enumerate(m.layers)
        co, si=layer.full ? (m.full_cos, m.full_sin) : (m.local_cos, m.local_sin)
        if isnothing(workspace)
            x=layer_forward(
                layer,
                x,
                m.config,
                state,
                index,
                co,
                si;
                last_query_only = last_only && index==length(m.layers),
            )
        else
            x=workspace_layer_forward(
                layer,
                x,
                m.config,
                state,
                index,
                co,
                si,
                workspace;
                last_query_only = last_only && index==length(m.layers),
            )
        end
    end
    state.position+=length(ids)
    hidden=last_only ? x[:, end:end] : x
    logits=project(m.head, rms_norm(hidden, m.norm, m.config.eps))
    return last_only ? vec(logits) : logits
end


embedding_lookup(weight::Matrix{Float32}, ids) = weight[:, ids .+ 1]
function embedding_lookup(weight::Matrix{UInt16}, ids)
    output=Matrix{Float32}(undef, size(weight, 1), length(ids))
    for (token, id) in enumerate(ids)
        column=Int(id)+1
        @inbounds @simd for feature in axes(weight, 1)
            output[feature, token]=reinterpret(Float32, UInt32(weight[feature, column])<<16)
        end
    end
    output
end

function embedding_lookup!(
    output::AbstractMatrix{Float32},
    weight::Union{Matrix{Float32},Matrix{UInt16}},
    ids,
)
    for (token, id) in enumerate(ids)
        column=Int(id)+1
        @inbounds @simd for feature in axes(weight, 1)
            value=weight[feature, column]
            output[feature, token]=value isa UInt16 ?
                                   reinterpret(Float32, UInt32(value)<<16) : value
        end
    end
    output
end
