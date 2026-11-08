# Compute every query head in one KV group as columns of one GEMM. This avoids
# repeatedly packing the same cache for each query head. Query columns are
# ordered [head-within-group, token], matching Julia's column-major layout.
function check_attention(
    output::AbstractMatrix{Float32},
    query::AbstractMatrix{Float32},
    keys::AbstractMatrix{Float32},
    values::AbstractMatrix{Float32},
    position::Int,
    start::Int,
    full::Bool,
    window::Int,
    dim::Int,
    heads::Int,
    kv_heads::Int,
)
    Base.require_one_based_indexing(output, query, keys, values)
    n=size(query, 2)
    dim>0 && heads>0 && kv_heads>0 && heads%kv_heads==0 && n>0 && position>=0 && window>0 ||
        throw(ArgumentError("invalid grouped attention dimensions"))
    total=position+n
    size(query, 1)==dim*heads &&
    size(output)==size(query) &&
    size(keys, 1)==dim*kv_heads &&
    size(values, 1)==dim*kv_heads &&
    size(keys, 2)>=total &&
    size(values, 2)>=total &&
    1<=start<=position+1 ||
        throw(DimensionMismatch("grouped attention cache/query shapes differ"))
    (
        Base.mightalias(output, query) ||
        Base.mightalias(output, keys) ||
        Base.mightalias(output, values)
    ) && throw(ArgumentError("attention output must not alias query or cache"))
    return n, total
end

function grouped_attention!(
    output::AbstractMatrix{Float32},
    query::AbstractMatrix{Float32},
    keys::AbstractMatrix{Float32},
    values::AbstractMatrix{Float32},
    position::Int,
    start::Int,
    full::Bool,
    window::Int,
    dim::Int,
    heads::Int,
    kv_heads::Int,
)
    n, total=check_attention(
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
    mask_needed=n>1 || (!full && start<max(1, position+2-window))
    groups=heads÷kv_heads
    scale=sqrt(Float32(dim))
    for kv = 0:(kv_heads-1)
        qr=(kv*dim*groups+1):((kv+1)*dim*groups)
        kr=(kv*dim+1):((kv+1)*dim)
        queries=reshape(copy(@view(query[qr, :])), dim, groups*n)
        key=@view keys[kr, start:total]
        value=@view values[kr, start:total]
        scores=transpose(key)*queries
        scores ./= scale
        for column in axes(scores, 2)
            if mask_needed
                token=(column-1)÷groups+1
                query_position=position+token
                @inbounds for row in axes(scores, 1)
                    key_position=start+row-1
                    if key_position>query_position ||
                       (!full && key_position+window<=query_position)
                        scores[row, column]=-Inf32
                    end
                end
            end
            probabilities=@view scores[:, column]
            maximum_value=maximum(probabilities)
            probabilities .= exp.(probabilities .- maximum_value)
            probabilities ./= sum(probabilities)
        end
        product=value*scores
        @views output[qr, :] .= reshape(product, dim*groups, n)
    end
    return output
end

# Retain per-head prompt attention: interleaved measurements found that grouping
# a cold 128-token prefill regresses on this BLAS backend. Single-query decode
# still benefits strongly from sharing each KV group across the query heads.
function per_head_attention!(
    output::AbstractMatrix{Float32},
    query::AbstractMatrix{Float32},
    keys::AbstractMatrix{Float32},
    values::AbstractMatrix{Float32},
    position::Int,
    start::Int,
    full::Bool,
    window::Int,
    dim::Int,
    heads::Int,
    kv_heads::Int,
)
    n, _=check_attention(
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
    groups=heads÷kv_heads
    for head = 0:(heads-1)
        kv=head÷groups
        qr=(head*dim+1):((head+1)*dim)
        kr=(kv*dim+1):((kv+1)*dim)
        scores=transpose(@view(keys[kr, start:(position+n)]))*(@view query[qr, :])
        scores ./= sqrt(Float32(dim))
        for token = 1:n
            query_position=position+token
            @inbounds for row in axes(scores, 1)
                key_position=start+row-1
                if key_position>query_position ||
                   (!full && key_position+window<=query_position)
                    scores[row, token]=-Inf32
                end
            end
            probabilities=@view scores[:, token]
            probabilities .= exp.(probabilities .- maximum(probabilities))
            probabilities ./= sum(probabilities)
        end
        mul!(@view(output[qr, :]), @view(values[kr, start:(position+n)]), scores)
    end
    return output
end

function attention!(
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
    if size(query, 2)==1
        return grouped_attention!(
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
    return per_head_attention!(
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
