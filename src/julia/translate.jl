"""Completed or truncated translation and measured inference time, excluding model loading."""
struct Translation
    text::String
    finish_reason::Symbol
    prompt_tokens::Int
    generated_tokens::Int
    time_to_first_token::Float64
    elapsed::Float64
end

"""
    Translator(directory; context_size=4096, weight_storage=:f32)

Load official PLaMo 3 safetensors and tokenizer assets directly into a native Julia
CPU model. `:f32` expands weights to Float32. Opt-in `:bf16` retains exact original
BF16 projection bits for native Julia SIMD prefill and decode, with Float32 activations
and accumulation. BF16 source embeddings are widened only for selected token rows. F32 source projections are rejected by
`:bf16`, rather than silently rounded. Each translation owns its KV cache.
"""
struct Translator
    model::Model
    tokenizer::Tokenizer
end
function Translator(
    directory::AbstractString;
    context_size::Int = 4096,
    weight_storage::Symbol = :f32,
)
    tokenizer=Tokenizer(directory)
    model=Model(directory; context_size, weight_storage)
    length(tokenizer.pieces)==model.config.vocabulary ||
        throw(ArgumentError("vocabulary/model size mismatch"))
    return Translator(model, tokenizer)
end

function decode_bytes(bytes)
    text = String(copy(bytes))
    isvalid(text) && return text
    io = IOBuffer()
    for char in text
        print(io, isvalid(char) ? char : '\ufffd')
    end
    return String(take!(io))
end

function valid_utf8_prefix(bytes)
    # Incomplete byte-fallback tokens are withheld until the next token completes
    # the UTF-8 scalar. Invalid complete sequences are replaced in the final text.
    for stop = length(bytes):-1:max(0, length(bytes)-3)
        candidate=String(copy(@view bytes[1:stop]))
        isvalid(candidate) && return candidate
    end
    return ""
end

"""
    translate(translator, request; callback=Returns(true))

Greedily translate with an independent per-request cache. `callback` receives the
accumulated UTF-8 text; returning `false` cancels. `finish_reason` is `:stop`,
`:length`, or `:cancelled`; callers must inspect it before accepting complete text.
"""
function translate(
    translator::Translator,
    request::TranslationRequest;
    callback = Returns(true),
)
    tokens=encode(translator.tokenizer, prompt(request))
    length(tokens)+request.max_tokens<=translator.model.capacity ||
        throw(ArgumentError("prompt plus output limit exceeds context capacity"))
    model=translator.model
    started=time_ns()
    # This request cannot consume more tokens than its explicit prompt/output
    # bound. The model's logical context and RoPE tables remain unchanged.
    state=request_state(model, length(tokens), request.max_tokens)
    logits=Float32[]
    for start = 1:128:length(tokens)
        logits=forward!(model, @view(tokens[start:min(start+127, end)]), state)
    end
    stops=(
        model.config.eos,
        translator.tokenizer.specials["<|plamo:tag|>"],
        translator.tokenizer.specials["<|plamo:bos|>"],
    )
    bytes=UInt8[]
    generated=0
    first=0.0
    finish=:length
    for step = 1:request.max_tokens
        token=argmax(logits)-1
        step==1 && (first=(time_ns()-started)/1e9)
        if token in stops
            finish=:stop
            break
        end
        append!(bytes, translator.tokenizer.pieces[token+1])
        generated+=1
        if !callback(valid_utf8_prefix(bytes))
            finish=:cancelled
            break
        end
        step<request.max_tokens && (logits=forward!(model, [token], state))
    end
    text=decode_bytes(bytes)
    return Translation(
        text,
        finish,
        length(tokens),
        generated,
        first,
        (time_ns()-started)/1e9,
    )
end
