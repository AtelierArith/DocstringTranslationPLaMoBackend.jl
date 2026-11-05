mutable struct TokenNode
    edges::Dict{Char,Int}
    id::Int
    score::Int64
end
TokenNode() = TokenNode(Dict{Char,Int}(), -1, 0)

struct Tokenizer
    nodes::Vector{TokenNode}
    pieces::Vector{Vector{UInt8}}
    specials::Dict{String,Int}
    bytes::Vector{Int}
    spaces::Int
    repeats::Int
end

function Tokenizer(directory::AbstractString)
    cfg = JSON.parsefile(joinpath(directory, "tokenizer_config.json"))
    nodes = [TokenNode()]
    pieces = Vector{UInt8}[]
    specials = Dict{String,Int}()
    bytes = fill(-1, 256)
    for (index, line) in enumerate(eachline(joinpath(directory, "tokenizer.jsonl")))
        row = JSON.parse(line)
        piece, score = String(row[1]), row[2]
        id = index - 1
        if length(row) >= 3 && row[3] == "BYTE"
            byte = parse(UInt8, piece[4:5]; base = 16)
            bytes[Int(byte)+1] = id
            push!(pieces, [byte])
            continue
        end
        push!(pieces, collect(codeunits(piece)))
        startswith(piece, "<|plamo:") && (specials[piece] = id)
        node = 1
        for char in piece
            if !haskey(nodes[node].edges, char)
                push!(nodes, TokenNode())
                nodes[node].edges[char] = length(nodes)
            end
            node = nodes[node].edges[char]
        end
        nodes[node].id = id
        nodes[node].score = round(Int64, score * 1e4)
    end
    any(==(-1), bytes) && throw(ArgumentError("vocabulary is missing byte tokens"))
    threshold(key) = something(get(cfg, key, nothing), typemax(Int))
    return Tokenizer(
        nodes,
        pieces,
        specials,
        bytes,
        threshold("break_around_consecutive_spaces_threshold"),
        threshold("break_around_repeated_chars_threshold"),
    )
end

function encode_plain!(out, tokenizer, chars)
    n = length(chars)
    costs = fill(typemax(Int64) ÷ 4, n + 1)
    lengths = ones(Int, n)
    ids = fill(-1, n)
    costs[end] = 0
    for i = n:-1:1
        costs[i] = costs[i+1] + 10_000_000
        node = 1
        for j = i:n
            node = get(tokenizer.nodes[node].edges, chars[j], 0)
            node == 0 && break
            token = tokenizer.nodes[node]
            if token.id >= 0 && token.score > -20_000_000
                candidate = costs[j+1] - token.score
                if candidate < costs[i] || (candidate == costs[i] && j-i+1 >= lengths[i])
                    costs[i], lengths[i], ids[i] = candidate, j-i+1, token.id
                end
            end
        end
    end
    i = 1
    while i <= n
        if ids[i] >= 0
            push!(out, ids[i])
        else
            for byte in codeunits(string(chars[i]))
                push!(out, tokenizer.bytes[Int(byte)+1])
            end
        end
        i += lengths[i]
    end
    return out
end

function encode(tokenizer::Tokenizer, text::AbstractString)
    # Insert hard boundaries around special tokens and repeated runs, without
    # normalizing whitespace or Unicode. IDs remain zero-based across languages.
    chars = collect(text)
    out = Int[]
    start = i = 1
    while i <= length(chars)
        consumed_special = false
        if chars[i] == '<'
            j = i
            while j <= min(length(chars)-1, i+80)
                if chars[j] == '|' && chars[j+1] == '>'
                    candidate = String(chars[i:(j+1)])
                    if haskey(tokenizer.specials, candidate)
                        encode_plain!(out, tokenizer, @view(chars[start:(i-1)]))
                        push!(out, tokenizer.specials[candidate])
                        i = j+2
                        start = i
                        consumed_special = true
                        break
                    end
                end
                j += 1
            end
            consumed_special && continue
        end
        char = chars[i]
        stop = i+1
        while stop <= length(chars) && chars[stop] == char
            if char == '<' &&
               stop + 7 <= length(chars) &&
               String(chars[stop:(stop+7)]) == "<|plamo:"
                break
            end
            stop += 1
        end
        count = stop-i
        if char == '\uee00' ||
           (char == ' ' && count >= tokenizer.spaces) ||
           (char != '\n' && count >= tokenizer.repeats)
            encode_plain!(out, tokenizer, @view(chars[start:(i-1)]))
            char != '\uee00' && encode_plain!(out, tokenizer, @view(chars[i:(stop-1)]))
            start = stop
            i = stop
        else
            i += 1
        end
    end
    encode_plain!(out, tokenizer, @view(chars[start:end]))
    return out
end
