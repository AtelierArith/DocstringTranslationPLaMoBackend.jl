"""One previous source/translation pair, or one glossary entry."""
struct TranslationPair
    source::String
    target::String
end

"""Translation options shared with the Rust Request and C JSON schema. History is explicit."""
Base.@kwdef struct TranslationRequest
    text::String
    target_lang::String
    source_lang::Union{Nothing,String} = nothing
    context::Union{Nothing,String} = nothing
    glossary::Vector{TranslationPair} = TranslationPair[]
    history::Vector{TranslationPair} = TranslationPair[]
    form::Union{Nothing,Symbol} = nothing
    max_tokens::Int = 512
end

function check_text(text; empty = false)
    (!empty && isempty(strip(text))) && throw(ArgumentError("text must not be empty"))
    (occursin("<|plamo:", text) || occursin('\uee00', text)) &&
        throw(ArgumentError("text contains reserved PFML tokens or U+EE00"))
end
function check_language(code)
    (code == "ja(easy)" || occursin(r"^[a-z]{2,3}(-[a-z]{4})?$", code)) ||
        throw(ArgumentError("use lowercase language codes, e.g. ja, en, zh-hans"))
end

"""Build PFML3 exactly as the official translation chat template, including one BOS."""
function prompt(request::TranslationRequest)
    check_text(request.text)
    check_language(request.target_lang)
    isnothing(request.source_lang) || check_language(request.source_lang)
    1 <= request.max_tokens <= 32768 || throw(ArgumentError("max_tokens must be 1..32768"))
    if !isnothing(request.form)
        request.target_lang == "ja" && request.form in (:plain, :polite) || throw(
            ArgumentError("form must be :plain or :polite and requires Japanese output"),
        )
    end
    io = IOBuffer()
    print(io, "<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation")
    if !isnothing(request.context)
        check_text(request.context; empty = true)
        isempty(request.context) ||
            print(io, "<|plamo:tag|>context<|plamo:msg|>", request.context)
    end
    function input(text)
        print(io, "<|plamo:tag|>input")
        isnothing(request.source_lang) ||
            print(io, "<|plamo:key|>lang<|plamo:val|>", request.source_lang)
        print(io, "<|plamo:msg|>", text)
    end
    output() =
        print(io, "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>", request.target_lang)
    for pair in request.history
        check_text(pair.source)
        check_text(pair.target)
        input(pair.source)
        output()
        print(io, "<|plamo:msg|>", pair.target)
    end
    if !isempty(request.glossary)
        print(io, "<|plamo:tag|>glossary")
        for pair in request.glossary
            check_text(pair.source)
            check_text(pair.target)
            print(io, "<|plamo:index|>", pair.source, "<|plamo:item|>", pair.target)
        end
    end
    input(request.text)
    output()
    isnothing(request.form) || print(io, "<|plamo:key|>form<|plamo:val|>", request.form)
    print(io, "<|plamo:msg|>")
    return String(take!(io))
end
