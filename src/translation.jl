default_lang() = DEFAULT_LANG[]

const LANGUAGE_CODES = Dict(
    "japanese"=>"ja",
    "english"=>"en",
    "korean"=>"ko",
    "chinese"=>"zh-hans",
    "german"=>"de",
    "french"=>"fr",
    "spanish"=>"es",
    "italian"=>"it",
    "portuguese"=>"pt",
    "russian"=>"ru",
    "arabic"=>"ar",
    "vietnamese"=>"vi",
    "thai"=>"th",
    "indonesian"=>"id",
)

function language_code(lang::Union{AbstractString,Symbol})
    key = replace(lowercase(strip(String(lang))), '_'=>'-')
    code = get(LANGUAGE_CODES, key, key)
    (code == "ja(easy)" || occursin(r"^[a-z]{2,3}(-[a-z]{4})?$", code)) ||
        throw(ArgumentError("Use a language code such as ja/en, or Japanese/English."))
    code
end

function translation_request(doc, lang, max_tokens)
    request = PLaMo.TranslationRequest(
        text = string(doc),
        target_lang = language_code(lang),
        max_tokens = max_tokens,
    )
    PLaMo.prompt(request)
    request
end

function translated_markdown(result::PLaMo.Translation)
    result.finish_reason == :stop || throw(
        ErrorException(
            "PLaMo translation did not complete ($(result.finish_reason)); increase max_tokens/context_size with configure_plamo!().",
        ),
    )
    isempty(strip(result.text)) &&
        throw(ErrorException("PLaMo returned an empty translation."))
    Markdown.parse(result.text)
end

function translate_with_plamo(
    doc::Union{Markdown.MD,AbstractString};
    lang::Union{AbstractString,Symbol} = default_lang(),
)
    lock(MODEL_LOCK) do
        request = translation_request(doc, lang, MODEL_OPTIONS[].max_tokens)
        translated_markdown(PLaMo.translate(native_translator(), request))
    end
end

translate_documenter_md_with_plamo(doc::Union{Markdown.MD,AbstractString}) =
    translate_with_plamo(doc)
function translate_docstring_with_plamo(doc::Union{Markdown.MD,AbstractString})
    translate_md!(doc isa Markdown.MD ? copy_markdown(doc) : Markdown.parse(doc))
end

function _create_hex(l::Markdown.Link)
    (bytes2hex(codeunits(join(l.text))) * "_" * bytes2hex(codeunits(l.url)))
end

function _translate!(p::Markdown.Paragraph)
    hex2link = Dict()
    link2hex = Dict()
    content = map(p.content) do c
        # Protect Link so that it does not break during translation
        if c isa Markdown.Link
            h = _create_hex(c)
            hex2link[string(h)] = c
            link2hex[c] = h
            "`" * h * "`"
        else
            c
        end
    end
    p_orig = deepcopy(p)
    p.content = content
    result = translate_documenter_md_with_plamo(Markdown.MD(p))
    try
        translated_content = map(result[1].content) do c
            if c isa Markdown.Code
                if isempty(c.language)
                    if c.code in keys(hex2link)
                        _c = hex2link[c.code]
                        delete!(hex2link, c.code)
                        c = _c
                        c
                    else
                        c
                    end
                else
                    c
                end
            else
                c
            end
        end
        if isempty(hex2link)
            p.content = translated_content
        else
            @warn "Failed to translate by hex2link. Fallback to original content"
            p.content = p_orig.content
        end
    catch e
        @warn "Failed to translate by $(e)" p
        p.content = p_orig.content
    end
    nothing
end

function _translate!(list::Markdown.List)
    for item in list.items
        Base.Threads.@threads for i in item
            _translate!(i)
        end
    end
end

function _translate!(c)
    if hasproperty(c, :content)
        Base.Threads.@threads for c in c.content
            _translate!(c)
        end
    end
    c
end

"""
    translate_md!(md::Markdown.MD)

Translate a Markdown document in-place using native PLaMo.jl inference.

# Arguments
- `md::Markdown.MD`: The Markdown document to translate

# Returns
The translated Markdown document (same object as input)

# Details
Recursively translates all content in the Markdown document using multiple threads.
Translation is done in-place, modifying the original document.
"""
function translate_md!(md::Markdown.MD)
    Base.Threads.@threads for c in md.content
        _translate!(c)
    end
    md
end
