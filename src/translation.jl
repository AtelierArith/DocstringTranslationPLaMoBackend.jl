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

function _translate!(p::Markdown.Paragraph; translator = translate_documenter_md_with_plamo)
    isempty(strip(string(Markdown.MD(p)))) && return nothing
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
    result = translator(Markdown.MD(p))
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

function _translate!(
    header::Markdown.Header;
    translator = translate_documenter_md_with_plamo,
)
    content = header.text isa AbstractVector ? copy(header.text) : Any[header.text]
    # Match Documenter's heading slug calculation before changing display text.
    # An explicit label already supplies the stable destination; translate its
    # contents rather than protecting the entire link as paragraph prose does.
    labeled =
        length(content) == 1 &&
        content[1] isa Markdown.Link &&
        occursin(r"^@id (.+)$", content[1].url)
    destination = if labeled
        content[1].url
    else
        "@id " * Documenter.slugify(sprint(Markdown.plain, Markdown.Paragraph(content)))
    end
    if labeled
        content = deepcopy(content[1].text)
    end
    paragraph = Markdown.Paragraph(content)
    _translate!(paragraph; translator)
    header.text = Any[Markdown.Link(paragraph.content, destination)]
    header
end

function _translate!(table::Markdown.Table; translator = translate_documenter_md_with_plamo)
    for row in table.rows, index in eachindex(row)
        content = row[index] isa AbstractVector ? copy(row[index]) : Any[row[index]]
        # An identifier-only or empty cell has no prose to translate.
        all(
            node ->
                node isa Markdown.Code || (node isa AbstractString && isempty(strip(node))),
            content,
        ) && continue
        paragraph = Markdown.Paragraph(content)
        _translate!(paragraph; translator)
        row[index] = paragraph.content
    end
    table
end

_translate!(code::Markdown.Code) = code

# GPU inference is serialized by MODEL_LOCK. Do not create Julia worker tasks
# for a traversal that only feeds the same Metal device.
function foreach_translation(f, items)
    if metal_selected()
        foreach(f, items)
    else
        Base.Threads.@threads for i in eachindex(items)
            f(items[i])
        end
    end
    return nothing
end

function _translate!(list::Markdown.List)
    for item in list.items
        foreach_translation(_translate!, item)
    end
end

function _translate!(c)
    if hasproperty(c, :content)
        foreach_translation(_translate!, c.content)
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
Recursively translates Markdown content, using Julia threads for the CPU backend.
Translation is done in-place, modifying the original document.
"""
function translate_md!(md::Markdown.MD)
    foreach_translation(_translate!, md.content)
    md
end
