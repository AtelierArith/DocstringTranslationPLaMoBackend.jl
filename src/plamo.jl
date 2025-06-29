function default_lang()
    return DEFAULT_LANG[]
end

function translate_docstring_with_plamo(doc::Union{Markdown.MD,AbstractString})
    translate_with_plamo(doc)
end

function translate_documenter_md_with_plamo(doc::Union{Markdown.MD,AbstractString})
    translate_with_plamo(doc)
end

function translate_with_plamo(
    doc::Union{Markdown.MD,AbstractString};
    lang::String = default_lang(),
)
    content = CondaPkg.withenv() do
        read(`plamo-translate --to $(lang) --input $(doc)`, String)
    end
    return Markdown.parse(content)
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

Translate a Markdown document in-place using OpenAI translation.

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
