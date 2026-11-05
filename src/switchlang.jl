function _switchlang!(lang::Union{String,Symbol})
    DEFAULT_LANG[] = language_code(lang)
end

# Preserve the original document and its metadata. Translation must succeed
# before a language-specific cache entry is written.
function cached_translation(md::Markdown.MD; translator = translate_docstring_with_plamo)
    original_hash = hashmd(md)
    cache_original(md)
    translated = if istranslated(md)
        load_translation(md)
    else
        result = translator(copy_markdown(md))
        merge!(result.meta, md.meta)
        cache_translation(original_hash, result)
        result
    end
    merge!(translated.meta, md.meta)
    translated
end

function translated_docstr(d::DocStr)
    original = get!(d.data, :plamo_original_md) do
        md = Docs.formatdoc(d)
        md.meta[:module] = get(d.data, :module, Main)
        md.meta[:path] = get(d.data, :path, "")
        md
    end
    key = (default_lang(), TRANSLATION_CACHE_DIR[])
    if get(d.data, :plamo_translation_key, nothing) != key
        translated = cached_translation(original)
        d.object = translated
        d.data[:plamo_translation_key] = key
    end
    d.object
end

function translated_page(
    source::AbstractString,
    build::AbstractString,
    workdir::AbstractString,
)
    mdpage = Markdown.parse(replace(read(source, String), '\r' => ""))
    mdpage.meta[:path] = String(source)
    mdpage = cached_translation(mdpage; translator = translate_md!)
    mdast = convert(Documenter.MarkdownAST.Node, mdpage)
    Documenter.Page(source, build, workdir, mdpage.content, Documenter.Globals(), mdast)
end

const TRANSLATION_HOOKS_INSTALLED = Ref(false)
function install_translation_hooks!()
    TRANSLATION_HOOKS_INSTALLED[] && return
    @eval function Docs.parsedoc(d::DocStr)
        translated_docstr(d)
    end
    @eval function Documenter.Page(
        source::AbstractString,
        build::AbstractString,
        workdir::AbstractString,
    )
        translated_page(source, build, workdir)
    end
    TRANSLATION_HOOKS_INSTALLED[] = true
    nothing
end

"""
    @switchlang! lang

Enable automatic docstring/Documenter translation and select the output language.
Accepts codes (`:ja`, `"en"`) and the existing names (`:Japanese`, `:English`).
This overrides `Docs.parsedoc` and `Documenter.Page` for the current Julia process.
Translations are cached per language; changing languages reuses the original
text rather than retranslating an already translated docstring.
"""
macro switchlang!(lang)
    # Install at macro expansion time so the calling expression sees the methods
    # in its world age, matching the original interactive API.
    install_translation_hooks!()
    quote
        _switchlang!($(esc(lang)))
    end
end
