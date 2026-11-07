using Test
using DocstringTranslationPLaMoBackend
using Markdown
using SHA
using TOML
import Documenter

const B = DocstringTranslationPLaMoBackend
const Native = B.PLaMo

@testset "Native source and lazy Scratch configuration" begin
    @test B.default_lang() == "ja"
    @test isnothing(B.NATIVE_TRANSLATOR[])
    @test isdir(B.model_directory())
    @test occursin(string(Base.PkgId(B).uuid), B.model_directory())
    @test occursin(B.MODEL_REVISION, B.model_directory())
    @test !startswith(B.model_directory(), pkgdir(B))
    source = joinpath(pkgdir(B), "src")
    checksums = TOML.parsefile(joinpath(source, "julia", "checksums.toml"))["sha256"]
    @test length(checksums) >= 10
    for (file, digest) in checksums
        @test B.file_sha256(joinpath(source, file)) == digest
    end
    project = TOML.parsefile(joinpath(pkgdir(B), "Project.toml"))
    @test !haskey(project["deps"], "PythonCall")
    @test !haskey(project["deps"], "CondaPkg")
    @test !isfile(joinpath(pkgdir(B), "CondaPkg.toml"))
    configure_plamo!(; context_size = 512, max_tokens = 32, weight_storage = :f32)
    @test B.MODEL_OPTIONS[] == (context_size = 512, max_tokens = 32, weight_storage = :f32, device = :auto)
    @test isnothing(B.NATIVE_TRANSLATOR[])
    @test_throws ArgumentError configure_plamo!(; context_size = 0)
    @test_throws ArgumentError configure_plamo!(; context_size = 32769)
    @test_throws ArgumentError configure_plamo!(; max_tokens = 0)
    @test_throws ArgumentError configure_plamo!(; context_size = 128, max_tokens = 128)
    @test_throws ArgumentError configure_plamo!(; weight_storage = :f16)
    configure_plamo!()
    @testset "Inference device selection" begin
        configure_plamo!(; device = :cpu)
        @test B.MODEL_OPTIONS[].device === :cpu
        configure_plamo!(; device = :auto)
        @test B.MODEL_OPTIONS[].device === :auto
        configure_plamo!(; device = :metal)
        @test B.MODEL_OPTIONS[].device === :metal
        caller_thread = Threads.threadid()
        observed_threads = Vector{Int}(undef, 32)
        B.foreach_translation(1:32) do index
            observed_threads[index] = Threads.threadid()
        end
        @test all(==(caller_thread), observed_threads)
        @test_throws ArgumentError configure_plamo!(; device = :cuda)
        configure_plamo!()
    end
end

@testset "Language compatibility and native request contract" begin
    for (language, code) in [
        (:Japanese, "ja"),
        ("English", "en"),
        (:ja, "ja"),
        (" JA ", "ja"),
        ("zh_Hans", "zh-hans"),
        ("ja(easy)", "ja(easy)"),
    ]
        @test B.language_code(language) == code
        request = B.translation_request("Translate this sentence.", language, 64)
        @test request isa Native.TranslationRequest
        @test request.target_lang == code
        @test request.max_tokens == 64
        @test occursin("Translate this sentence.", Native.prompt(request))
    end
    @test_throws ArgumentError B.translate_with_plamo("Hello"; lang = "../../outside")
    @test_throws ArgumentError B.translate_with_plamo("  "; lang = "ja")
    @test_throws ArgumentError B.translate_with_plamo("<|plamo:tag|>"; lang = "ja")
    @test isnothing(B.NATIVE_TRANSLATOR[])
    result(text, finish) = Native.Translation(text, finish, 21, 7, 0.1, 0.4)
    @test occursin("翻訳", string(B.translated_markdown(result("翻訳です。", :stop))))
    @test_throws ErrorException B.translated_markdown(result("途中", :length))
    @test_throws ErrorException B.translated_markdown(result("中断", :cancelled))
    @test_throws ErrorException B.translated_markdown(result("  ", :stop))
    code = "```julia\nx = sin(1.0)\n```"
    @test occursin("x = sin(1.0)", string(B.translate_docstring_with_plamo(code)))
    @test isnothing(B.NATIVE_TRANSLATOR[])
end

@testset "Build accepts an offline model source" begin
    mktempdir() do directory
        stub = joinpath(directory, "DocstringTranslationPLaMoBackend", "src")
        mkpath(stub)
        write(joinpath(stub, "DocstringTranslationPLaMoBackend.jl"), """
            module DocstringTranslationPLaMoBackend
            export install_model!
            function install_model!(; source_directory = nothing)
                write(ENV["PLAMO_BUILD_MARKER"], string(source_directory))
                return "cache"
            end
            end
            """)
        marker = joinpath(directory, "source-used")
        source = joinpath(directory, "offline-checkpoint")
        env = copy(ENV)
        env["JULIA_LOAD_PATH"] = directory * (Sys.iswindows() ? ";" : ":") * "@stdlib"
        env["PLAMO_SOURCE_DIR"] = source
        env["PLAMO_BUILD_MARKER"] = marker
        build = joinpath(pkgdir(B), "deps", "build.jl")
        run(pipeline(setenv(`$(Base.julia_cmd()) --startup-file=no $build`, env); stdout = devnull))
        @test read(marker, String) == source
    end
end

@testset "Verified model cache, offline reuse and atomic recovery" begin
    mktempdir() do directory
        content = Dict(
            "config.json"=>"{\"native\":true}",
            "LICENSE/en"=>"model license",
            "model.safetensors"=>"test bytes",
        )
        assets =
            [name => bytes2hex(sha256(text)) for (name, text) in sort!(collect(content))]
        calls = String[]
        download =
            (url, path) -> begin
                push!(calls, url)
                file = split(url, B.MODEL_REVISION * "/"; limit = 2)[2]
                write(path, content[file])
            end
        @test B.install_assets!(directory; assets, download) == directory
        @test length(calls) == 3
        @test all(url->occursin(B.MODEL_REVISION, url), calls)
        @test B.verify_model(directory, assets) == directory
        B.install_assets!(
            directory;
            assets,
            download = (args...)->error("must reuse cache"),
        )
        write(joinpath(directory, "config.json"), "corrupt")
        @test_throws ArgumentError B.verify_model(directory, assets)
        B.install_assets!(directory; assets, download)
        @test length(calls) == 4
        @test B.verify_model(directory, assets) == directory
        rm(joinpath(directory, "config.json"))
        @test_throws ArgumentError B.verify_model(directory, assets)
        @test_throws ArgumentError B.install_assets!(
            directory;
            assets,
            download = (url, path)->write(path, "wrong"),
        )
        @test !isfile(joinpath(directory, "config.json"))
        @test sort(readdir(directory)) == ["LICENSE", "model.safetensors"]
        B.install_assets!(directory; assets, download)
        mktempdir() do imported
            B.install_assets!(
                imported;
                assets,
                source_directory = directory,
                download = (args...)->error("no network"),
            )
            @test B.verify_model(imported, assets) == imported
            @test read(joinpath(imported, "model.safetensors"), String) == "test bytes"
        end
    end
end

@testset "Heading and table prose with preserved code" begin
    md = Markdown.parse(
        "# Examples\n\n| Name | Meaning |\n|:---|---:|\n| `x` | Input |\n\n```julia\nf(x) = x\n```\n",
    )
    heading, table, code = md.content
    original_alignment = copy(table.align)
    original_code = code.code
    calls = String[]
    translated =
        Dict("Examples"=>"使用例", "Name"=>"名前", "Meaning"=>"意味", "Input"=>"入力")
    translator = doc -> begin
        text = strip(string(doc))
        push!(calls, text)
        Markdown.parse(translated[text])
    end
    B._translate!(heading; translator)
    B._translate!(table; translator)
    B._translate!(code)
    @test heading isa Markdown.Header{1}
    @test occursin("使用例", string(Markdown.MD(heading)))
    @test occursin("名前", string(Markdown.MD(table)))
    @test occursin("意味", string(Markdown.MD(table)))
    @test occursin("入力", string(Markdown.MD(table)))
    @test table.align == original_alignment
    @test length(table.rows) == 2
    @test length.(table.rows) == [2, 2]
    @test table.rows[2][1][1] isa Markdown.Code
    @test table.rows[2][1][1].code == "x"
    @test code.code == original_code
    @test sort(calls) == sort(collect(keys(translated)))
    empty = Markdown.Table([Any[Any[], Any[Markdown.Code("", "y")]]], [:l, :r])
    B._translate!(
        empty;
        translator = doc->error("empty and code-only cells must be skipped"),
    )
    @test isempty(empty.rows[1][1])
    @test empty.rows[1][2][1].code == "y"
end

@testset "Translation cache and safe absolute document paths" begin
    old = B.TRANSLATION_CACHE_DIR[]
    oldlang = B.default_lang()
    try
        mktempdir() do directory
            B.switchtranslationcachedir!(directory)
            md = Markdown.parse("Source documentation.")
            md.meta[:module] = Main
            md.meta[:path] = "fixture.jl"
            calls = Ref(0)
            translator = doc -> begin
                calls[] += 1
                Markdown.parse("翻訳された説明。")
            end
            B._switchlang!(:Japanese)
            translated = B.cached_translation(md; translator)
            @test calls[] == 1
            @test string(md) == "Source documentation.\n"
            @test translated.meta[:module] === Main
            @test B.istranslated(md)
            @test isfile(joinpath(B.pathofcachedir(md), B.hashmd(md), "ja.md"))
            @test string(B.cached_translation(md; translator)) == string(translated)
            @test calls[] == 1
            B._switchlang!(:English)
            @test !B.istranslated(md)
            B.cached_translation(
                md;
                translator = doc->Markdown.parse("English documentation."),
            )
            @test occursin("English", string(B.load_translation(md)))
            B._switchlang!(:Japanese)
            @test occursin("翻訳", string(B.load_translation(md)))
            absolute = Markdown.parse("A page.")
            absolute.meta[:path] = joinpath(directory, "docs", "page.md")
            @test startswith(B.pathofcachedir(absolute), directory)
            traversal = Markdown.parse("Another page.")
            traversal.meta[:path] = "../../outside/page.md"
            @test !startswith(relpath(B.pathofcachedir(traversal), directory), "..")
            @test B.prevminor(v"1.0.3") == v"1.0.0"
            @test B.insertversion(["Package", "Module"], v"1.2.3") ==
                  ["Package", "1.2", "Module"]
            failure = Markdown.parse("Never cache truncated output.")
            failure.meta[:module] = Main
            @test_throws ErrorException B.cached_translation(
                failure;
                translator = doc->error("incomplete translation"),
            )
            @test !B.istranslated(failure)
        end
    finally
        B.switchtranslationcachedir!(old)
        B._switchlang!(oldlang)
    end
end

if haskey(ENV, "PLAMO_SOURCE_DIR")
    @testset "Real native PLaMo through Scratch" begin
        directory = install_model!(; source_directory = ENV["PLAMO_SOURCE_DIR"])
        @test directory == B.model_directory()
        configure_plamo!(; max_tokens = 64)
        output = B.translate_with_plamo("The meeting starts at ten."; lang = :Japanese)
        @test strip(string(output)) == "会議は10時に始まります。"
        first_model = B.NATIVE_TRANSLATOR[]
        @test first_model isa Native.Translator
        @test strip(
            string(B.translate_with_plamo("The meeting starts at ten."; lang = :ja)),
        ) == "会議は10時に始まります。"
        @test B.NATIVE_TRANSLATOR[] === first_model
        structured = B.translate_docstring_with_plamo(
            "# Examples\n\n| Meaning |\n|---|\n| Input |\n\n```julia\nf(x) = x\n```\n",
        )
        @test !occursin("Examples", string(structured.content[1].text[1].text))
        @test structured.content[1].text[1].url == "@id Examples"
        @test !occursin("Meaning", string(Markdown.MD(structured.content[2])))
        @test occursin("f(x) = x", string(structured))
        configure_plamo!(; max_tokens = 1)
        @test_throws ErrorException B.translate_with_plamo(
            "The meeting starts at ten.";
            lang = :Japanese,
        )
        configure_plamo!()
    end
end

@testset "Public docstring and Documenter hooks use language caches" begin
    old = B.TRANSLATION_CACHE_DIR[]
    oldlang = B.default_lang()
    try
        mktempdir() do directory
            B.switchtranslationcachedir!(directory)
            doc = Base.Docs.docstr(
                "Original documentation.",
                Dict{Symbol,Any}(:module=>Main, :path=>"testfixture.jl"),
            )
            original = Base.Docs.formatdoc(doc)
            original.meta[:module] = Main
            original.meta[:path] = "testfixture.jl"
            doc.object = B.copy_markdown(original) # Already parsed before switching language.
            for (language, text) in
                [(:Japanese, "日本語の説明。"), (:English, "English explanation.")]
                B._switchlang!(language)
                translated = Markdown.parse(text)
                merge!(translated.meta, original.meta)
                B.cache_translation(B.hashmd(original), translated)
            end
            @test (@switchlang! :Japanese) == "ja"
            @test occursin("日本語", string(Base.Docs.parsedoc(doc)))
            @test (@switchlang! :English) == "en"
            @test occursin("English explanation", string(Base.Docs.parsedoc(doc)))
            @test occursin("Original", string(doc.data[:plamo_original_md]))
            @switchlang! :Japanese
            @test occursin("日本語", string(Base.Docs.parsedoc(doc)))
            source = joinpath(directory, "page.md")
            write(source, "Page source.\r\n")
            page_original = Markdown.parse("Page source.\n")
            page_original.meta[:path] = source
            translated = Markdown.parse("ページの説明。")
            translated.meta[:path] = source
            B.cache_translation(B.hashmd(page_original), translated)
            page = Documenter.Page(source, joinpath(directory, "build"), directory)
            @test page isa Documenter.Page
            @test occursin("ページ", string(Markdown.MD(page.elements)))
            @test startswith(B.pathofcachedir(page_original), directory)
        end
    finally
        B.switchtranslationcachedir!(old)
        B._switchlang!(oldlang)
    end
end

@testset "Translated Documenter section references" begin
    old = B.TRANSLATION_CACHE_DIR[]
    oldmodel = B.MODEL_CACHE_DIR[]
    try
        mktempdir() do directory
            B.switchtranslationcachedir!(joinpath(directory, "cache"))
            # A cache miss must fail instead of invoking locally installed weights.
            B.MODEL_CACHE_DIR[] = joinpath(directory, "no-model")
            source = joinpath(directory, "src")
            mkpath(source)
            pages = [
                "index.md" => """
                # Examples

                See [Examples](@ref), [Details](@ref), and [named](@ref stable-id).

                ## Details

                ### [Named section](@id stable-id)

                ```julia
                x = :Examples
                ```
                """,
                "other.md" => """
                # Other page

                See [Examples](@ref), [Details](@ref), and [named](@ref stable-id).
                Also see [direct](index.md#Examples).
                """,
            ]
            translations = Dict(
                "Examples" => "使用例",
                "Details" => "詳細",
                "Named section" => "名前付きの節",
                "Other page" => "別のページ",
            )
            translator = doc -> Markdown.parse(translations[strip(string(doc))])
            for (file, text) in pages
                path = joinpath(source, file)
                write(path, text)
                original = Markdown.parse(text)
                # makedocs changes into root and passes source-relative paths.
                original.meta[:path] = joinpath("src", file)
                translated = B.cached_translation(
                    original;
                    translator = md -> begin
                        for node in md.content
                            node isa Markdown.Header && B._translate!(node; translator)
                        end
                        md
                    end,
                )
                # Exercise serialized cache reload, as the Page hook will do.
                reloaded = B.load_translation(original)
                @test string(reloaded) == string(translated)
                if file == "index.md"
                    headings = filter(node -> node isa Markdown.Header, reloaded.content)
                    @test headings[1] isa Markdown.Header{1}
                    @test headings[2] isa Markdown.Header{2}
                    @test headings[3] isa Markdown.Header{3}
                    @test [h.text[1].url for h in headings] == ["@id Examples", "@id Details", "@id stable-id"]
                    @test headings[3].text[1].text == ["名前付きの節"]
                    @test reloaded.content[end].code == "x = :Examples"
                    # Repeated translation keeps the original ID.
                    B._translate!(headings[1]; translator = doc -> doc)
                    @test headings[1].text[1].url == "@id Examples"
                end
            end
            Documenter.makedocs(;
                root = directory,
                sitename = "Offline translated references",
                pages = ["index.md", "other.md"],
                format = Documenter.HTML(
                    prettyurls = false,
                    edit_link = nothing,
                    repolink = nothing,
                    inventory_version = "0.0.0",
                ),
                remotes = nothing,
                doctest = false,
            )
            index = read(joinpath(directory, "build", "index.html"), String)
            other = read(joinpath(directory, "build", "other.html"), String)
            @test occursin("使用例", index)
            @test occursin("詳細", index)
            @test occursin("名前付きの節", index)
            for id in ["Examples", "Details", "stable-id"]
                @test occursin("id=\"$id\"", index)
                @test occursin("href=\"#$id\"", index)
                @test occursin("href=\"index.html#$id\"", other)
            end
            @test isnothing(B.NATIVE_TRANSLATOR[])
        end
    finally
        B.switchtranslationcachedir!(old)
        B.MODEL_CACHE_DIR[] = oldmodel
    end
end
