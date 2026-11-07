module DocstringTranslationPLaMoBackend

using Base.Docs: DocStr

using SHA
using Markdown

using Scratch
import Documenter
import Downloads
include("PLaMo.jl")

const DEFAULT_LANG = Ref("ja")
const TRANSLATION_CACHE_DIR = Ref{String}()
const MODEL_CACHE_DIR = Ref{String}()
const DOCUMENTER_TARGET_PACKAGE = Ref{Dict{Symbol,Any}}()

function switchtranslationcachedir!(dir::AbstractString)
    TRANSLATION_CACHE_DIR[] = dir
end

function switchtargetpackage!(pkg::Module)
    DOCUMENTER_TARGET_PACKAGE[] =
        Dict(:name=>string(pkg), :version=>something(pkgversion(pkg), VERSION))
end

include("util.jl")
include("scratchspace.jl")
include("model.jl")
# Metal.jl only works on Apple Silicon; other platforms never load it.
@static if Sys.isapple() && Sys.ARCH === :aarch64
    include(joinpath("metal", "MetalBackend.jl"))
end
include("translation.jl")

include("switchlang.jl")
export @switchlang!
export configure_plamo!, install_model!

function __init__()
    scratch_name = "translation-plamo3-v3-" * MODEL_REVISION
    DOCUMENTER_TARGET_PACKAGE[] = Dict(:name=>"julia", :version=>VERSION)
    global TRANSLATION_CACHE_DIR[] = @get_scratch!(scratch_name)
    MODEL_CACHE_DIR[] = @get_scratch!("plamo-3-translate-0.8b-" * MODEL_REVISION)
end

end # module DocstringTranslationPLaMoBackend
