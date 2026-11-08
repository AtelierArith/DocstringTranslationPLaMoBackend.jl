"""
Native Julia PLaMo 3 Translate inference, independent of Rust, Python and llama.cpp.
CPU projections use LinearAlgebra or native Julia SIMD for original BF16 weights. Assets are shared
with the Rust implementation; no remote Python code is loaded.
"""
module PLaMo

import JSON
using LinearAlgebra
using SIMD

export Translator, TranslationRequest, TranslationPair, Translation, translate, prompt

include("tokenizer.jl")
include("prompt.jl")
include("weights.jl")
include("compressed_projection.jl")
include("packed_projection.jl")
include("projection.jl")
include("attention.jl")
include("model.jl")
include("translate.jl")

end
