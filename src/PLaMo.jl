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

include("julia/tokenizer.jl")
include("julia/prompt.jl")
include("julia/weights.jl")
include("julia/compressed_projection.jl")
include("julia/packed_projection.jl")
include("julia/projection.jl")
include("julia/attention.jl")
include("julia/model.jl")
include("julia/translate.jl")

end
