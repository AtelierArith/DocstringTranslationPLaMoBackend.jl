# DocstringTranslationPLaMoBackend.jl

## TL; DR

```julia
julia> using DocstringTranslationPLaMoBackend
julia> @switchlang! :Japanese
julia> @doc sin
```

## Setup

```
git clone https://github.com/AtelierArith/DocstringTranslationPLaMoBackend.jl.git
cd DocstringTranslationPLaMoBackend.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project install_model.jl
julia --project -e 'using DocstringTranslationPLaMoBackend; @switchlang! :Japanese; @doc sin'
```