# DocstringTranslationPLaMoBackend.jl

## Description (Japanese)

[PLaMo](https://plamo.preferredai.jp/) という国産 LLM をバックエンドとするJuliaのマニュアル翻訳パッケージを開発しました．
このパッケージは CLI ツール [plamo-translate-cli](https://github.com/pfnet/plamo-translate-cli) をラップすることで翻訳機能を実現しています．

## Description (English)

(Built with PLaMo)

We have developed a manual translation package for Julia that utilizes [PLaMo](https://plamo.preferredai.jp/), a domestically developed LLM, as the backend.
This package implements translation functionality by wrapping the CLI tool [plamo-translate-cli](https://github.com/pfnet/plamo-translate-cli).

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
julia --project -e 'using DocstringTranslationPLaMoBackend; @switchlang! :Japanese; display(@doc sin)'
```
