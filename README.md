# DocstringTranslationPLaMoBackend.jl

## Description (Japanese)

この Julia パッケージは日本産の LLM [PLaMo](https://plamo.preferredai.jp/) をバックエンドとする Julia のマニュアルを翻訳する機能を提供します．
このパッケージは CLI ツール [plamo-translate-cli](https://github.com/pfnet/plamo-translate-cli) をラップすることで翻訳機能を実現しています．

## Description (English)

(Built with PLaMo)

This Julia package provides functionality to translate manuals for Julia by utilizing the Japanese-origin LLM [PLaMo] (https://plamo.preferredai.jp/) as its backend.
This package implements translation functionality by wrapping the CLI tool [plamo-translate-cli](https://github.com/pfnet/plamo-translate-cli).

## TL; DR

```julia
julia> using DocstringTranslationPLaMoBackend

julia> @switchlang! :Japanese
"Japanese"

julia> @doc sin
  sin(x)

  x の正弦を計算します。ここで x はラジアン単位です。

  関連項目: sind, sinpi, sincos, cis, asin.

  使用例
  ≡≡≡≡≡≡

  julia> round.(sin.(range(0, 2pi, length=9)'), digits=3)
  1×9 Matrix{Float64}:
   0.0  0.707  1.0  0.707  0.0  -0.707  -1.0  -0.707  -0.0

  julia> sind(45)
  0.7071067811865476

  julia> sinpi(1/4)
  0.7071067811865475

  julia> round.(sincos(pi/6), digits=3)
  (0.5, 0.866)

  julia> round(cis(pi/6), digits=3)
  0.866 + 0.5im

  julia> round(exp(im*pi/6), digits=3)
  0.866 + 0.5im

  sin(A::AbstractMatrix)

  正方行列 A の行列正弦を計算します。

  A が対称行列またはエルミート行列の場合、その固有分解（eigen）を用いて正弦を計算します。それ以外の場合は、exp を呼び出すことで正弦を算出します。

  使用例
  ≡≡≡≡≡≡

  julia> sin(fill(1.0, (2,2))))
  2×2 Matrix{Float64}:
   0.454649  0.454649
   0.454649  0.454649

```

## Setup

```
git clone https://github.com/AtelierArith/DocstringTranslationPLaMoBackend.jl.git
cd DocstringTranslationPLaMoBackend.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project install_model.jl
julia --project -e 'using DocstringTranslationPLaMoBackend; @switchlang! :Japanese; display(@doc sin)'
```
