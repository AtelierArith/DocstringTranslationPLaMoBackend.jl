# DocstringTranslationPLaMoBackend.jl

Julia の docstring と Documenter ページを、同梱した **PLaMo.jl のネイティブ Julia 実装**で翻訳します。Python・Conda・`plamo-translate` CLI は不要です。

This package translates Julia docstrings and Documenter pages using native Julia PLaMo inference. The PLaMo implementation is included in `src/PLaMo.jl` and `src/julia/`; installation does not require access to its private upstream repository. Model assets are cached with Scratch.jl, and one lazily loaded model is reused across translations.

## Setup

Julia **1.13 or newer** is required.

```sh
git clone https://github.com/AtelierArith/DocstringTranslationPLaMoBackend.jl.git
cd DocstringTranslationPLaMoBackend.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project install_model.jl
julia --threads=8 --project
```

```julia
using DocstringTranslationPLaMoBackend

@switchlang! :Japanese # :ja and "ja" also work
@doc sin

@switchlang! :English # :en and "en" also work
```

`@switchlang!` changes `Docs.parsedoc` and `Documenter.Page` for the current Julia process, as in the original package. Call it from your REPL or documentation build script, rather than during another package's precompilation. Already parsed docstrings can switch languages; the original Markdown remains available. Fenced Julia examples are not sent for translation.

## Model cache: Scratch.jl

```julia
using DocstringTranslationPLaMoBackend

install_model!()
DocstringTranslationPLaMoBackend.model_directory()
```

The installer downloads **PLaMo 3 Translate 0.8B**, pinned to model revision
`cc8e88169807d5a272e8ff06db12eb0129b67f81`, using Julia's `Downloads` stdlib. Each asset is checked against its recorded SHA-256 before atomic replacement. Subsequent installs reuse valid files and repair missing or corrupted files.

The model directory is managed by Scratch.jl under this package's UUID. With the default depot it resembles:

```text
~/.julia/scratchspaces/691b352d-20d8-448b-9a22-61c32af05849/plamo-3-translate-0.8b-<revision>/
```

It follows Julia's depot configuration; no model weights are stored in this repository, the package installation directory, or the working directory. Importing the package creates its Scratch directories but does not download or load the model. Run `install_model!()` explicitly before the first translation; loading then happens lazily once.

To copy an existing official checkpoint into the Scratch cache without another download:

```julia
install_model!(; source_directory="/path/to/plamo-3-translate-0.8b")
```

Equivalently:

```sh
julia --project install_model.jl /path/to/plamo-3-translate-0.8b
```

The imported files must match the pinned official revision, including both license files. They are verified and copied; the external directory does not become the runtime model location.

The downloaded assets occupy about **1.78 GB**. The default BF16 implementation retains approximately **2.73 GB of weight payload** on an x86 CPU with SSSE3, plus normalization, KV caches, workspace, allocator overhead and load-time temporaries. It keeps original BF16 values and uses F32 arithmetic; dictionary compression is lossless. Other Julia CPU architectures use the portable path, but actual inference has only been validated on Intel CPU.

The weights follow the [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en), independently of this package's MIT license. The installer also caches the original English and Japanese license files.

## Inference settings

```julia
configure_plamo!(;
    context_size=4096,
    max_tokens=1024,
    weight_storage=:bf16, # :f32 expands weights and uses more memory
)
```

Changing settings releases the cached translator; the next translation reloads it. `context_size` must accommodate the prompt plus `max_tokens`. Incomplete or empty translations raise an error and are not saved as completed documentation. For longer paragraphs/docstrings, increase both limits, for example `context_size=8192, max_tokens=2048`.

Native requests share one model and are serialized, including requests from threaded Markdown traversal. The PLaMo projection kernels can still use Julia threads. For CPU tuning you can separately set `LinearAlgebra.BLAS.set_num_threads(1)`; the package does not change global BLAS settings.

## Translation cache and Documenter

Translations also use Scratch.jl, in a separate `translation-plamo3-v3-<model-revision>` cache. Entries are keyed by original Markdown, package/version and normalized language code. The revision/schema namespace avoids mixing previous Python or older traversal results with the new model; old caches are not deleted. Absolute Documenter paths are kept inside Scratch rather than becoming absolute cache destinations.

```julia
using DocstringTranslationPLaMoBackend
import Documenter

DocstringTranslationPLaMoBackend.switchtargetpackage!(MyPackage)
@switchlang! :Japanese
Documenter.makedocs(; modules=[MyPackage])
```

Heading and table prose is translated while preserving heading levels, table alignment and code-only cells. Heading display text retains the original Documenter section ID (including explicit `@id` labels), so references within and across pages continue to resolve. Code blocks remain unchanged. Existing Markdown link protection and its fallback to the original paragraph are retained. Generated translations still need review before publishing documentation.

## Tests and implementation provenance

```sh
julia --project -e 'using Pkg; Pkg.test()'
```

The normal tests run without downloading a model or starting Python. They cover source integrity, Scratch configuration, checksum validation, atomic installation, offline reuse, language aliases, incomplete-output errors, translation caches, and the docstring/Documenter hooks. To run the optional real-model checks too:

```sh
PLAMO_SOURCE_DIR=/path/to/plamo-3-translate-0.8b \
  JULIA_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 \
  julia --project -e 'using Pkg; Pkg.test()'
```

The native sources were copied unchanged from PLaMo.jl at commit
`e919a8a13e18aa997ee0d7708cd35fdb00812fdc`. See [provenance and update instructions](src/julia/UPSTREAM.md), [source checksums](src/julia/checksums.toml), and the [upstream MIT license](src/julia/LICENSE). Runtime setup uses only registered Julia packages and stdlibs.
