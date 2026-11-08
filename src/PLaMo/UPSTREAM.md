# Vendored PLaMo.jl

Pure Julia inference sources from AtelierArith/plamo-rs, commit
`e919a8a13e18aa997ee0d7708cd35fdb00812fdc` (PLaMo 0.1.0).
Upstream: https://github.com/AtelierArith/plamo-rs

The Julia sources are licensed under the accompanying MIT LICENSE and kept
byte-for-byte identical to upstream, except that the `include` paths in
`PLaMo.jl` are rewritten from `julia/*.jl` to `*.jl` to match the flat layout
used here. Their SHA-256 checksums are recorded in `checksums.toml`, keyed by
paths relative to the parent package's `src/` directory. They are loaded as the
private submodule `DocstringTranslationPLaMoBackend.PLaMo`. This avoids
requiring access to the private upstream repository or an unregistered package
at install time. Runtime dependencies JSON, SIMD and LinearAlgebra are declared
by the parent package. No Rust, Python, model weights or upstream build
artifacts are included. Model assets have their own PLaMo Community License.

Layout:

```
src/PLaMo/
  PLaMo.jl              # module entry point (include paths adapted)
  tokenizer.jl prompt.jl weights.jl
  compressed_projection.jl packed_projection.jl projection.jl
  attention.jl model.jl translate.jl
  checksums.toml UPSTREAM.md LICENSE
```

To update, copy upstream `src/PLaMo.jl` and `src/julia/*.jl` into `src/PLaMo/`,
rewrite the `include` paths, update this revision, regenerate `checksums.toml`,
review dependency compatibility, and rerun the backend and real-model tests. Do
not edit vendored inference sources independently.
