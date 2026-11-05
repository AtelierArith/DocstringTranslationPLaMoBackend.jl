# Vendored PLaMo.jl

Pure Julia inference sources from AtelierArith/plamo-rs, commit
`e919a8a13e18aa997ee0d7708cd35fdb00812fdc` (PLaMo 0.1.0).
Upstream: https://github.com/AtelierArith/plamo-rs

The Julia sources are unchanged and licensed under the accompanying MIT LICENSE.
Their SHA-256 checksums are recorded in checksums.toml. They are loaded as the
private submodule DocstringTranslationPLaMoBackend.PLaMo. This avoids requiring
access to the private upstream repository or an unregistered package at install
time. Runtime dependencies JSON, SIMD and LinearAlgebra are declared by the
parent package. The sources are placed in src/PLaMo.jl and src/julia/. No Rust, Python,
model weights or upstream build artifacts are included. Model assets have their own PLaMo Community License.

To update, copy upstream src/PLaMo.jl and src/julia/*.jl, update this revision,
regenerate checksums.toml, review dependency compatibility, and rerun the backend
and real-model tests. Do not edit vendored inference sources independently.
