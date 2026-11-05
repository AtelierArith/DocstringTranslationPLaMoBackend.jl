using DocstringTranslationPLaMoBackend

directory =
    isempty(ARGS) ? install_model!() : install_model!(; source_directory = only(ARGS))
println("PLaMo model cached in Scratch: ", directory)
