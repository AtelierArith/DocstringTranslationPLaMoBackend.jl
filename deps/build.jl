using DocstringTranslationPLaMoBackend

source = get(ENV, "PLAMO_SOURCE_DIR", "")
directory = isempty(source) ? install_model!() : install_model!(; source_directory = source)
println("PLaMo model cached in Scratch: ", directory)
