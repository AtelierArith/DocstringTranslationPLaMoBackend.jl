const MODEL_REPOSITORY = "pfnet/plamo-3-translate-0.8b"
const MODEL_REVISION = "cc8e88169807d5a272e8ff06db12eb0129b67f81"
# Official assets at the pinned revision; installation never executes remote code.
const MODEL_ASSETS = (
    "config.json" => "b825929ff52a9b94f911df21622cde1a2edf0a71c0710ab06b0dc60c83cacbd0",
    "tokenizer.jsonl" => "bdce9e430e65931b28c56de2054f52b4cf8ed59b322db125e016d724d06dbb0e",
    "tokenizer_config.json" => "5468ed5c68b5cf4cf971bcb34be6bfbfd749fe30cc5bc65ace5835ae3bff65b3",
    "LICENSE/en" => "d9979039b8f1a96fa81e1be61c4550d06f8542b25119e3f765c7c011a68597fb",
    "LICENSE/ja" => "988b37b0b21ca130ad84a092429e4e5bceec5f86dfe85a8ca3e06ea703b960ca",
    "model.safetensors" => "7c9f0066d166c075bdbe83ada3d444e08bd7a6ee3f345f47a1f3c442aa6898c5",
)
const MODEL_LOCK = ReentrantLock()
const NATIVE_TRANSLATOR = Ref{Union{Nothing,PLaMo.Translator}}(nothing)
const MODEL_OPTIONS = Ref((context_size = 4096, max_tokens = 1024, weight_storage = :bf16))

"""Return the revision-specific model cache managed by Scratch.jl."""
model_directory() = MODEL_CACHE_DIR[]

file_sha256(path) = open(path) do io
    bytes2hex(sha256(io))
end

function verify_model(directory, assets = MODEL_ASSETS)
    for (name, expected) in assets
        path = joinpath(directory, name)
        isfile(path) ||
            throw(ArgumentError("Missing model asset $name. Run install_model!() first."))
        file_sha256(path) == expected || throw(
            ArgumentError(
                "Model checksum mismatch: $name. Run install_model!() to repair the Scratch cache.",
            ),
        )
    end
    directory
end

function install_assets!(
    directory;
    assets = MODEL_ASSETS,
    source_directory = nothing,
    download = Downloads.download,
)
    mkpath(directory)
    for (name, expected) in assets
        path = joinpath(directory, name)
        isfile(path) && file_sha256(path) == expected && continue
        mkpath(dirname(path))
        mktemp(dirname(path)) do temporary, io
            close(io)
            if isnothing(source_directory)
                url = "https://huggingface.co/$MODEL_REPOSITORY/resolve/$MODEL_REVISION/$name"
                @info "Downloading PLaMo asset" name
                download(url, temporary)
            else
                cp(joinpath(source_directory, name), temporary; force = true)
            end
            file_sha256(temporary) == expected ||
                throw(ArgumentError("Downloaded model checksum mismatch: $name"))
            mv(temporary, path; force = true)
        end
    end
    directory
end

"""
    install_model!(; source_directory=nothing)

Cache the pinned official PLaMo 3 Translate 0.8B assets in Scratch.jl. Downloads
use Julia's Downloads stdlib and verify SHA-256 before atomic replacement.
Existing valid files are reused. To import an existing official checkpoint
without downloading it again, provide `source_directory`; verified files are
copied into Scratch. The model is not loaded until the first translation.
Model weights follow the PLaMo Community License shipped in the cache.
"""
function install_model!(; source_directory::Union{Nothing,AbstractString} = nothing)
    lock(MODEL_LOCK) do
        directory = install_assets!(model_directory(); source_directory)
        NATIVE_TRANSLATOR[] = nothing
        directory
    end
end

"""
    configure_plamo!(; context_size=4096, max_tokens=1024, weight_storage=:bf16)

Configure native Julia inference. The model stays in Scratch and loads lazily;
changing settings releases the cached translator. `:bf16` retains original BF16
weights; `:f32` uses expanded Float32 weights. Truncated translations raise an
error rather than being cached as complete documentation.
"""
function configure_plamo!(;
    context_size::Integer = 4096,
    max_tokens::Integer = 1024,
    weight_storage::Symbol = :bf16,
)
    1 <= context_size <= 32768 || throw(ArgumentError("context_size must be 1..32768"))
    1 <= max_tokens < context_size ||
        throw(ArgumentError("max_tokens must be positive and smaller than context_size"))
    weight_storage in (:bf16, :f32) ||
        throw(ArgumentError("weight_storage must be :bf16 or :f32"))
    lock(MODEL_LOCK) do
        MODEL_OPTIONS[] = (
            context_size = Int(context_size),
            max_tokens = Int(max_tokens),
            weight_storage = weight_storage,
        )
        NATIVE_TRANSLATOR[] = nothing
    end
    nothing
end

# Called under MODEL_LOCK; concurrent Markdown traversal shares one model and
# serializes requests instead of starting a model/process per paragraph.
function native_translator()
    if isnothing(NATIVE_TRANSLATOR[])
        directory = verify_model(model_directory())
        options = MODEL_OPTIONS[]
        NATIVE_TRANSLATOR[] = PLaMo.Translator(
            directory;
            context_size = options.context_size,
            weight_storage = options.weight_storage,
        )
    end
    NATIVE_TRANSLATOR[]::PLaMo.Translator
end
