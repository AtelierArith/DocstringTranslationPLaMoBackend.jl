# Keep linear weights in (input, output) column-major layout, so the
# safetensors row-major bytes can be read without an axis permutation.
function read_weights(path::AbstractString; dtypes = nothing, keep_bf16::Bool = false)
    tensors = Dict{String,Union{Array{Float32},Array{UInt16}}}()
    open(path) do io
        total = filesize(path)
        total >= 8 || throw(ArgumentError("truncated safetensors file"))
        n = ltoh(read(io, UInt64))
        0 < n <= min(total-8, 100_000_000) ||
            throw(ArgumentError("invalid safetensors header"))
        header = JSON.parse(String(read(io, Int(n))))
        base = 8 + Int(n)
        for (name, spec) in header
            name == "__metadata__" && continue
            shape = Tuple(reverse(Int.(spec["shape"])))
            all(>=(0), shape) || throw(ArgumentError("negative tensor dimension"))
            first, last = Int.(spec["data_offsets"])
            0 <= first <= last <= total-base ||
                throw(ArgumentError("invalid tensor offsets"))
            dtype = spec["dtype"]
            T =
                dtype == "BF16" ? UInt16 :
                dtype == "F16" ? Float16 :
                dtype == "F32" ? Float32 :
                throw(ArgumentError("unsupported weight dtype $dtype"))
            count = prod(big.(shape))
            count*sizeof(T) == last-first ||
                throw(ArgumentError("tensor size mismatch for $name"))
            seek(io, base+first)
            raw = read!(io, Vector{T}(undef, Int(count)))
            values = if dtype == "BF16" && keep_bf16
                ENDIAN_BOM==0x04030201 ? raw : ltoh.(raw)
            elseif dtype == "BF16"
                reinterpret(Float32, UInt32.(ltoh.(raw)) .<< 16)
            elseif dtype == "F16"
                Float32.(reinterpret(Float16, ltoh.(reinterpret(UInt16, raw))))
            else
                reinterpret(Float32, ltoh.(reinterpret(UInt32, raw)))
            end
            shaped=reshape(values, shape)
            tensors[String(name)] = shaped isa Array ? shaped : Array(shaped)
            isnothing(dtypes) || (dtypes[String(name)] = String(dtype))
        end
    end
    return tensors
end

function checkpoint_weights(directory; dtypes = nothing, keep_bf16::Bool = false)
    index = joinpath(directory, "model.safetensors.index.json")
    isfile(index) ||
        return read_weights(joinpath(directory, "model.safetensors"); dtypes, keep_bf16)
    filenames =
        sort!(unique!(String.(collect(values(JSON.parsefile(index)["weight_map"])))))
    weights = Dict{String,Union{Array{Float32},Array{UInt16}}}()
    for filename in filenames
        basename(filename) == filename && filename ∉ (".", "..") ||
            throw(ArgumentError("shard must be a filename"))
        merge!(weights, read_weights(joinpath(directory, filename); dtypes, keep_bf16))
    end
    return weights
end

widen_f32(values::Array{UInt16}) = reinterpret.(Float32, UInt32.(values) .<< 16)
widen_f32(values::Array{Float32}) = values
