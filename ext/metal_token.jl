# Preserve Base.argmax's first-index and NaN ordering on both GPU and host.
@inline token_precedes(value, index, best, best_index) =
    isless(best, value) || (isequal(value, best) && index < best_index)

function token_partials_kernel!(partials, logits, count)
    tid = Int(thread_index_in_threadgroup())
    lane = Int(thread_index_in_simdgroup())
    simd = Int(simdgroup_index_in_threadgroup())
    group = Int(threadgroup_position_in_grid_1d())
    best, index = -Inf32, typemax(Int32)
    shared = MtlThreadGroupArray(Tuple{Float32,Int32}, 8)
    start = (group - 1) * 1024
    for i = (start+tid):256:min(start+1024, count)
        value = @inbounds logits[i]
        if token_precedes(value, Int32(i), best, index)
            best, index = value, Int32(i)
        end
    end
    for offset in (16, 8, 4, 2, 1)
        value = simd_shuffle_down(best, offset)
        candidate = simd_shuffle_down(index, offset)
        if token_precedes(value, candidate, best, index)
            best, index = value, candidate
        end
    end
    lane == 1 && (@inbounds shared[simd] = (best, index))
    threadgroup_barrier(Metal.MemoryFlagThreadGroup)
    if simd == 1
        best, index = lane <= 8 ? (@inbounds shared[lane]) : (-Inf32, typemax(Int32))
        for offset in (16, 8, 4, 2, 1)
            value = simd_shuffle_down(best, offset)
            candidate = simd_shuffle_down(index, offset)
            if token_precedes(value, candidate, best, index)
                best, index = value, candidate
            end
        end
        lane == 1 && (@inbounds partials[group] = (best, index))
    end
    return
end

function gpu_token!(logits, partials)
    @metal threads=256 groups=length(partials) token_partials_kernel!(
        partials,
        logits,
        length(logits),
    )
    # One compact readback also synchronizes the request. The vocabulary scan
    # runs on the GPU; only one winner per 1024 logits is compared on the host.
    best, index = -Inf32, typemax(Int32)
    for (value, candidate) in Array(partials)
        if token_precedes(value, candidate, best, index)
            best, index = value, candidate
        end
    end
    return Int(index) - 1
end
