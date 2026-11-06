# Binary serialization of Arb matrices (the SDP caches sdp*.jls). Arblib serializes every entry as a type tag plus
# a hex string (arb_dump_str) and parses it back on load: the (14,18) cache (9.2 GB) took ~25 min to load and the
# (18,20) cache was still being written after 4 h. Here an arb_mat_struct is written as raw words: per entry the
# arf exponent, size word and mantissa limbs (inline limbs, or the heap limbs copied), then the mag exponent and
# mantissa. Loading stores inline entries directly and rebuilds heap mantissas with arf_set_mpn + arf_mul_2exp_si.
# Matrices with an exponent too large for a small fmpz fall back to the string format (tagged :k5str1). Files
# written by plain Arblib (tuple header) still load; K5_SER_OLD[] records that an old-format matrix was read.
using ClusteredLowRankSolver, Serialization
if !isdefined(@__MODULE__, :K5_SER_OLD)
const KS = ClusteredLowRankSolver.Arblib
const K5_SER_OLD = Ref(false)
const _ARBWORDS = sizeof(KS.arb_struct) ÷ 8        # 6: arf (exp, size, 2 limbs) + mag (exp, man)
@assert _ARBWORDS == 6
_small(x::UInt) = (x >> 62) != 0x1                 # fmpz: an mpz pointer has top bits 01
@inline _entry(v::KS.arb_mat_struct, i, j) = Ptr{UInt}(unsafe_load(v.rows, i) + (j - 1) * sizeof(KS.arb_struct))

function Serialization.serialize(s::Serialization.AbstractSerializer, v::KS.arb_mat_struct)
    r, c = size(v)
    ok = true
    for i = 1:r, j = 1:c
        p = _entry(v, i, j)
        if !_small(unsafe_load(p, 1)) || !_small(unsafe_load(p, 5))
            ok = false; break
        end
    end
    Serialization.serialize_type(s, typeof(v))
    if !ok
        Serialization.serialize(s, :k5str1)
        Serialization.serialize(s, (r, c))
        for i = 1:r, j = 1:c
            Serialization.serialize(s, unsafe_load(v[i, j]))
        end
        return
    end
    Serialization.serialize(s, :k5bin1)
    Serialization.serialize(s, (r, c))
    buf = UInt[]
    sizehint!(buf, 6 * r * c)
    for i = 1:r, j = 1:c
        p = _entry(v, i, j)
        sz = unsafe_load(p, 2); n = Int(sz >> 1)
        push!(buf, unsafe_load(p, 1), sz)
        if n <= 2
            push!(buf, unsafe_load(p, 3), unsafe_load(p, 4))
        else
            limbs = Ptr{UInt}(unsafe_load(p, 4))
            for k = 1:n
                push!(buf, unsafe_load(limbs, k))
            end
        end
        push!(buf, unsafe_load(p, 5), unsafe_load(p, 6))
    end
    write(s.io, Int64(length(buf)))
    write(s.io, buf)
    return
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, T::Type{KS.arb_mat_struct})
    h = Serialization.deserialize(s)
    if h isa Tuple || h === :k5str1
        h isa Tuple ? (K5_SER_OLD[] = true) : nothing
        r, c = h isa Tuple ? h : Serialization.deserialize(s)
        res = T(r, c)
        for i = 1:r, j = 1:c
            res[i, j] = Serialization.deserialize(s)
        end
        return res
    end
    h === :k5bin1 || error("k5ser: unknown arb_mat format tag $h")
    r, c = Serialization.deserialize(s)
    len = read(s.io, Int64)
    buf = Vector{UInt}(undef, len)
    read!(s.io, buf)
    res = T(r, c)
    k = 1
    GC.@preserve buf res begin
        for i = 1:r, j = 1:c
            p = _entry(res, i, j)
            ex = buf[k]; sz = buf[k+1]; n = Int(sz >> 1)
            if n <= 2
                unsafe_store!(p, ex, 1); unsafe_store!(p, sz, 2)
                unsafe_store!(p, buf[k+2], 3); unsafe_store!(p, buf[k+3], 4)
                k += 4
            else
                ccall((:arf_set_mpn, KS.libarb), Cvoid, (Ptr{UInt}, Ptr{UInt}, Int, Cint), p, pointer(buf, k + 2), n, Cint(sz & 1))
                ccall((:arf_mul_2exp_si, KS.libarb), Cvoid, (Ptr{UInt}, Ptr{UInt}, Int), p, p, reinterpret(Int, ex) - 64 * n)
                k += 2 + n
            end
            unsafe_store!(p, buf[k], 5); unsafe_store!(p, buf[k+1], 6)
            k += 2
        end
    end
    k == len + 1 || error("k5ser: length mismatch ($k vs $(len + 1))")
    return res
end
end
