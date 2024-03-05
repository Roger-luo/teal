# Genearte TOBCs at different checkpoints

using Yao
using Yao.EasyBuild
using LinearAlgebra
using Arpack: eigs
using Transducers
using JLD2
using ProgressMeter
BLAS.set_num_threads(1)

data_dir(paths::String...) = joinpath(@__DIR__, "..", "data", "exact", paths...)

function ad(A::Matrix, B::Matrix, sign::Symbol)
    if sign === :+
        return A * B + B * A
    else
        return A * B - B * A
    end
end

function ad(As::Tuple{Matrix}, B::Matrix, signs::Tuple{Symbol})
    return ad(As[1], B, signs[1])
end

function ad(As::NTuple{N, Matrix}, B::Matrix, signs::NTuple{N, Symbol}) where {N}
    return ad(As[1], ad(As[2:end], B, signs[2:end]), signs[1])
end

function tobc(n_sites::Int, T::Real, signs::NTuple{N, Symbol}, ts::NTuple{N, <:Real}) where {N}
    h = transverse_ising(n_sites, -1.0, periodic=false)
    H = Matrix(mat(h))
    st = statevec(zero_state(n_sites))
    rho = st * st'
    O = mat(chain(n_sites, put(1=>Z), put(2=>Z)))/4
    L = mat(chain(n_sites, put(n_sites => Z)))
    O_T = exp(im * T * H) * O * exp(-im * T * H)
    L_ts = map(ts) do t
        exp(im * t * H) * L * exp(-im * t * H)
    end
    return real(tr(rho * ad(L_ts, O_T, signs)))
end

function scan_tobc(n_sites::Int, T::Real, t_range, ts::NTuple{N, <:Real}) where {N}
    isdir(data_dir()) || mkdir(data_dir())
    filename = join(("tobc", "$(N+2)", ts...), "_") * ".jld2"
    data_file = data_dir(filename)
    jldopen(data_file, "w") do file
        file["n_sites"] = n_sites
        file["T"] = T
        file["t_range"] = t_range
    end
    @show ts
    for signs in Iterators.product(ntuple(_->(:+, :-), N+2)...)
        result = zeros(Union{Float64, Missing}, length(t_range), length(t_range))
        f = MapSplat() do (i_1, t_1), (i_2, t_2)
            t_1 ≤ t_2 || return (i_1, i_2), missing
            return (i_1, i_2), tobc(n_sites, T, signs, (t_1, t_2, ts...))
        end

        total_task_itr = Iterators.partition(Iterators.product(enumerate(t_range), enumerate(t_range)), 2 * Threads.nthreads())
        p = Progress(length(total_task_itr); desc=join(signs), showspeed=true)
        for task in total_task_itr
            for ((x, y), value) in tcollect(f, task)
                result[x, y] = value
            end
            next!(p)
        end

        jldopen(data_file, "a") do file
            file[join(signs)] = result
        end
    end
end

# ts = map(ARGS) do arg
#     parse(Float64, arg)
# end

# scan_tobc(5, 5.0, 0.0:1e-2:5.0, (ts...,))
for t in 0.2:0.05:0.45
    scan_tobc(5, 5.0, 0.0:1e-2:5.0, (t, ))
end

for t in 0.55:0.05:1.45
    scan_tobc(5, 5.0, 0.0:1e-2:5.0, (t, ))
end

for t in 1.55:0.05:2.45
    scan_tobc(5, 5.0, 0.0:1e-2:5.0, (t, ))
end

# scan_tobc(5, 5.0, 0.0:1e-2:5.0, (2.5, ))
# scan_tobc(5, 5.0, 0.0:1e-2:5.0, (3.0, ))
# scan_tobc(5, 5.0, 0.0:1e-2:5.0, (3.5, ))
# scan_tobc(5, 5.0, 0.0:1e-2:5.0, (4.0, ))
# scan_tobc(5, 5.0, 0.0:1e-2:5.0, (4.5, ))
# scan_tobc(5, 5.0, 0.0:1e-2:5.0, (5.0, ))


