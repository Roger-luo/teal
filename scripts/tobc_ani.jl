using JLD2
using Printf
using CairoMakie
using Observables
using LaTeXStrings
using LinearAlgebra

exact_dir(xs...) = joinpath(@__DIR__, "..", "data", "exact", xs...)

shape = (2, :)
ts = 0.05:0.05:5.0
data = map(ts) do t
    d = jldopen(exact_dir("tobc_3_$t.jld2")) do file
        map([k for k in keys(file) if !(k in ("t_range", "T", "n_sites"))]) do k
            raw = abs.(real(file[k]))
		    raw = Symmetric(Matrix(map(x->ismissing(x) ? 0.0 : x, raw)))
            raw
        end
    end
    reshape(d, shape)
end

max_d = maximum(map(d->maximum(maximum, d), data))
min_d = minimum(map(d->minimum(minimum, d), data))
data_keys = map(Iterators.product(ntuple(_->(:+,:-), 3)...)) do signs
    join(signs)
end
data_keys = reshape(data_keys, shape)
figsize = (400 * size(data[1], 2), 400 * size(data[1], 1))

fig = Figure(size=figsize, fontsize=22, interpolate=true)
idx = Observable(1)
time = @lift ts[$idx]
current_data = @lift data[$idx]

for x in 1:size(current_data[], 1), y in 1:size(current_data[], 2)
    title_key = join(data_keys[x, y], ",")
    ax = Axis(
        fig[x, y];
        title=L"\chi_{\mathbf{i,t},\{; %$(title_key)\};}",
        xlabel=L"t_1",
        ylabel=L"t_2",
        xticks=0:1:5,
        yticks=0:1:5,
        titlesize=26,
    )
    heatmap!(
        ax, ts, ts, @lift($current_data[x, y]),
        colormap=Reverse(:Spectral_11),
        colorrange = (min_d, max_d),
        interpolate=false,
        rasterize=10,
    )
end


Colorbar(fig[:, end+1], colorrange = (min_d, max_d))
fig[0, :] = Label(fig, @lift(latexstring(@sprintf("t_3 = %1.2f", $time))))
fig

record(fig, "tobc_3.mp4", enumerate(ts), framerate=5) do (k, t)
    @show k
    idx[] = k
end
