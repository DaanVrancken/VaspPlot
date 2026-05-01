module VaspPlot

export extract_energies, extract_projections, extract_MLWFs, plot_bands, calc_hopping

using LinearAlgebra, Statistics, CairoMakie

# ============================================================
# §1  Array / index helpers
# ============================================================

"""
    remove_slices(arr, dim, indices_to_remove)

Return a copy of `arr` with the given slices removed along `dim`.
"""
function remove_slices(arr::AbstractArray, dim::Int64, indices_to_remove)
    keep = filter(i -> !(i in indices_to_remove), 1:size(arr, dim))
    return arr[ntuple(d -> d == dim ? keep : Colon(), ndims(arr))...]
end

"""
    orbs_to_indices(orbs)

Convert orbital names (e.g. `"s"`, `"t2g"`) or integers to a sorted,
unique list of 1-based orbital indices.
"""
function orbs_to_indices(orbs::Union{Vector{String}, Vector{Int64}})
    lut = Dict(
        "s"=>1, "py"=>2, "pz"=>3, "px"=>4, "p"=>[2,3,4],
        "dxy"=>5, "dyz"=>6, "dz2"=>7, "dxz"=>8, "x2-y2"=>9,
        "d"=>[5,6,7,8,9], "t2g"=>[5,6,8], "eg"=>[7,9],
        "fy3x2"=>10, "fxyz"=>11, "fyz2"=>12, "fz3"=>13,
        "fxz2"=>14, "fzx2"=>15, "fx3"=>16, "f"=>10:16)

    indices = Int[]
    for orb in orbs
        if orb isa Int
            push!(indices, orb)
        elseif !haskey(lut, orb)
            error("Orbital $orb not found in dictionary.")
        else
            v = lut[orb]
            v isa Int ? push!(indices, v) : append!(indices, v)
        end
    end
    return sort(unique(indices))
end

# ============================================================
# §2  k-path helpers
# ============================================================

"""
    find_klabels(kpoints, path_length, doubles)

Identify high-symmetry k-points and return their labels and (doubles-corrected) indices.
"""
function find_klabels(kpoints::Matrix{Float64}, path_length::Int64, doubles::Vector{Int64})
    N_labels = size(kpoints, 1) / path_length
    k_ind = vcat(1, [[i*path_length, i*path_length+1] for i in 1:N_labels]...)
    pop!(k_ind)
    k_ind = map(Int, setdiff(k_ind, doubles))

    standard_labels = Dict(
        "Γ"=>[0.0,0.0,0.0], "X"=>[0.5,0.0,0.0], "Y"=>[0.0,0.5,0.0],
        "Z"=>[0.0,0.0,0.5], "S"=>[0.5,0.5,0.0], "U"=>[0.5,0.0,0.5],
        "T"=>[0.0,0.5,0.5], "R"=>[0.5,0.5,0.5])

    klabels = fill(" ", length(k_ind))
    for (i, ki) in enumerate(k_ind), (label, coords) in standard_labels
        if isapprox(kpoints[ki, :], coords, atol=1e-10)
            klabels[i] = label; break
        end
    end

    # Correct indices for removed double points
    d = 0
    k_ind_orig = copy(k_ind)
    for i in eachindex(k_ind_orig)
        k_ind[i] -= d
        k_ind_orig[i]+1 in doubles && (d += 1)
    end
    return klabels, k_ind
end

"""
    merge_indices_and_labels(ind, lab)

Merge adjacent index/label pairs that differ by 1 into a single `"A|B"` entry.
"""
function merge_indices_and_labels(ind::Vector{Int}, lab::Vector{String})
    new_ind, new_lab = Float64[], eltype(lab)[]
    i = 1
    while i <= length(ind)
        if i < length(ind) && ind[i] == ind[i+1] - 1
            push!(new_ind, ind[i] + 0.5)
            push!(new_lab, lab[i] * "|" * lab[i+1])
            i += 2
        else
            push!(new_ind, ind[i]); push!(new_lab, lab[i])
            i += 1
        end
    end
    return new_ind, new_lab
end

"""
    find_path_lengths(path_edges, klabels)

Return Euclidean distances between consecutive high-symmetry points.
"""
function find_path_lengths(path_edges::Matrix{Float64}, klabels::Vector{String})
    path_lengths = zeros(length(klabels) - 1)
    d = 0
    for i in eachindex(path_lengths)
        # Only merged boundary labels like "A|B" skip one entry in path_edges.
        occursin("|", klabels[i]) && (d += 1)
        path_lengths[i] = norm(path_edges[i+1+d, :] - path_edges[i+d, :])
    end
    return path_lengths
end

# ============================================================
# §3  Extraction functions
# ============================================================

"""
    extract_energies(OUTCAR)

Parse a VASP OUTCAR to return a `Dict` with keys `"E"` (Nk × Nb × Nspin,
Fermi-shifted), `"klabels"`, `"k_ind"`, `"system"`, and `"path_edges"`.
"""
function extract_energies(OUTCAR::String)
    lines = readlines(OUTCAR)
    Nk = Nb = spin = path_length = i0 = 0
    system = ""
    Ef = [0.0, 0.0]
    reciprocal_vectors = zeros(3, 3)

    for i in eachindex(lines)
        if occursin("reciprocal lattice vectors", lines[i])
            for j in 1:3
                length(split(lines[i+j])) != 6 &&
                    @warn "Line $(i+j) does not have 6 elements; check reciprocal vectors."
                reciprocal_vectors[j, :] = parse.(Float64, split(lines[i+j])[end-2:end])
            end
            i0 = i + 4; break
        end
    end

    for i in i0:length(lines)
        if occursin("points on each line segment", lines[i])
            path_length = Int(parse(Float64, split(lines[i])[2]))
            i0 = i + 1; break
        end
    end
    path_length == 0 && error("'points on each line segment' not found in OUTCAR.")

    for i in i0:length(lines)
        if occursin("Dimension of arrays:", lines[i])
            Nk     = parse(Int, split(lines[i+1])[4])
            Nb     = parse(Int, last(split(lines[i+1])))
            system = last(split(lines[i+13]))
            i0 = i + 14; break
        end
    end
    Nk == 0 && error("'Dimension of arrays:' not found in OUTCAR.")

    for i in i0:length(lines)
        if occursin("Fermi energy:", lines[i])
            Ef[1] = parse(Float64, last(split(lines[i])))
            i0 = i; break
        end
    end

    if occursin("spin component 1", lines[i0+2])
        spin = 2
        i0  += 2
        Ef[2] = parse(Float64, split(lines[(Nk*(Nb+3)+3) + i0 - 2])[3])
    elseif occursin("k-point     1 :", lines[i0+2])
        spin = 1
    else
        error("'Fermi energy:' block not found in OUTCAR.")
    end

    kpoints = zeros(Nk, 3)
    E       = zeros(Nk, Nb, spin)
    for k in 1:Nk
        kpoints[k, :] = parse.(Float64, split(lines[(k-1)*(Nb+3) + i0+2])[4:6])
        for b in 1:Nb, s in 1:spin
            E[k, b, s] = parse(Float64,
                split(lines[b + (k-1)*(Nb+3) + (s-1)*(Nk*(Nb+3)+3) + i0+3])[2]) - Ef[s]
        end
    end

    doubles = [k for k in 2:Nk if isapprox(kpoints[k, :], kpoints[k-1, :], atol=1e-10)]
    klabels, k_ind = find_klabels(kpoints, path_length, doubles)
    kpoints = remove_slices(kpoints, 1, doubles)
    E       = remove_slices(E, 1, doubles)

    return Dict(
        "klabels"    => klabels,
        "k_ind"      => k_ind,
        "E"          => E,
        "system"     => system,
        "path_edges" => kpoints[k_ind, :] * reciprocal_vectors)
end

"""
    extract_projections(filename, ions, orbs)

Parse a VASP PROCAR and return an (Nk × Nb × Nspin) array of summed
projections for the given `ions` and `orbs`.
"""
function extract_projections(filename::String, ions::Vector{Int64},
                              orbs::Union{Vector{String}, Vector{Int64}})
    isfile(filename) || error("File not found: $filename")
    orb_indices = orbs_to_indices(orbs)

    # ── read header ──────────────────────────────────────────
    lines2 = readlines(filename)
    m = match(r"# of k-points:\s*(\d+)\s*# of bands:\s*(\d+)\s*# of ions:\s*(\d+)", lines2[2])
    m === nothing && error("Could not parse k-points/bands/ions from PROCAR header.")
    Nk, Nb = parse(Int, m[1]), parse(Int, m[2])

    # ── detect spin components ────────────────────────────────
    Nspin = 1
    after_tot = false
    for line in lines2
        if after_tot
            occursin(r"[^\s]", line) && (Nspin = 4)
            break
        end
        occursin("occ.", line)       && parse(Int, last(split(line))) == 1 && (Nspin = 2)
        startswith(strip(line), "tot") && (after_tot = true)
    end

    projections = zeros(Float64, Nk, Nb, Nspin)
    k = b = 0; s = 1

    open(filename) do io
        readline(io); readline(io); readline(io)   # skip 3-line header
        for line in eachline(io)
            sl = strip(line)
            startswith(sl, "# of k-points") && (s += 1; continue)
            if startswith(sl, "k-point")
                m2 = match(r"k-point\s+(\d+)\s*:", sl)
                m2 === nothing && error("Could not parse k-point index: $sl")
                k = parse(Int, m2[1]); b = 0
            elseif startswith(sl, "band")
                m2 = match(r"band\s+(\d+)\s*#", sl)
                m2 === nothing && error("Could not parse band index: $sl")
                b = parse(Int, m2[1])
            elseif any(n -> startswith(sl, string(n)*" "), ions)
                maximum(orb_indices) > length(split(sl)) - 2 &&
                    error("Orbital index out of range: $sl")
                projections[k, b, s] += sum(parse.(Float64, split(sl))[orb_indices .+ 1])
            end
        end
    end
    return projections
end

"""
    extract_MLWFs(wannier_file)

Return `(energies, kx)` where `energies` is an (NMLWFs x Nk) matrix and
`kx` is the length-Nk path coordinate as written by Wannier90.
"""
function extract_MLWFs(wannier_file::String)
    MLWFs  = Vector{Vector{Float64}}()
    kxs    = Vector{Vector{Float64}}()
    E_cur  = Float64[]
    kx_cur = Float64[]
    for line in readlines(wannier_file)
        if all(isspace, line)
            if !isempty(E_cur)
                push!(MLWFs, E_cur);  E_cur  = Float64[]
                push!(kxs,   kx_cur); kx_cur = Float64[]
            end
        else
            vals = parse.(Float64, split(line))
            push!(kx_cur, vals[1])
            push!(E_cur,  vals[2])
        end
    end
    # flush last block if file does not end with a blank line
    !isempty(E_cur) && (push!(MLWFs, E_cur); push!(kxs, kx_cur))

    Nw = length(MLWFs); Nk = length(MLWFs[1])
    out = zeros(Nw, Nk)
    for (i, m) in enumerate(MLWFs); out[i, :] = m; end
    return out, kxs[1]   # kx is identical for every WF block
end

function extract_MLWFs(wannier_1::String, wannier_2::String)
    E_up,   kx = extract_MLWFs(wannier_1)
    E_down, _  = extract_MLWFs(wannier_2)
    return Dict("up" => E_up, "down" => E_down), kx
end

# ============================================================
# §4  Plotting
# ============================================================

"""
    plot_bands(bands; proj, wann, ylims, path, double_plot, k_labels, spinor_component)

Save a band-structure PNG.  See README for argument details.
"""
function plot_bands(bands;
        proj             = nothing,
        wann             = nothing,   # Matrix/Dict, OR (Matrix/Dict, kx::Vector) tuple
        ylims            = (-5.0, 5.0),
        path             = "",
        double_plot      = false,
        k_labels::Vector{String}  = [" "],
        spinor_component::Int64   = 4)

    klabels = k_labels == [" "] ? bands["klabels"] : begin
        length(k_labels) == length(bands["k_ind"]) ||
            error("$(length(k_labels)) labels given, $(length(bands["k_ind"])) expected.")
        k_labels
    end

    # If automatic label detection misses points, keep ticks readable
    # by assigning deterministic fallback labels (K1, K2, ...).
    if k_labels == [" "]
        fallback_id = 1
        for i in eachindex(klabels)
            if isempty(strip(klabels[i]))
                klabels[i] = "K$(fallback_id)"
                fallback_id += 1
            end
        end
    end

    E      = bands["E"]
    system = bands["system"]
    k_ind, klabels = merge_indices_and_labels(bands["k_ind"], klabels)

    # ── build x-axis ─────────────────────────────────────────
    path_lengths = find_path_lengths(bands["path_edges"], klabels)
    X = zeros(size(E, 1))
    for i in eachindex(path_lengths)
        X[ceil(Int, k_ind[i]):floor(Int, k_ind[i+1])] .+=
            range(0, path_lengths[i], length = floor(Int, k_ind[i+1]) - ceil(Int, k_ind[i]) + 1)
        i != length(path_lengths) && (X[floor(Int, k_ind[i+1])+1:end] .+= path_lengths[i])
    end
    x_ticks = vcat(0, cumsum(path_lengths))

    # ── colorbar / proj setup ─────────────────────────────────
    bar_limits = (0, 1)
    if !isnothing(proj) && size(proj, 3) == 4
        proj       = copy(proj[:, :, spinor_component])
        bar_limits = (-1, 1)
    end

    # ── figure ───────────────────────────────────────────────
    inch       = 96;  pt = 4/3
    double_plot = double_plot && size(E, 3) == 2
    fig = Figure(
        size            = double_plot ? (2*3.40inch, 2.55inch) : (3.40inch, 2.55inch),
        fonts           = (; regular = "Dejavu", weird = "Blackchancery"),
        figure_padding  = (4, 4, 4, 8))

    function make_axis(col)
        Axis(fig[1, col];
            ylabel          = L"E - E_F \;\; \textrm{[eV]}",
            limits          = (first(X), last(X), ylims[1], ylims[2]),
            xticks          = (collect(Float64, x_ticks), klabels),
            xlabelsize      = 11pt, ylabelsize     = 11pt,
            xticklabelsize  = 8pt,  yticklabelsize = 8pt,
            xticksize       = 5pt,  yticksize      = 5pt,
            xtickalign = 1, ytickalign = 1,
            xminortickalign = 1, yminortickalign = 1,
            xminorticksvisible = false, xgridvisible = true,
            yminorticksvisible = true,  ygridvisible = false,
            xminorticks = IntervalsBetween(5), yminorticks = IntervalsBetween(5),
            xminorticksize = 2.5pt, yminorticksize = 2.5pt)
    end
    ax  = make_axis(1)
    ax2 = double_plot ? make_axis(!isnothing(proj) ? 3 : 2) : nothing
    hlines!(ax, 0.0, color=:grey41, linestyle=:dash, linewidth=1.5)
    !isnothing(ax2) && hlines!(ax2, 0.0, color=:grey41, linestyle=:dash, linewidth=1.5)

    # ── colorbars ─────────────────────────────────────────────
    if !isnothing(proj)
        cmap = bar_limits == (-1, 1) ? :RdBu_8 : :Blues_8
        Colorbar(fig[1, 2]; limits=bar_limits, colormap=cmap, flipaxis=true, ticklabelsize=8pt)
        if bar_limits != (-1, 1)
            if double_plot
                Colorbar(fig[1, 4]; limits=bar_limits, colormap=:Reds_8, flipaxis=true, ticklabelsize=8pt)
            elseif size(E, 3) == 2
                Colorbar(fig[1, 3]; limits=bar_limits, colormap=:Reds_8, flipaxis=true, ticklabelsize=8pt)
            end
        end
    end

    # ── band lines ────────────────────────────────────────────
    for b in axes(E, 2), s in axes(E, 3)
        panel    = (double_plot && s == 2) ? ax2 : ax
        Ec       = E[:, b, s]
        colormap = s == 1 ? :Blues_8 : :Reds_8
        color    = s == 1 ? RGBf(0.251, 0.388, 0.847) : RGBf(0.796, 0.235, 0.2)

        if !isnothing(proj)
            color    = proj[:, b, s]
            colormap = bar_limits == (-1, 1) ? :RdBu_8 : colormap
        end

        plot_fn = (!double_plot && s == 2) ? linesegments! : lines!
        plot_fn(panel, X, Ec; color=color, linestyle=:solid, linewidth=2,
                colormap=colormap, colorrange=isnothing(proj) ? nothing : bar_limits)
    end

    # ── Wannier overlay ───────────────────────────────────────
    if !isnothing(wann)
        # Unpack (data, kx) tuple if extract_MLWFs was called directly
        wann_data, kx_wann = wann isa Tuple ? wann : (wann, nothing)
        wann1 = wann_data isa Dict ? wann_data["up"]   : wann_data
        wann2 = wann_data isa Dict ? wann_data["down"] : wann_data

        # Rescale Wannier kx onto the DFT X grid.
        # Prefer exact reuse when both samplings have same number of k-points.
        if !isnothing(kx_wann)
            if length(kx_wann) == length(X)
                X_wann = copy(X)
            else
                # Segment boundaries in Wannier path: detect resets/non-monotonic steps.
                # This is robust for files that restart k-distance at each segment.
                diffs = diff(kx_wann)
                seg_ends = [i for i in eachindex(diffs) if diffs[i] <= 0]
                seg_bounds = vcat(1, seg_ends .+ 1, length(kx_wann) + 1)
                Nseg_wann = length(seg_bounds) - 1

                if Nseg_wann == length(path_lengths)
                    X_wann = zeros(length(kx_wann))
                    for i in 1:Nseg_wann
                        i0w = seg_bounds[i]
                        i1w = seg_bounds[i + 1] - 1
                        offset = i == 1 ? 0.0 : sum(path_lengths[1:i-1])
                        X_wann[i0w:i1w] .= offset .+ range(0, path_lengths[i], length = i1w - i0w + 1)
                    end
                else
                    # Conservative fallback: globally rescale to DFT x-range.
                    kmin, kmax = extrema(kx_wann)
                    scale = isapprox(kmax, kmin) ? 0.0 : (last(X) - first(X)) / (kmax - kmin)
                    X_wann = first(X) .+ (kx_wann .- kmin) .* scale
                end
            end
        else
            X_wann = X   # fallback: assume same grid
        end

        for w in axes(wann1, 1)
            scatter!(ax,  X_wann, wann1[w, :]; color=RGBf(0.22, 0.596, 0.149), markersize=3)
        end
        if double_plot
            for w in axes(wann2, 1)
                scatter!(ax2, X_wann, wann2[w, :]; color=RGBf(0.22, 0.596, 0.149), markersize=3)
            end
        end
    end

    suffix = isnothing(proj) ? (isnothing(wann) ? "_bands" : "_wann") : "_proj"
    save(joinpath(path, system * suffix * ".png"), fig; px_per_unit = 300/inch)
end

# ============================================================
# §5  Hopping / Wannier coefficient extraction
# ============================================================

function extract_excluded_bands(win::String)
    line = nothing
    for l in readlines(win)
        occursin(r"^\s*exclude_bands", l) && (line = l; break)
    end
    line === nothing && return Int[]

    clean = replace(line, r"^\s*exclude_bands\s*[:=]?\s*" => "")
    excluded = Int[]
    for part in split(clean, ",")
        part = strip(part)
        isempty(part) && continue
        if occursin("-", part)
            a, b = split(part, "-"); append!(excluded, parse(Int,a):parse(Int,b))
        else
            push!(excluded, parse(Int, part))
        end
    end
    return sort(unique(excluded))
end

function map_to_original_bands(bands_kept::Vector{Int}, excluded::Vector{Int})
    all_bands   = 1:maximum(vcat(bands_kept, excluded))
    non_excluded = setdiff(all_bands, excluded)
    return non_excluded[bands_kept]
end

function extract_energies_eig(wannier_eig::String, win::String, Nb_min::Int, Nb_max::Int)
    Nk = Nb_found = 0
    for line in eachline(wannier_eig)
        f = split(strip(line))
        Nb_found = max(Nb_found, parse(Int, f[1]))
        Nk       = max(Nk,       parse(Int, f[2]))
    end
    bands = map_to_original_bands(collect(1:Nb_found), extract_excluded_bands(win))
    (maximum(bands) < Nb_max || minimum(bands) > Nb_min) &&
        error("Bands in $wannier_eig don't match range $(Nb_min)–$(Nb_max) after exclusions.")
    E = zeros(Nk, Nb_max - Nb_min + 1)
    for line in eachline(wannier_eig)
        f = split(strip(line))
        b_idx = bands[parse(Int, f[1])] - Nb_min + 1
        E[parse(Int, f[2]), b_idx] = parse(Float64, f[3])
    end
    return E
end

function extract_coefficents(wanproj::String)
    Spin = Nk = Nb = Nw = 0
    Nb_max_up = Nb_max_down = 0
    Nb_min_up = Nb_min_down = typemax(Int)
    s = 0

    for line in eachline(wanproj)
        if (m = match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)) !== nothing
            Spin, Nk, Nb, Nw = parse.(Int, m.captures)
        elseif (m = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s*$", line)) !== nothing
            s = parse(Int, m[1])
        elseif (m = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s*$", line)) !== nothing
            b = parse(Int, m[1])
            if s == 1
                Nb_max_up  = max(Nb_max_up, b); Nb_min_up  = min(Nb_min_up, b)
            elseif s == 2
                Nb_max_down = max(Nb_max_down, b); Nb_min_down = min(Nb_min_down, b)
            else
                error("Unexpected spin value $s in $wanproj")
            end
        end
    end
    (Spin * Nk * Nb * Nw == 0 || Nb_min_up == typemax(Int)) &&
        error("Missing header info in $wanproj")

    Nb_max = max(Nb_max_up, Nb_max_down)
    Nb_min = min(Nb_min_up, Nb_min_down)
    coefficients = zeros(ComplexF64, Nk, Nb_max - Nb_min + 1, Nw, Spin)
    kpoints      = zeros(Float64, Nk, 3)
    k = 0; s = 0

    for line in eachline(wanproj)
        if (m = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s*$", line)) !== nothing
            k = mod1(k + 1, Nk); s = parse(Int, m[1])
            s == 1 && (kpoints[k, :] = parse.(Float64, m.captures[3:5]))
        elseif (m = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s+(-?\d+\.\d+[eE]?[-+]?\d*)\s*$", line)) !== nothing
            b, w = parse(Int, m[1]), parse(Int, m[2])
            coefficients[k, b - Nb_min + 1, w, s] = complex(parse(Float64, m[3]), parse(Float64, m[4]))
        end
    end
    return coefficients, kpoints, Nb_min_up, Nb_max_up, Nb_min_down, Nb_max_down
end

"""
    calc_hopping(neighbours; file_directory)

Compute hopping matrices for the given real-space `neighbours` (in fractional
coordinates) from WANPROJ / wannier90.eig data.
"""
function calc_hopping(
        neighbours::Vector{Vector{Int64}} = [[0,0,0],[1,0,0],[0,1,0],[0,0,1]];
        file_directory = ".")

    f(name) = joinpath(file_directory, name)
    required(path, msg) = isfile(path) || error(msg)

    required(f("WANPROJ"), "WANPROJ not found in $file_directory")

    spin_split = isfile(f("wannier90.1.win"))
    if spin_split
        required(f("wannier90.2.win"),  "wannier90.2.win not found"); 
        required(f("wannier90.1.eig"),  "wannier90.1.eig not found")
        required(f("wannier90.2.eig"),  "wannier90.2.eig not found")
    else
        required(f("wannier90.win"),    "No wannier90.win found in $file_directory")
        required(f("wannier90.eig"),    "No wannier90.eig found in $file_directory")
    end

    coefficients, kpoints, Nb_min_up, Nb_max_up, Nb_min_down, Nb_max_down =
        extract_coefficents(f("WANPROJ"))
    Nk, Nw, Spin = size(coefficients, 1), size(coefficients, 3), size(coefficients, 4)

    if spin_split
        E_up   = extract_energies_eig(f("wannier90.1.eig"), f("wannier90.1.win"), Nb_min_up,   Nb_max_up)
        E_down = extract_energies_eig(f("wannier90.2.eig"), f("wannier90.2.win"), Nb_min_down, Nb_max_down)
    else
        Spin == 2 && error("Spin-polarized WANPROJ but only one wannier90.eig present.")
        E_up = extract_energies_eig(f("wannier90.eig"), f("wannier90.win"), Nb_min_up, Nb_max_up)
    end

    t = Spin == 1 ? zeros(ComplexF64, Nw, Nw, length(neighbours)) :
                    zeros(ComplexF64, Nw, Nw, length(neighbours), Spin)

    for (R_idx, R) in enumerate(neighbours), i in 1:Nw, j in 1:Nw, k in 1:Nk
        phase = exp(2im * π * dot(kpoints[k, :], R))
        for s in 1:Spin
            E_s = s == 1 ? E_up : E_down
            Nb_s = size(E_s, 2)
            for b in 1:Nb_s
                c = coefficients[k, b, i, s]' * coefficients[k, b, j, s] * E_s[k, b] * phase
                Spin == 1 ? (t[i, j, R_idx] -= c) : (t[i, j, R_idx, s] -= c)
            end
        end
    end
    return t ./ Nk
end

end # module
