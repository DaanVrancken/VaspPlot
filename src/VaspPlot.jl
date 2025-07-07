module VaspPlot

export extract_energies, extract_projections, extract_MLWFs, plot_bands, calc_hopping

using CairoMakie

"""
    remove_slices(arr::AbstractArray, dim::Int64, indices_to_remove)

Removes slices from an N-dimensional array along a specified dimension.

# Arguments
- `arr::AbstractArray`: The input N-dimensional array.
- `dim::Int`: The dimension along which to remove slices.
- `indices_to_remove`: An iterable (e.g., Vector, Set) of indices to remove along `dim`.

# Returns
- A new array with the specified slices removed.

# Examples
```julia
julia> A = [1 2 3; 4 5 6; 7 8 9]
3×3 Matrix{Int64}:
 1  2  3
 4  5  6
 7  8  9

julia> remove_slices(A, 1, [2])
2×3 Matrix{Int64}:
 1  2  3
 7  8  9

julia> remove_slices(A, 2, [1, 3])
3×1 Matrix{Int64}:
 2
 5
 8
```
"""
function remove_slices(arr::AbstractArray, dim::Int64, indices_to_remove)
    all_indices = 1:size(arr, dim)
    indices_to_keep = filter(i -> !(i in indices_to_remove), all_indices)
    slices = ntuple(d -> d == dim ? indices_to_keep : Colon(), ndims(arr))
    return arr[slices...]
end

"""
    find_klabels(kpoints, path_length::Int64)

Identifies high-symmetry k-point labels (e.g., Γ, X, Y) from a list of k-points.
This function is specifically tailored for VASP band structure calculations where
k-points are generated along high-symmetry lines.

# Arguments
- `kpoints`: A matrix where each row represents a k-point coordinate (e.g., `Nk x 3` matrix).
- `path_length::Int`: The number of steps (k-points) along each high-symmetry path segment 
                        as defined in the VASP KPOINTS file.

# Returns
- `klabels::Vector{String}`: A vector of strings containing the identified high-symmetry
                              labels or empty strings if no standard label is found.
- `k_ind::Vector{Int}`: A vector of indices in the `kpoints` array corresponding to the
                        positions where labels are assigned.

# Notes
- The function uses a small tolerance (`atol=1e-10`) for floating-point comparisons
  to identify k-points.
- Currently supports common high-symmetry points: Γ, X, Y, Z, S, U, T, R.
"""
function find_klabels(kpoints, path_length::Int64)
    steps = path_length-1
    N_labels = (size(kpoints,1)-1) ÷ steps
    k_ind = 1:steps:(1+N_labels*steps)
    klabels = collect(" " for _ in 1:length(k_ind))

    for i in k_ind
        if isapprox(kpoints[i,:], [0.0 0.0 0.0]', atol=1e-10)
            klabels[Int((i-1)/steps + 1)] = "Γ"
        elseif isapprox(kpoints[i,:], [0.5 0 0]', atol=1e-10)
            klabels[Int((i-1)/steps + 1)] = "X"
        elseif isapprox(kpoints[i,:], [0 0.5 0]', atol=1e-10)   
            klabels[Int((i-1)/steps + 1)] = "Y"
        elseif isapprox(kpoints[i,:], [0 0 0.5]', atol=1e-10)
            klabels[Int((i-1)/steps + 1)] = "Z"
        elseif isapprox(kpoints[i,:], [0.5 0.5 0]', atol=1e-10)
            klabels[Int((i-1)/steps + 1)] = "S"
        elseif isapprox(kpoints[i,:], [0.5 0 0.5]', atol=1e-10)
            klabels[Int((i-1)/steps + 1)] = "U"
        elseif isapprox(kpoints[i,:], [0 0.5 0.5]', atol=1e-10)
            klabels[Int((i-1)/steps + 1)] = "T"
        elseif isapprox(kpoints[i,:], [0.5 0.5 0.5]', atol=1e-10)
            klabels[Int((i-1)/steps + 1)] = "R"
        end
    end

    return klabels, k_ind
end

"""
    orbs_to_indices(orbs::Union{Vector{String}, Vector{Int64}})

Converts a vector of orbital names (strings) or direct orbital indices (integers)
into a sorted, unique list of numerical orbital indices.

# Arguments
- `orbs::Union{Vector{String}, Vector{Int64}}`: A vector containing orbital names
  (e.g., "s", "px", "dxy", "t2g") or integer indices (e.g., 1, 2, 5).
  Supported string orbitals and their corresponding 1-based indices:
  - "s": 1
  - "py": 2, "pz": 3, "px": 4, "p": [2, 3, 4]
  - "dxy": 5, "dyz": 6, "dz2": 7, "dxz": 8, "x2-y2": 9, "d": [5, 6, 7, 8, 9]
  - "t2g": [5, 6, 8], "eg": [7, 9]
  - "fy3x2": 10, "fxyz": 11, "fyz2": 12, "fz3": 13, "fxz2": 14, "fzx2": 15, "fx3": 16,
    "f": [10, 11, 12, 13, 14, 15, 16]

# Returns
- `indices::Vector{Int}`: A sorted vector of unique integer orbital indices.

# Throws
- `ErrorException`: If an orbital name is not found in the predefined dictionary.

# Examples
```julia
julia> orbs_to_indices(["s", "pz", "dxy"])
3-element Vector{Int64}:
 1
 3
 5

julia> orbs_to_indices([1, "p", 7])
4-element Vector{Int64}:
 1
 2
 3
 4
 7
```
"""
function orbs_to_indices(orbs::Union{Vector{String}, Vector{Int64}})
    orb_indices = Dict("s"=>1, "py" => 2, "pz"=>3, "px"=>4, "p" => [2,3,4],
                    "dxy"=>5, "dyz"=>6, "dz2"=>7, "dxz"=>8, "x2-y2"=>9, "d"=>[5,6,7,8,9],
                    "t2g"=>[5,6,8], "eg"=>[7,9],
                    "fy3x2"=>10, "fxyz"=>11, "fyz2"=>12, "fz3"=>13, "fxz2"=>14, "fzx2"=>15, "fx3"=>16,
                    "f"=>[10,11,12,13,14,15,16])

    indices = []
    for orb in orbs
        if typeof(orb) == Int
            push!(indices, orb)
        elseif !haskey(orb_indices, orb)
            error("Orbital $orb not found in dictionary.")
        else
            if typeof(orb_indices[orb]) == Int
                push!(indices, orb_indices[orb])
            else
                append!(indices, orb_indices[orb])
            end
        end
    end

    return sort(unique(indices))
end

"""
    extract_energies(OUTCAR::String)

Extracts band energies, k-points, Fermi energy, and system information from a VASP OUTCAR file.
The extracted energies are shifted such that the Fermi energy is at 0 eV.

# Arguments
- `OUTCAR::String`: The path to the VASP OUTCAR file.

# Returns
- `Dict`: A dictionary containing the extracted data:
    - `"klabels"::Vector{String}`: High-symmetry k-point labels.
    - `"k_ind"::Vector{Int}`: Indices of high-symmetry k-points.
    - `"E"::Array{Float64, 3}`: Band energies (k-points x bands x spin components),
                                 relative to the Fermi energy.
    - `"system"::String`: The system name extracted from the OUTCAR.

# Throws
- `ErrorException`: If critical information (e.g., "points on each line segment",
                    "Dimension of arrays:", "Fermi energy:") is not found in the OUTCAR.
- `ErrorException`: If the OUTCAR structure for spin components is unexpected.

# Notes
- This function assumes a specific format for the OUTCAR file, typical for VASP band structure calculations.
- It automatically handles single-spin and spin-polarized calculations.
- Duplicate k-points (often present at high-symmetry points) are removed.
"""
function extract_energies(OUTCAR::String)
    lines = readlines(OUTCAR)

    Nk = 0
    Nb = 0
    system = ""
    spin = 1
    Ef = [0.0, 0.0]
    path_length = 0
    i0 = 0

    for i in eachindex(lines)
        if occursin("points on each line segment", lines[i])
            path_length = Int(parse.(Float64, split(lines[i])[2]))
            i0 = i+1
            break
        end
    end
    if path_length == 0
        error("Line with 'points on each line segment' not found in OUTCAR.")
    end

    for i in i0:length(lines)
        if occursin("Dimension of arrays:", lines[i])
            Nk = parse.(Int, split(lines[i+1])[4])
            Nb = parse.(Int, last(split(lines[i+1])))
            system = last(split(lines[i+13]))
            i0 = i+14
            break
        end
    end
    if Nk == 0
        error("Line with 'Dimension of arrays:' not found in OUTCAR.")
    end

    for i in i0:length(lines)
        if occursin("Fermi energy:", lines[i])
            Ef[1] = parse.(Float64, last(split(lines[i])))
            i0 = i
            break
        end
    end

    if occursin("spin component 1", lines[i0+2])
        spin = 2
        i0 = i0+2
        Ef[2] = parse.(Float64, split(lines[(Nk*(Nb+3)+3) + i0-2])[3])
    elseif !occursin("k-point     1 :", lines[i0+2])
        error("Line with 'Fermi energy:' not found in OUTCAR.")
    end

    kpoints = zeros(Nk, 3)
    E = zeros(Nk, Nb, spin)

    for k in range(1, Nk)
        kpoints[k, :] = parse.(Float64, split(lines[(k-1)*(Nb+3) + i0+2])[4:6])
        for b in range(1, Nb)
            for s in range(1, spin)
                E[k, b, s] = parse(Float64, split(lines[b + (k-1)*(Nb+3) + (s-1)*(Nk*(Nb+3)+3) + i0+3])[2]) - Ef[s]
            end
        end
    end

    doubles = []
    for k in range(2, Nk)
        if isapprox(kpoints[k,:], kpoints[k-1,:], atol=1e-10)
            push!(doubles, k)
        end
    end
    kpoints = remove_slices(kpoints, 1, doubles)
    E = remove_slices(E, 1, doubles)

    klabels, k_ind = find_klabels(kpoints, path_length)

    return Dict("klabels" => klabels, "k_ind" => k_ind, "E" => E, "system" => system)
end

"""
    extract_projections(filename::String, ions::Vector{Int64}, orbs::Union{Vector{String}, Vector{Int64}})

Extracts projected band contributions from a VASP PROCAR-like file.

# Arguments
- `filename::String`: The path to the PROCAR-like file (e.g., PROCAR).
- `ions::Vector{Int64}`: A vector of 1-based indices of ions for which to sum projections.
- `orbs::Union{Vector{String}, Vector{Int64}}`: A vector of orbital names (e.g., "s", "p", "d")
                                               or direct orbital indices (1-based) to sum.
                                               See `orbs_to_indices` for supported orbital names.

# Returns
- `projections::Array{Float64, 3}`: An array of projected weights with dimensions
                                    (k-points x bands x spin components).
                                    For non-spin-polarized calculations, spin component dimension is 1.
                                    For spin-polarized, it's 2 (spin-up, spin-down).
                                    For non-collinear, it's 4 (spin-up, spin-down, spin-x, spin-y).

# Throws
- `ErrorException`: If the specified `filename` does not exist.
- `ErrorException`: If header information (k-points, bands, ions) cannot be parsed.
- `ErrorException`: If k-point or band indices cannot be parsed from the file.
- `ErrorException`: If orbital indices exceed the number of available orbitals in a line.

# Notes
- This function assumes a specific format for the PROCAR-like file.
- It detects the number of spin components based on the file content.
"""
function extract_projections(filename::String, ions::Vector{Int64}, orbs::Union{Vector{String}, Vector{Int64}})
    if !isfile(filename)
        error("File not found: $filename")
    end

    num_k_points = 0
    num_bands = 0
    num_spin_components = 1

    current_k_point_idx = 0
    current_band_idx = 0
    current_spin_idx = 1

    projections = Array{Float64, 3}(undef, 0, 0, 0)
    total_band_projection = 0.0

    orb_indices = orbs_to_indices(orbs)

    open(filename, "r") do io
        header_lines = [readline(io) for _ in 1:3]

        # Extract system information from the header
        if length(header_lines) >= 2
            match_params = match(r"# of k-points:\s*(\d+)\s*# of bands:\s*(\d+)\s*# of ions:\s*(\d+)", header_lines[2])
            if match_params !== nothing
                num_k_points = parse(Int, match_params.captures[1])
                num_bands = parse(Int, match_params.captures[2])
            else
                error("Could not parse k-points, bands, and ions from header line 2.")
            end
        else
            error("Header line 2 does not contain all information.")
        end
        flag = false
        for (i, line) in enumerate(eachline(filename))
            if occursin("occ.", line)
                occ = Int(parse(Float64, last(split(line))))
                if occ == 1
                    num_spin_components = 2
                end
            end
            if flag
                if occursin(r"[^\s]", line)
                    num_spin_components = 4
                end
                break
            end
            if startswith(strip(line), "tot")
                flag = true
            end
        end

        projections = zeros(Float64, num_k_points, num_bands, num_spin_components)

        for line in eachline(io)
            # Delete extra whitespace
            stripped_line = strip(line)

            # Go to next spin component
            if startswith(stripped_line, "# of k-points")
                current_spin_idx +=1
            end

            # Detect k-punt section
            if startswith(stripped_line, "k-point")
                match_kpt = match(r"k-point\s+(\d+)\s*:", stripped_line)
                if match_kpt !== nothing
                    current_k_point_idx = parse(Int, match_kpt.captures[1])
                    # Reset band index
                    current_band_idx = 0
                else
                    error("Could not parse k-point index from line: $stripped_line")
                end
            # Detect band section
            elseif startswith(stripped_line, "band")
                match_band = match(r"band\s+(\d+)\s*#", stripped_line)
                if match_band !== nothing
                    current_band_idx = parse(Int, match_band.captures[1])
                    total_band_projection = 0.0
                else
                    error("Could not parse band index from line: $stripped_line")
                end
            # Add projections
            elseif any(n -> startswith(stripped_line, string(n)*" "), ions)
                if maximum(orb_indices) > length(split(stripped_line)) - 2
                    error("Orbital indices exceed number of orbitals in line: $stripped_line")
                end
                projections[current_k_point_idx, current_band_idx, current_spin_idx] += sum(parse.(Float64, split(stripped_line))[orb_indices.+1])
            end
        end
    end

    return projections
end

"""
    extract_MLWFs(wannier_file::String)

Extracts Maximally Localized Wannier Functions (MLWFs) data from a Wannier90 `.eig` or similar output file.
This function assumes the file contains energy values for each MLWF, separated by blank lines.

# Arguments
- `wannier_file::String`: The path to the Wannier90 output file (e.g., `wannier90.eig`).

# Returns
- `MLWFs_array::Matrix{Float64}`: A 2D array where each row represents an MLWF and
                                   columns represent the energy values at different k-points.

# Examples
```julia
# Assuming 'wannier90.eig' exists with appropriate content
# MLWFs_data = extract_MLWFs("wannier90.eig")
# size(MLWFs_data) # (number of MLWFs, number of k-points)
```
"""
function extract_MLWFs(wannier_file::String)
    lines = readlines(wannier_file)

    MLWFs = []
    MLWF = []
    for l in lines
        if all(isspace, l)
            push!(MLWFs, MLWF)
            MLWF = []
            continue
        end
        push!(MLWF, parse.(Float64, split(l))[2])
    end

    MLWFs_array = zeros(length(MLWFs), length(MLWFs[1]))
    for i in eachindex(MLWFs)
        MLWFs_array[i, :] = MLWFs[i]
    end

    return MLWFs_array
end

"""
    plot_bands(bands; proj=nothing, wann=nothing, ylims=(-5.0,5.0), path="", double_plot=false, k_labels::Vector{String}=[" "], spinor_component::Int64=4)

Plots electronic band structures, optionally with projected contributions or Wannier functions.

# Arguments
- `bands::Dict`: A dictionary containing band structure data, typically from `extract_energies`.
                 Expected keys: `"klabels"`, `"k_ind"`, `"E"`, `"system"`.
- `proj::Union{Nothing, AbstractArray}`: Optional. An array of projected weights (k-points x bands x spin components),
                                         typically from `extract_projections`. If provided, bands will be colored
                                         according to these projections.
- `wann::Union{Nothing, AbstractMatrix}`: Optional. A matrix of Maximally Localized Wannier Function (MLWF) energies
                                          (MLWFs x k-points), typically from `extract_MLWFs`. If provided, MLWFs
                                          will be overlaid as scatter points.
- `ylims::Tuple{Float64, Float64}`: Optional. Y-axis limits for the plot (energy range). Default is `(-5.0, 5.0)`.
- `path::String`: Optional. The directory path where the plot image will be saved. Default is current directory.
- `double_plot::Bool`: Optional. If `true` and `bands["E"]` has two spin components, it will create two separate
                       panels for spin-up and spin-down bands. Default is `false`.
- `k_labels::Vector{String}`: Optional. Custom k-point labels to use instead of those extracted from `bands`.
                              Must match the number of symmetry points (`length(bands["k_ind"])`).
                              Default uses labels from `bands`.
- `spinor_component::Int64`: Optional. Relevant for non-collinear spin calculations (`proj` with 4 spin components).
                             Specifies which spinor component (1 to 4) to plot. Default is `4`.

# Returns
- `Nothing`: The function saves the plot to a file and does not return a value.

# Side Effects
- Creates a PNG image file named `bands_SYSTEM.png` in the specified `path`.

# Dependencies
- Requires `Makie.jl` and `Colors.jl` (for `RGBf`).
- Assumes `L` for LaTeX strings is available (e.g., from `LaTeXStrings.jl`).

# Examples
```julia
# Assuming 'band_data' is a dictionary from extract_energies
# plot_bands(band_data)

# Plot with projections and custom y-limits
# plot_bands(band_data, proj=projection_data, ylims=(-3.0, 2.0))

# Plot spin-polarized bands in two separate panels
# plot_bands(band_data_spin, double_plot=true)
```
"""
function plot_bands(bands; proj=nothing, wann=nothing, ylims=(-5.0,5.0), path="", double_plot=false, k_labels::Vector{String}=[" "], spinor_component::Int64=4)
    klabels = bands["klabels"]
    k_ind = bands["k_ind"]
    if k_labels != [" "]
        if length(k_labels) != length(k_ind)
            error("$(length(k_labels)) labels provided, but $(length(k_ind)) symmetry points expected.")
        end
        klabels = k_labels
    end

    E = bands["E"]
    system = bands["system"]

    bar_limts = (0,1)
    if !isnothing(proj)
        if size(proj, 3) == 4
            proj = copy(proj[:, :, spinor_component])
            bar_limts = (-1, 1)
        end
    end

    inch = 96
    pt = 4/3
    fig_size = (3.40inch, 2.55inch)
    double_plot = double_plot*(size(E, 3) == 2)
    if double_plot
        fig_size = (2*3.40inch, 2.55inch)
    end

    fig = Figure(size = fig_size, fonts = (; regular = "Dejavu", weird = "Blackchancery"), figure_padding = (4,4,4,8))

    ax = Axis(fig[1, 1], 
          ylabel = L"E - E_F \;\; \textrm{[eV]}",
          limits = (1, size(E, 1), ylims[1], ylims[2]),
          xlabelsize = 11pt,
          ylabelsize = 11pt, 
          xticklabelsize = 8pt,
          yticklabelsize = 8pt,
          xticks = (k_ind, klabels),
          xticksize = 5pt,
          yticksize = 5pt,
          xtickalign=1,
          ytickalign=1,
          xminortickalign=1,
          yminortickalign=1,
          xminorticksvisible = false,
          xgridvisible = true,
          yminorticksvisible = true,
          ygridvisible = false,
          xminorticks = IntervalsBetween(5),
          yminorticks = IntervalsBetween(5),
          xminorticksize = 2.5pt,
          yminorticksize = 2.5pt)
    hlines!(ax, 0.0, color = :grey41, linestyle = :dash, linewidth = 1.5)

    if !isnothing(proj)
        Colorbar(fig[1,2], limits = bar_limts, colormap = :Blues_8, flipaxis = true, ticklabelsize=8pt)
        if !double_plot && size(E, 3) == 2
            Colorbar(fig[1,3], limits = bar_limts, colormap = :Reds_8, flipaxis = true, ticklabelsize=8pt)
        end
    end

    if double_plot
        panel_number = 2
        if !isnothing(proj)
            panel_number = 3
            Colorbar(fig[1,4], limits = bar_limts, colormap = :Reds_8, flipaxis = true, ticklabelsize=8pt)
        end
        ax2 = Axis(fig[1, panel_number], 
            limits = (1, size(E, 1), ylims[1], ylims[2]),
            xlabelsize = 11pt,
            ylabelsize = 11pt, 
            xticklabelsize = 8pt,
            yticklabelsize = 8pt,
            xticks = (k_ind, klabels),
            xticksize = 5pt,
            yticksize = 5pt,
            xtickalign=1,
            ytickalign=1,
            xminortickalign=1,
            yminortickalign=1,
            xminorticksvisible = false,
            xgridvisible = true,
            yminorticksvisible = true,
            ygridvisible = false,
            xminorticks = IntervalsBetween(5),
            yminorticks = IntervalsBetween(5),
            xminorticksize = 2.5pt,
            yminorticksize = 2.5pt)
        hlines!(ax2, 0.0, color = :grey41, linestyle = :dash, linewidth = 1.5)
    end

    for b in axes(E, 2)
        for s in axes(E, 3)
            panel = ax
            Ec = copy(E[:, b, s])
            x = range(1,length(Ec))
            ls = :solid
            if s==1
                color=RGBf(0.251, 0.388, 0.847)
                colormap=:Blues_8
            else
                color=RGBf(0.796, 0.235, 0.2)
                colormap=:Reds_8
                if double_plot
                    panel = ax2
                end
            end
            if !isnothing(proj)
                color = proj[:, b, s]
                append!(color, 0, 1)
                append!(Ec, 0, 0)
                x = range(1,length(Ec))
            end
            if !double_plot && s==2
                linesegments!(panel, x, Ec, color=color, linestyle=ls, linewidth=2, colormap=colormap)
            else
                lines!(panel, x, Ec, color=color, linestyle=ls, linewidth=2, colormap=colormap)
            end
        end
    end

    if !isnothing(wann)
        for w in axes(wann, 1)
            scatter!(ax, range(1,size(E,1), length(wann[w,:])), wann[w,:], color=RGBf(0.22, 0.596, 0.149), markersize=3)
        end
        if double_plot
            for w in axes(wann, 1)
                scatter!(ax2, range(1,size(E,1), length(wann[w,:])), wann[w,:], color=RGBf(0.22, 0.596, 0.149), markersize=3)
            end
        end
    end

    app = "_bands"
    if !isnothing(proj)
        app = "_proj"
    end
    if !isnothing(wann)
        app = "_wann"
    end

    save(joinpath(path,system*app*".png"), fig, px_per_unit = 300/inch)
end

function extract_coefficents(wanproj::String)
    Spin, Nk, Nb, Nw = 0, 0, 0, 0
    Nb_max = 0
    Nb_min = Inf

    # Extract header information
    for line in eachline(wanproj)
        # Find the header line with Spin, Nk, Nb, Nw
        match_header = match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        # Find each coefficient line with Nb, Nw, and coefficients
        match_coeff = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$", line)
        if match_header !== nothing
            Spin, Nk, Nb, Nw = parse.(Int, match_header.captures)
        elseif match_coeff !== nothing
            Nb_max = max(Nb_max, parse(Int, match_coeff.captures[1]))
            Nb_min = Int(min(Nb_min, parse(Int, match_coeff.captures[1])))
        end
    end
    if Spin*Nk*Nb*Nw*Nb_max/Nb_min == 0
        error("One or more variables not found in $wanproj. Nk, Nb, Nw, Spin, Nb_max, Nb_min: $Nk, $Nb, $Nw, $Spin, $Nb_max, $Nb_min")
    end

    # Initialize arrays for coefficients and k-points
    coefficients = zeros(ComplexF64, Nk, Nb_max-Nb_min+1, Nw, Spin)
    kpoints = zeros(Float64, Nk, 3)
    k = 0
    s = 0

    for line in eachline(wanproj)
        # Find each block line with spin, local Nb, and k-point
        match_block = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$", line)
        # Find each coefficient line with band number, wannier function, and coefficients
        match_coeff = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$", line)

        if match_block !== nothing
            k += 1
            kpoints[k,:] = parse.(Float64, match_block.captures[3:5])
            s = parse(Int, match_block.captures[1])
        end
        if match_coeff !== nothing
            b, w = parse.(Int, match_coeff.captures[1:2])
            b = b - Nb_min + 1  # Adjust band index to be 1-based and within range
            c_real, c_im = parse.(Float64, match_coeff.captures[3:4])
            coefficients[k, b, w, s] = ComplexF64(c_real, c_im)
        end
    end

    return coefficients, kpoints
end

function calc_hopping(wanproj::String, outcar::String; neighbours::Vector{Vector{Int64}}=[[1,0,0],[0,1,0],[0,0,1]])
    coefficients, kpoints = extract_coefficents(wanproj)
    bands = extract_energies(outcar)
    E = bands["E"]

    # compute t_ij = - <w_i|H|w_j> = -∑_α,β,k <c_αi^* exp(ik.R_i)ψ_α^*|H|c_βj exp(-ik.R_j)ψ_β> / Nk^2 = -∑_α,k exp(ik.(R_i-R_j)) E_α / Nk^2
end

end
