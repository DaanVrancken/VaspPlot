module VaspPlot

export extract_energies, extract_projections, extract_MLWFs, plot_bands, calc_hopping

using LinearAlgebra, CairoMakie

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
    find_klabels(kpoints::Matrix{Float64}, path_length::Int64, doubles::Vector{Int64})

Identifies and labels high-symmetry k-points within a k-point path, crucial for band structure analysis.

Matches k-points in `kpoints` against standard high-symmetry labels (e.g., Γ, X, R) based on `path_length`. It intelligently handles duplicate k-points specified in `doubles` to ensure correct indexing.

# Arguments
- `kpoints::Matrix{Float64}`: Matrix of 3D k-points `[kx, ky, kz]` per row.
- `path_length::Int64`: Number of k-points per segment between expected high-symmetry points.
- `doubles::Vector{Int64}`: 1-based indices in `kpoints` that are duplicate points (e.g., segment ends).

# Returns
- `klabels::Vector{Char}`: Vector of assigned high-symmetry labels (e.g., 'Γ', 'X') or ' ' if no match.
- `k_ind::Vector{Int64}`: Corrected 1-based indices in `kpoints` corresponding to the identified symmetry points, adjusted for `doubles`.
"""
function find_klabels(kpoints::Matrix{Float64}, path_length::Int64, doubles::Vector{Int64})
    N_labels = (size(kpoints,1)) /path_length
    k_ind = vcat([1], [[i*path_length, i*path_length+1] for i in 1:N_labels]...)
    pop!(k_ind)
    k_ind = map(Int, setdiff(k_ind, doubles))

    # Find corresponding symmetry points
    klabels = collect(" " for _ in eachindex(k_ind))

    standard_labels = Dict(
        "Γ" => [0.0, 0.0, 0.0],
        "X" => [0.5, 0.0, 0.0],
        "Y" => [0.0, 0.5, 0.0],
        "Z" => [0.0, 0.0, 0.5],
        "S" => [0.5, 0.5, 0.0],
        "U" => [0.5, 0.0, 0.5],
        "T" => [0.0, 0.5, 0.5],
        "R" => [0.5, 0.5, 0.5]
    )

    for i in eachindex(k_ind)
        kpoint = kpoints[k_ind[i], :]
        for (label, coords) in standard_labels
            if isapprox(kpoint, coords, atol=1e-10)
                klabels[i] = label
                break
            end
        end
    end

    # Correct indices for double points in kpoints
    d = 0
    k_ind_original = copy(k_ind)
    for i in eachindex(k_ind_original)
        k_ind[i] -= d
        if k_ind_original[i]+1 in doubles
            d += 1
        end
    end

    return klabels, k_ind
end

"""
    merge_indices_and_labels(ind::Vector{Int}, lab::Vector{String}) -> (Vector{Float64}, Vector{String})

Merges adjacent elements in `ind` and `lab` arrays based on a specific index condition.

If two subsequent indices `ind[i]` and `ind[i+1]` differ by exactly one (i.e., `ind[i] == ind[i+1] - 1`),
these two elements are replaced by a single new element:
- The new index will be `ind[i] + 0.5`.
- The new label will be a concatenation of the original labels `lab[i] * "|" * lab[i+1]`.

Elements that do not meet this merging condition are kept as they are.

# Arguments
- `ind::Vector{Int}`: An array of integer indices.
- `lab::Vector{String}`: An array of corresponding string labels, of the same length as `ind`.

# Returns
A tuple containing two new arrays:
- `new_ind::Vector{Float64}`: The array of modified indices, which may now contain `Float64` values due to merging.
- `new_lab::Vector{String}`: The array of modified labels.

# Examples
```julia
ind = [1, 2, 4, 5, 7]
lab = ["A", "B", "C", "D", "E"]
new_ind, new_lab = merge_indices_and_labels(ind, lab)
# new_ind will be [1.5, 4.5, 7]
# new_lab will be ["A|B", "C|D", "E"]

ind2 = [10, 11, 13, 14, 15]
lab2 = ["X", "Y", "Z", "W", "V"]
new_ind2, new_lab2 = merge_indices_and_labels(ind2, lab2)
# new_ind2 will be [10.5, 13.5]
# new_lab2 will be ["X|Y", "Z|W|V"] # Note: The example in the original code had an error here.
                                   # This docstring reflects the correct output for the given logic.
"""
function merge_indices_and_labels(ind::Vector{Int}, lab::Vector{String})
    new_ind = Float64[] # Initialize an empty array for new indices
    new_lab = eltype(lab)[] # Initialize an empty array for new labels

    i = 1
    while i <= length(ind)
        if i < length(ind) && ind[i] == ind[i+1] - 1
            # Merge condition met
            push!(new_ind, ind[i] + 0.5)
            push!(new_lab, lab[i] * "|" * lab[i+1])
            i += 2 # Skip the next element as it's been merged
        else
            # No merge, just add the current elements
            push!(new_ind, ind[i])
            push!(new_lab, lab[i])
            i += 1
        end
    end
    return new_ind, new_lab
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
    reciprocal_vectors = zeros(3,3)

    for i in eachindex(lines)
        if occursin("reciprocal lattice vectors", lines[i])
            for j in 1:3
                reciprocal_vectors[j, :] = parse.(Float64, split(lines[i+j])[4:6])
            end
            i0 = i+4
            break
        end
    end

    for i in i0:length(lines)
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

    doubles::Vector{Int64} = []
    for k in range(2, Nk)
        if isapprox(kpoints[k,:], kpoints[k-1,:], atol=1e-10)
            push!(doubles, k)
        end
    end

    klabels, k_ind = find_klabels(kpoints, path_length, doubles)

    kpoints = remove_slices(kpoints, 1, doubles)
    E = remove_slices(E, 1, doubles)

    edges = kpoints[k_ind,:]
    path_edges = edges * reciprocal_vectors

    return Dict("klabels" => klabels, "k_ind" => k_ind, "E" => E, "system" => system, "path_edges" => path_edges)
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
                                    For non-collinear, it's 4.

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

function extract_MLWFs(wannier_1::String, wannier_2::String)
    MLWFs_up = extract_MLWFs(wannier_1)
    MLWFs_down = extract_MLWFs(wannier_2)

    return Dict("up" => MLWFs_up, "down" => MLWFs_down)
end

"""
    find_path_lengths(path_edges::LinearAlgebra.Adjoint{Float64, Matrix{Float64}}, 
                      klabels::Vector{String})

Compute the Euclidean distances between consecutive points along a path.

# Arguments
- `path_edges::LinearAlgebra.Adjoint{Float64, Matrix{Float64}}`: 
    A 2D array (as an adjoint of a matrix) containing the coordinates of the 
    path points. Each row corresponds to a point in space.
- `klabels::Vector{String}`: 
    Labels for the path points. If a label has length > 1, it indicates a 
    duplicated or merged point, and the indexing is adjusted accordingly.

# Returns
- `Vector{Float64}`: Distances between consecutive valid path points. 
  The result has length `length(klabels) - 1`.

# Notes
- The function maintains a counter to skip over duplicated labels.
- Distances are computed using the Euclidean norm (`LinearAlgebra.norm`).
"""
function find_path_lengths(path_edges::Matrix{Float64}, klabels::Vector{String})
    path_lengths = zeros(length(klabels)-1)
    double_counter = 0
    for i in eachindex(path_lengths)
        if length(klabels[i]) > 1
            double_counter += 1
        end
        path_lengths[i] = norm(path_edges[i+1+double_counter,:] - path_edges[i+double_counter,:])
    end

    return path_lengths
end

"""
    plot_bands(bands; proj=nothing, wann=nothing, ylims=(-5.0,5.0), path="", double_plot=false, k_labels::Vector{String}=[" "], spinor_component::Int64=4)

Plots electronic band structures, optionally with projected contributions or Wannier functions.

# Arguments
- `bands::Dict`: A dictionary containing band structure data, typically from `extract_energies`.
                 Expected keys: `"klabels"`, `"k_ind"`, `"E"`, `"system"`, `"path_edges"`.
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
                              Must match the number of symmetry points (`length(bands["k_ind"])`). For discontinuous paths, use e.g. `"G|X"`.
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

    k_ind, klabels = merge_indices_and_labels(k_ind, klabels)

    # Construct data x-axis
    path_edges = bands["path_edges"]
    path_lengths = find_path_lengths(path_edges, klabels)
    X = zeros(size(E,1))
    for i in eachindex(path_lengths)
        X[ceil(Int, k_ind[i]):floor(Int, k_ind[i+1])] .+= range(0, path_lengths[i], length=floor(Int,k_ind[i+1])-ceil(Int,k_ind[i])+1)
        if i != length(path_lengths)
            X[floor(Int, k_ind[i+1])+1:end] .+= path_lengths[i]
        end
    end
    x_ticks = vcat(0, cumsum(path_lengths))

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
          limits = (first(X), last(X), ylims[1], ylims[2]),
          xlabelsize = 11pt,
          ylabelsize = 11pt, 
          xticklabelsize = 8pt,
          yticklabelsize = 8pt,
          xticks = (x_ticks, klabels),
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
            limits = (first(X), last(X), ylims[1], ylims[2]),
            xlabelsize = 11pt,
            ylabelsize = 11pt, 
            xticklabelsize = 8pt,
            yticklabelsize = 8pt,
            xticks = (x_ticks, klabels),
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
            x_values = copy(X)
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
                push!(x_values, maximum(X)+0.1, maximum(X)+0.2)
            end
            if !double_plot && s==2
                linesegments!(panel, x_values, Ec, color=color, linestyle=ls, linewidth=2, colormap=colormap)
            else
                lines!(panel, x_values, Ec, color=color, linestyle=ls, linewidth=2, colormap=colormap)
            end
        end
    end

    if !isnothing(wann)
        if typeof(wann) == Dict{String, Matrix{Float64}}
            wann1 = wann["up"]
            wann2 = wann["down"]
        else
            wann1 = copy(wann)
            wann2 = copy(wann)
        end
        for w in axes(wann1, 1)
            scatter!(ax, range(0,maximum(X),length(wann1[w,:])), wann1[w,:], color=RGBf(0.22, 0.596, 0.149), markersize=3)
        end
        if double_plot
            for w in axes(wann2, 1)
                scatter!(ax2, range(0,maximum(X),length(wann2[w,:])), wann2[w,:], color=RGBf(0.22, 0.596, 0.149), markersize=3)
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
    Nb_max_up = 0
    Nb_min_up = Inf
    Nb_max_down = 0
    Nb_min_down = Inf

    # Extract header information
    for line in eachline(wanproj)
        # Find the header line with Spin, Nk, Nb, Nw
        match_header = match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        # Find each coefficient line with Nb, Nw, and coefficients
        match_coeff = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$", line)
        match_block_header = match(r"^\s*(-?\d+)\s+(-?\d+)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s+(-?\d+\.\d+(?:[eE][-+]?\d+)?)\s*$", line)
        if match_header !== nothing
            Spin, Nk, Nb, Nw = parse.(Int, match_header.captures)
        elseif match_block_header !== nothing
            s =  parse(Int, match_block_header.captures[1])
        elseif match_coeff !== nothing
            if s==1
                Nb_max_up = max(Nb_max_up, parse(Int, match_coeff.captures[1]))
                Nb_min_up = Int(min(Nb_min_up, parse(Int, match_coeff.captures[1])))
            elseif s==2
                Nb_max_down = max(Nb_max_down, parse(Int, match_coeff.captures[1]))
                Nb_min_down = Int(min(Nb_min_down, parse(Int, match_coeff.captures[1])))
            else
                error("Unexpected spin value $s found in $wanproj")
            end
        end
    end
    if Spin*Nk*Nb*Nw*Nb_max_up/Nb_min_up == 0
        error("One or more variables not found in $wanproj. Nk, Nb, Nw, Spin, Nb_max, Nb_min: $Nk, $Nb, $Nw, $Spin, $Nb_max_up, $Nb_min_up")
    end
    
    Nb_max = max(Nb_max_up, Nb_max_down)
    Nb_min = min(Nb_min_up, Nb_min_down)

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
            k = mod1(k+1, Nk)
            s = parse(Int, match_block.captures[1])
            if s == 1
                kpoints[k,:] = parse.(Float64, match_block.captures[3:5])
            end
        end
        if match_coeff !== nothing
            b, w = parse.(Int, match_coeff.captures[1:2])
            b = b - Nb_min + 1  # Adjust band index to be 1-based and within range
            c_real, c_im = parse.(Float64, match_coeff.captures[3:4])
            coefficients[k, b, w, s] = ComplexF64(c_real, c_im)
        end
    end

    return coefficients, kpoints, Nb_min_up, Nb_max_up, Nb_min_down, Nb_max_down
end

function extract_excluded_bands(win::String)
       # Read all lines from the file
    lines = readlines(win)
    
    # Find the line containing "exclude_bands"
    line = nothing
    for l in lines
        if occursin(r"^\s*exclude_bands", l)
            line = l
            break
        end
    end
    line === nothing && return Int[]  # no excluded bands found
    
    # Remove "exclude_bands", separators (:, =, or whitespace), and extra spaces
    clean = replace(line, r"^\s*exclude_bands\s*[:=]?\s*" => "")
    
    # Split by commas
    parts = split(clean, ",")
    
    excluded = Int[]
    for part in parts
        part = strip(part)
        if isempty(part)
            continue
        elseif occursin("-", part)  # handle ranges like 6-8
            a, b = split(part, "-")
            append!(excluded, parse(Int, a):parse(Int, b))
        else
            push!(excluded, parse(Int, part))
        end
    end
    
    return sort(unique(excluded))
end

function map_to_original_bands(bands_kept::Vector{Int}, excluded::Vector{Int})
    max_band = maximum(vcat(bands_kept, excluded))
    all_bands = collect(1:max_band)
    
    non_excluded = setdiff(all_bands, excluded)
    
    return non_excluded[bands_kept]
end

function extract_energies_eig(wannier_eig::String, win::String, Nb_min::Int64, Nb_max::Int64)
    Nk = 0
    Nb_found = 0

    # First pass: find dimensions
    for line in eachline(wannier_eig)
        fields = split(strip(line))
        b = parse(Int, fields[1])
        k = parse(Int, fields[2])
        Nb_found = max(Nb_found, b)
        Nk = max(Nk, k)
    end

    excluded = extract_excluded_bands(win)
    bands = map_to_original_bands(collect(1:Nb_found), excluded)

    if maximum(bands) < Nb_max || minimum(bands) > Nb_min
        error("Bands in $wannier_eig do not match the specified range Nb_min=$Nb_min, Nb_max=$Nb_max after excluding bands: $excluded")
    end

    E = zeros(Nk, Nb_max-Nb_min+1)

    for line in eachline(wannier_eig)
        fields = split(strip(line))
        b = parse(Int, fields[1])
        b_index = bands[b] - Nb_min + 1
        k = parse(Int, fields[2])
        E[k, b_index] = parse(Float64, fields[3])
    end

    return E
end

function calc_hopping(neighbours::Vector{Vector{Int64}}=[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]; file_directory=".")
    win = joinpath(file_directory, "wannier90.win")
    wanproj = joinpath(file_directory, "WANPROJ")
    wannier_eig = joinpath(file_directory, "wannier90.eig")
    win1 = joinpath(file_directory, "wannier90.1.win")
    win2 = joinpath(file_directory, "wannier90.2.win")
    wannier_eig1 = joinpath(file_directory, "wannier90.1.eig")
    wannier_eig2 = joinpath(file_directory, "wannier90.2.eig")

    if !isfile(wanproj)
        error("WANPROJ file not found in $file_directory")
    end
    if !isfile(win) && (!isfile(win1) || !isfile(win2))
        error("No wannier90.win file found in $file_directory")
    end
    if !isfile(wannier_eig) && (!isfile(wannier_eig1) || !isfile(wannier_eig2))
        error("No wannier90.eig file found in $file_directory")
    end

    coefficients, kpoints, Nb_min_up, Nb_max_up, Nb_min_down, Nb_max_down = extract_coefficents(wanproj)

    Nk = size(coefficients, 1)
    Nw = size(coefficients, 3)
    Spin = size(coefficients, 4)
    Nb = maximum([Nb_max_up-Nb_min_up+1, Nb_max_down-Nb_min_down+1])

    if isfile(wannier_eig)
        E = extract_energies_eig(wannier_eig, win, Nb_min_up, Nb_max_up)
        if Spin == 2
            error("Spin-polarized coefficients found in WANPROJ, but only one wannier90.eig file present.")
        end
    else
        E_up = extract_energies_eig(wannier_eig1, win1, Nb_min_up, Nb_max_up)
        E_down = extract_energies_eig(wannier_eig2, win2, Nb_min_down, Nb_max_down)
    end

    if Spin == 1
        t = zeros(ComplexF64, Nw, Nw, length(neighbours))
    else
        t = zeros(ComplexF64, Nw, Nw, length(neighbours), Spin)
    end
    
    for R in eachindex(neighbours)
        for i in range(1,Nw)
            for j in range(1,Nw)
                for k in range(1,Nk)
                    for b in range(1,Nb)
                        if Spin == 1
                            t[i,j,R] -= coefficients[k,b,i,1]' * coefficients[k,b,j,1] * E[k,b] * exp(2im * π * dot(kpoints[k,:], neighbours[R]))
                        else
                            if b <= Nb_max_up-Nb_min_up+1
                                t[i,j,R,1] -= coefficients[k,b,i,1]' * coefficients[k,b,j,1] * E_up[k,b] * exp(2im * π * dot(kpoints[k,:], neighbours[R]))
                            end
                            if b <= Nb_max_down-Nb_min_down+1
                                t[i,j,R,2] -= coefficients[k,b,i,2]' * coefficients[k,b,j,2] * E_down[k,b] * exp(2im * π * dot(kpoints[k,:], neighbours[R]))
                            end
                        end
                    end
                end
            end
        end
    end

    return t./Nk
end


end
