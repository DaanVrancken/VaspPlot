module VaspPlot

export extract_energies, extract_projections, extract_MLWFs, plot_bands

using CairoMakie

# Define functions
function remove_slices(arr::AbstractArray, dim::Int, indices_to_remove)
    all_indices = 1:size(arr, dim)
    indices_to_keep = filter(i -> !(i in indices_to_remove), all_indices)
    slices = ntuple(d -> d == dim ? indices_to_keep : Colon(), ndims(arr))
    return arr[slices...]
end

function find_klabels(kpoints, path_length)
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

function extract_energies(OUTCAR)
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
            Ef[1] = parse.(Float64, split(lines[i])[3])
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

function extract_projections(filename::String, ions::Vector{Int64}, orbs::Union{Vector{String}, Vector{Int64}})
    if !isfile(filename)
        error("File not found: $filename")
    end

    num_k_points = 0
    num_bands = 0
    num_ions = 0
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
                num_ions = parse(Int, match_params.captures[3])
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
            # Go to next spin component
            elseif startswith(stripped_line, "tot")
                current_spin_idx +=1
                current_spin_idx = mod1(current_spin_idx, num_spin_components)
            end
        end
    end

    return projections
end

function extract_MLWFs(wannier_file)
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

    save(joinpath(path,"bands_"*system*".png"), fig, px_per_unit = 300/inch)
end

end # module VaspPlot
