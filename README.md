# VaspPlot
This repository contains Julia code for plotting VASP output. To use the functionalities, first add the package to your environment:
```
julia> using Pkg
julia> Pkg.add(url="https://github.com/DaanVrancken/VaspPlot")
```
Then, simply write 
```
using VaspPlot
```
at the top of your script.

A simple example would look like this
```
using VaspPlot

# Extract energy bands
bands = extract_energies("OUTCAR")

# Extract projections for specific ions and orbitals
# In this case, the pz orbitals of the first four ions
ions = collect(1:4)
orbs = ["pz"]
proj = extract_projections("PROCAR", ions, orbs)

# Extract Wannier interpolated bands
Ef_LWL = -3.2576978695
wann = extract_MLWFs("wannier90_band.dat") .- Ef_LWL

# Plotting
plot_bands(bands; proj=proj, wann=wann, ylims=(-2.0, 2.0))
```
