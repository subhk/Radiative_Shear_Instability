using LazyGrids
using BlockArrays
using Printf
using Interpolations
using SparseArrays
using SparseMatrixDicts
using SpecialFunctions
using FillArrays
using Parameters
using Test
using BenchmarkTools
using BasicInterpolators: BicubicInterpolator

using Serialization
using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod
using Dierckx
using LinearAlgebra
using JacobiDavidson
using JLD
using SparseMatricesCSR

using CairoMakie
using LaTeXStrings
CairoMakie.activate!()
using DelimitedFiles
using ColorSchemes
using MAT
using IterativeSolvers
using DelimitedFiles

# include("dmsuite.jl")
# include("transforms.jl")
# include("utils.jl")

## interpolating the eigenfunction
function Interp2D_eigenFun(yn, zn, An, y0, z0)
    itp = BicubicInterpolator(yn, zn, transpose(An))
    A₀ = zeros(Float64, length(y0), length(z0))
    A₀ = [itp(yᵢ, zᵢ) for yᵢ ∈ y0, zᵢ ∈ z0]
    return A₀
end

function Interp2D(rn, zn, An, grid)
    itp = BicubicInterpolator(rn, zn, An)
    #itp = LinearInterpolation((rn, zn), An)
    A₀ = zeros(Float64, length(grid.r), length(grid.z))
    A₀ = [itp(rᵢ, zᵢ) for rᵢ in grid.r, zᵢ in grid.z]
    A₀ = transpose(A₀)
    A₀ = A₀[:]
    return A₀
end

function ddz(z)
    del = z[2] - z[1]
    N = length(z)
    d = zeros(N, N)
    for n in 2:N-1
        d[n, n-1] = -1.0
        d[n, n+1] = 1.0
    end
    d[1, 1]   = -3.0; d[1, 2] = 4.0
    d[1, 3]   = -1.0; d[N, N] = 3.0
    d[N, N-1] = -4.0; d[N, N-2] = 1.0
    d = d / (2 * del)
    return d
end


function ddz2(z)
    del = z[2] - z[1]
    N = length(z)

    d = zeros(N, N)
    for n in 2:N-1
        d[n, n-1] = 1.0; d[n, n] = -2.0
        d[n, n+1] = 1.0
    end
    d[1, 1] = 2.0;   d[1, 2] = -5.0; d[1, 3] = 4.0
    d[1, 4] = -1.0;  d[N, N] = 2.0;  d[N, N-1] = -5.0
    d[N, N-2] = 4.0; d[N, N-3] = -1.0
    d = d / del^2
    return d
end

function read_eigenvec(which_file, no_eqs, Nr, Nz)
    @assert which_file > 0
    eigenvec = readdlm("real_eigenvec_" * string(which_file) * ".dat", '\t', Float64, '\n')
    (row, col) = size(eigenvec)
    @printf "row: %i  col: %i \n" row col

    N::Int = row/no_eqs
    println(N)
    @assert N == Nr*Nz
    
    uᵣ = eigenvec[   1:1N, 1]; 
    uₜ = eigenvec[1N+1:2N, 1]; 
    w  = eigenvec[2N+1:3N, 1]; 
    p  = eigenvec[3N+1:4N, 1]; 
    b  = eigenvec[4N+1:5N, 1]; 

    # file = matopen("eddy_structure_nd.mat");
    # rn   = transpose( read(file, "r" ) )[:,1];
    # zn   = transpose( read(file, "z" ) )[:,1];
    # Un   = transpose( read(file, "U" ) );
    # Bn   = transpose( read(file, "Bz") );
    # close(file)


    # ------------- setup discrete diff matrices  -------------------
    # chebyshev in y-direction
    # r, Dʳ  = chebdif(Nr, 1)
    # r, Dʳʳ = chebdif(Nr, 2)
    # # Transform [0, Rₘₐₓ]
    # r, Dʳ, Dʳʳ = chebder_transform(r, Dʳ, Dʳʳ, zerotoL_transform, 3.0)
    # # chebyshev in z-direction
    # z, Dᶻ  = chebdif(Nz, 1)
    # z, Dᶻᶻ = chebdif(Nz, 2)
    # # Transform the domain and derivative operators from [-1, 1] → [0, H]
    # z, Dᶻ, Dᶻᶻ = chebder_transform(z, Dᶻ, Dᶻᶻ, zerotoL_transform, 1.0)

    r  = collect(range(0.0,  stop=12.0, length=Nr))
    z  = collect(range(-3.0, stop=3.0,  length=Nz))
    # Dʳ = ddz(r); Dʳʳ = ddz2(r);
    # Dᶻ = ddz(z); Dᶻᶻ = ddz2(z);

    uᵣ = reshape( uᵣ, (length(z), length(r)) )
    uₜ = reshape( uₜ, (length(z), length(r)) )
    w  = reshape( w,  (length(z), length(r)) )
    b  = reshape( b,  (length(z), length(r)) )

    r_interp = collect(LinRange(minimum(r), maximum(r), 5000))
    z_interp = collect(LinRange(minimum(z), maximum(z), 500) )

    # interpolate U and B on (r_interp, z_interp)
    # U₀ = Interp2D_eigenFun(rn, zn, Un, r_interp, z_interp)
    # B₀ = Interp2D_eigenFun(rn, zn, Bn, r_interp, z_interp)

    fig = Figure(fontsize=30, resolution = (1800, 580)) #, font="Times")

    ax1 = Axis(fig[1, 1], xlabel=L"$r/L$", xlabelsize=30, ylabel=L"$z/H$", ylabelsize=30)

    interp_  = Interp2D_eigenFun(r, z, uᵣ, r_interp, z_interp)
    max_val = maximum(abs.(interp_))
    levels = range(-0.7max_val, 0.7max_val, length=16)
    co = contourf!(r_interp, z_interp, interp_, colormap=cgrad(:RdBu, rev=false),
        levels=levels, extendlow = :auto, extendhigh = :auto )

    # levels = range(minimum(U₀), maximum(U₀), length=8)
    # contour!(r_interp, z_interp, U₀, levels=levels, linestyle=:dash, color=:black, linewidth=2) 
                
    tightlimits!(ax1)
    cbar = Colorbar(fig[1, 2], co)
    xlims!(0., maximum(r))
    ylims!(minimum(z), maximum(z))

    ax2 = Axis(fig[1, 3], xlabel=L"$r/L$", xlabelsize=30, ylabel=L"$z/H$", ylabelsize=30)

    interp_ = Interp2D_eigenFun(r, z, uₜ, r_interp, z_interp)
    max_val = maximum(abs.(interp_))
    levels = range(-0.7max_val, 0.7max_val, length=16)
    co = contourf!(r_interp, z_interp, interp_, colormap=cgrad(:RdBu, rev=false),
        levels=levels, extendlow = :auto, extendhigh = :auto )

    # levels = range(minimum(B₀), maximum(B₀), length=8)
    # contour!(r_interp, z_interp, B₀, levels=levels, linestyle=:dash, color=:black, linewidth=2) 

    # levels = range(minimum(U₀), maximum(U₀), length=8)
    # contour!(r_interp, z_interp, U₀, levels=levels, linestyle=:dash, color=:black, linewidth=2) 
        
    tightlimits!(ax2)
    cbar = Colorbar(fig[1, 4], co)
    xlims!(0., maximum(r))
    ylims!(minimum(z), maximum(z))

    ax1.title = L"Re$(\hat{u}_r)$"
    ax2.title = L"Re$(\hat{u}_\theta)$"

    fig
    #save("AAI2d_" * string(which_file) * ".png", fig, px_per_unit=4)
end

# read_eigenvec(which_file, no_eqs, Nr, Nz)
read_eigenvec(2, 5, 180, 90)