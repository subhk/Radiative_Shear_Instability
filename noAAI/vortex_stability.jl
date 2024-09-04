using LazyGrids
using BlockArrays
using Printf
using Statistics
using SparseArrays
using SparseMatrixDicts
using SpecialFunctions
using FillArrays
using Parameters
using Test
using BenchmarkTools
using Trapz
using BasicInterpolators: BicubicInterpolator
#using Suppressor: @suppress_err

using Serialization
using Dierckx
using LinearAlgebra
using JLD2
#using SparseMatricesCSR
using MatrixMarket: mmwrite
using SparseMatricesCOO

using CairoMakie
using LaTeXStrings
CairoMakie.activate!()
using DelimitedFiles
using ColorSchemes
using MAT
using IterativeSolvers
using DSP

using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod

#using Dierckx: Spline2D, evaluate

ENV["PYTHON"] = "/home/subhajitkar/anaconda_nw/bin/python"
using PyCall
@pyimport scipy.interpolate as si

# include("../feast.jl")
# using ..feastLinear

# include("../FEASTSolver/src/FEASTSolver.jl")
# using Main.FEASTSolver

include("dmsuite.jl")
include("transforms.jl")
include("utils.jl")
include("shift_invert.jl")

# function Interp2D(rn, zn, An, grid)
#     itp = BicubicInterpolator(rn, zn, An)
#     A₀ = [itp(rᵢ, zᵢ) for rᵢ in grid.r, zᵢ in grid.z]
#     return A₀
# end

function low_PassFilter_y(cutoff_freq, fs, data)
    responsetype    = Lowpass(cutoff_freq; fs=fs)
    designmethod    = Butterworth(6)
    setfilter       = digitalfilter(responsetype, designmethod);
    lowpass  = similar(data)
    for jt in 1:size(data,2)
        for kt in 1:size(data,3)
            lowpass[:,jt,kt] = filtfilt(setfilter, data[:,jt,kt])
        end
    end
    return lowpass
end

function low_PassFilter_x(cutoff_freq, fs, data)
    responsetype    = Lowpass(cutoff_freq; fs=fs)
    designmethod    = Butterworth(6)
    setfilter       = digitalfilter(responsetype, designmethod);
    lowpass  = similar(data)
    for jt in 1:size(data,1)
        for kt in 1:size(data,3)
            lowpass[jt,:,kt] = filtfilt(setfilter, data[jt,:,kt])
        end
    end
    return lowpass
end

function Interp2D(rn, zn, An, grid)
    spl = Spline2D(rn, zn, An; s=0.0)
    A₀ = [spl(rᵢ, zᵢ) for rᵢ in grid.r, zᵢ in grid.z]
    return A₀
end

#### plotting the eigenfunction
function Interp2D_eigenFun(yn, zn, An, y0, z0)
    itp = BicubicInterpolator(yn, zn, transpose(An))
    A₀ = zeros(Float64, length(y0), length(z0))
    A₀ = [itp(yᵢ, zᵢ) for yᵢ ∈ y0, zᵢ ∈ z0]
    return A₀
end

# function Interp2D(rn, zn, An, grid)
#     spl   = si.RectBivariateSpline(rn, zn, An)
#     value = [spl(rᵢ, zᵢ) for rᵢ in grid.r, zᵢ in grid.z]
#     A₀  = zeros(Float64, length(grid.r), length(grid.z))
#     A₀ .= value
#     return A₀ #value
# end

# function Interp2D_eigenFun(rn, zn, An, r0, z0)
#     spl = Spline2D(rn, zn, An)
#     A₀ = zeros(Float64, length(r0), length(z0))
#     A₀ = [spl(rᵢ, zᵢ) for rᵢ ∈ r0, zᵢ ∈ z0]
#     return A₀
# end

function twoDContour(r, z, Χ, N, it)
    which::Int = 1
    uᵣ = cat( real(Χ[   1:1N, which]), imag(Χ[   1:1N, which]), dims=2 ); 
    uₜ  = cat( real(Χ[1N+1:2N, which]), imag(Χ[1N+1:2N, which]), dims=2 ); 
    w  = cat( real(Χ[2N+1:3N, which]), imag(Χ[2N+1:3N, which]), dims=2 ); 
    p  = cat( real(Χ[3N+1:4N, which]), imag(Χ[3N+1:4N, which]), dims=2 ); 
    b  = cat( real(Χ[4N+1:5N, which]), imag(Χ[4N+1:5N, which]), dims=2 ); 

    uᵣ = reshape( uᵣ[:,1], (length(z), length(r)) )
    uₜ  = reshape( uₜ[:,1], (length(z), length(r)) )
    w  = reshape( w[:,1],  (length(z), length(r)) )
    b  = reshape( b[:,1],  (length(z), length(r)) )

    #U  = reshape( U,  (length(z), length(r)) )
    #B  = reshape( B,  (length(z), length(r)) )

    r_interp = collect(LinRange(minimum(r), maximum(r), 1000))
    z_interp = collect(LinRange(minimum(z), maximum(z), 100) )

    Δr = r_interp[2] - r_interp[1]
    Δz = z_interp[2] - z_interp[1]
	cutoff_freq_x = 0.12/Δr  # Cutoff frequency
	fs_x = 1.0/Δr
	cutoff_freq_y = 0.12/Δz  # Cutoff frequency
	fs_y = 1.0/Δz

    #U_interp = Interp2D_eigenFun(r, z, U, r_interp, z_interp)
    #B_interp = Interp2D_eigenFun(r, z, B, r_interp, z_interp)

    fig = Figure(fontsize=30, size = (1800, 580), )

    ax1 = Axis(fig[1, 1], xlabel=L"$r/R$", xlabelsize=30, ylabel=L"$z/H$", ylabelsize=30)

    interp_ = Interp2D_eigenFun(r, z, uᵣ, r_interp, z_interp)
    interp_ = low_PassFilter_y(cutoff_freq_y, fs_y, interp_)
	interp_ = low_PassFilter_x(cutoff_freq_x, fs_x, interp_)
    max_val = maximum(abs.(interp_))
    levels = range(-0.7max_val, 0.7max_val, length=16)
    co = contourf!(r_interp, z_interp, interp_, colormap=cgrad(:RdBu, rev=false),
        levels=levels, extendlow = :auto, extendhigh = :auto )

    # levels = range(minimum(U), maximum(U), length=8)
    # contour!(r_interp, z_interp, U_interp, levels=levels, linestyle=:dash, color=:black, linewidth=2) 

    # contour!(rn, zn, AmS, levels=levels₋, linestyle=:dash,  color=:black, linewidth=2) 
    # contour!(rn, zn, AmS, levels=levels₊, linestyle=:solid, color=:black, linewidth=2) 

    tightlimits!(ax1)
    cbar = Colorbar(fig[1, 2], co)
    xlims!(0.0, maximum(r))
    ylims!(0.0, maximum(z))

    ax2 = Axis(fig[1, 3], xlabel=L"$r/R$", xlabelsize=30, ylabel=L"$z/H$", ylabelsize=30)

    interp_ = Interp2D_eigenFun(r, z, w, r_interp, z_interp)
    interp_ = low_PassFilter_y(cutoff_freq_y, fs_y, interp_)
	interp_ = low_PassFilter_x(cutoff_freq_x, fs_x, interp_)
    max_val = maximum(abs.(interp_))
    levels = range(-0.7max_val, 0.7max_val, length=16)
    co = contourf!(r_interp, z_interp, interp_, colormap=cgrad(:RdBu, rev=false),
        levels=levels, extendlow = :auto, extendhigh = :auto )

    # levels = range(minimum(U), maximum(U), length=8)
    # contour!(r_interp, z_interp, U_interp, levels=levels, linestyle=:dash, color=:black, linewidth=2) 
        
    # contour!(rn, zn, AmS, levels=levels₋, linestyle=:dash,  color=:black, linewidth=2) 
    # contour!(rn, zn, AmS, levels=levels₊, linestyle=:solid, color=:black, linewidth=2) 

    tightlimits!(ax2)
    cbar = Colorbar(fig[1, 4], co)
    xlims!(0.0, maximum(r))
    ylims!(0.0, maximum(z))

    ax1.title = L"$\mathfrak{R}(\hat{u}_r)$"
    ax2.title = L"$\mathfrak{R}(\hat{w})$"

    fig
    filename = "AAI2d_" * string(it) * ".png"
    save(filename, fig, px_per_unit=4)

end


@with_kw mutable struct Grid{Nr, Nz, T} 
    Dʳ::Array{T, 2}   = SparseMatrixCSC(Zeros(Nr, Nr))
    Dᶻ::Array{T, 2}   = SparseMatrixCSC(Zeros(Nz, Nz))
    Dʳʳ::Array{T, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    Dᶻᶻ::Array{T, 2}  = SparseMatrixCSC(Zeros(Nz, Nz))
    r::Vector{T}      = zeros(Float64, Nr)
    z::Vector{T}      = zeros(Float64, Nz)
end

@with_kw mutable struct Operator{N, T} 
    𝒟ʳ::Array{T, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟ᶻ::Array{T, 2}  = SparseMatrixCSC(Zeros(N, N))
    𝒟²::Array{T, 2}  = SparseMatrixCSC(Zeros(N, N))
end

@with_kw mutable struct MeanFlow{N, T} 
    U₀::Array{T, 1}    = zeros(Float64, N) #SparseMatrixCSC(Zeros(N))
    Ω₀::Array{T, 1}    = zeros(Float64, N)
    B₀::Array{T, 1}    = zeros(Float64, N) #SparseMatrixCSC(Zeros(N))
    ζ₀::Array{T, 1}    = zeros(Float64, N) #SparseMatrixCSC(Zeros(N))

    ∂ʳU₀::Array{T, 1}  = zeros(Float64, N)
    ∂ʳB₀::Array{T, 1}  = zeros(Float64, N)
    ∂ᶻU₀::Array{T, 1}  = zeros(Float64, N)
    ∂ᶻΩ₀::Array{T, 1}  = zeros(Float64, N)
    ∂ᶻB₀::Array{T, 1}  = zeros(Float64, N)
end

function ddz(z)
    del = z[2] - z[1]
    N = length(z)
    d = zeros(N, N)
    for n in 2:N-1
        d[n, n-1] = -1.0
        d[n, n+1] = 1.0
    end
    d[1, 1] = -3.0; d[1, 2]   = 4.0;  d[1, 3]   = -1.0; 
    d[N, N] = 3.0;  d[N, N-1] = -4.0; d[N, N-2] = 1.0
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

function spectral_zderiv(var, z, L, parity, nmax)
    @assert ndims(var) == 2
    @assert length(z)  == size(var)[2]
    @assert parity == "cos" || parity == "sin"
    n   = 1:1:nmax
    Ay  = zeros(size(var))
    ny = size(var)[1]
    if parity == "cos"
        for it in n
            m = it*π/L
            @inbounds for iy ∈ 1:ny
                An  = trapz((z), var[iy,:] .* cos.(m*z)) * 2.0/L
                @. Ay[iy,:] += -An * sin(m*z) * m
            end
        end
    else
        for it in n
            m = it*π/L
            @inbounds for iy ∈ 1:ny
                An  = trapz((z), var[iy,:] .* sin.(m*z)) * 2.0/L
                @. Ay[iy,:] += An * cos(m*z) * m
            end
        end
    end
    return Ay
end

function ChebMatrix!(grid, params)
    # ------------- setup discrete diff matrices  -------------------
    ## chebyshev in y-direction
    grid.r, grid.Dʳ  = chebdif(params.Nr, 1)
    grid.r, grid.Dʳʳ = chebdif(params.Nr, 2)
    # Transform [0, Rₘₐₓ]
    grid.r, grid.Dʳ, grid.Dʳʳ = chebder_transform(grid.r, 
                                                grid.Dʳ, 
                                                grid.Dʳʳ, 
                                                zerotoL_transform, 
                                                params.R)

    # @printf "maximum(grid.Dʳ * grid.r): %f \n" maximum(grid.Dʳ * grid.r)

    ## chebyshev in z-direction
    grid.z, grid.Dᶻ  = chebdif(params.Nz, 1)
    grid.z, grid.Dᶻᶻ = chebdif(params.Nz, 2)
    # Transform the domain and derivative operators from [-1, 1] → [0, H]
    grid.z, grid.Dᶻ, grid.Dᶻᶻ = chebder_transform(grid.z, 
                                                grid.Dᶻ, 
                                                grid.Dᶻᶻ, 
                                                zerotoL_transform, 
                                                params.H)

    # grid.z, grid.Dᶻ = cheb(params.Nz-1)
    # grid.Dᶻᶻ = grid.Dᶻ * grid.Dᶻ

    grid.r    = range(0.0, stop=params.R, length=params.Nr) |> collect
    grid.Dʳ   = ddz(  grid.r );
    grid.Dʳʳ  = ddz2( grid.r );

    grid.z    = range(0.0, stop=params.H, length=params.Nz) |> collect
    grid.Dᶻ   = ddz(  grid.z );
    grid.Dᶻᶻ  = ddz2( grid.z );

    # grid.r   = collect(range(0.0, stop=params.R, length=params.Nr));
    # grid.Dʳ  = ddz_4(  grid.r ); 
    # grid.Dʳʳ = ddz2_4( grid.r );

    # grid.z   = collect(range(0.0, stop=params.H, length=params.Nz));
    # grid.Dᶻ  = ddz_4(  grid.z ); 
    # grid.Dᶻᶻ = ddz2_4( grid.z );

    @printf "grid.r[1], grid.r[2]: %f %f \n" grid.r[1] grid.r[2]

    @assert maximum(grid.r) ≈ params.R
    @assert maximum(grid.z) ≈ params.H

    return nothing
end

function ChebDiff_Matrix!(Op, params, grid, T)
    N  = params.Nr * params.Nz
    Iʳ = sparse(1.0I, params.Nr, params.Nr) 
    Iᶻ = sparse(1.0I, params.Nz, params.Nz) 
    I⁰ = Eye{T}(N)

    R, Z = ndgrid(grid.r, grid.z)
    R  = transpose(R); Z = transpose(Z); 
    R  = R[:]; Z = Z[:];
    R² = @. R^2;

    𝒟ʳʳ::Array{T, 2} = SparseMatrixCSC(Zeros(N, N))
    𝒟ᶻᶻ::Array{T, 2} = SparseMatrixCSC(Zeros(N, N))

    kron!( Op.𝒟ʳ, grid.Dʳ , Iᶻ )
    kron!( 𝒟ʳʳ  , grid.Dʳʳ, Iᶻ )
    kron!( Op.𝒟ᶻ, Iʳ, grid.Dᶻ  )
    kron!( 𝒟ᶻᶻ  , Iʳ, grid.Dᶻᶻ )

    @testset "Checking derivative operators ..." begin
        t1 = Op.𝒟ᶻ * Z;
        @test maximum(t1) ≈ 1.0 atol=1.0e-5
        @test minimum(t1) ≈ 1.0 atol=1.0e-5
        t1 = Op.𝒟ʳ * R;
        @test maximum(t1) ≈ 1.0 atol=1.0e-5
        @test minimum(t1) ≈ 1.0 atol=1.0e-5
        n::Int32 = 2
        p1 = @. Z^n; 
        t1 = 𝒟ᶻᶻ * p1;
        @test maximum(t1) ≈ factorial(n) atol=1.0e-5
        @test minimum(t1) ≈ factorial(n) atol=1.0e-5
        p1 = @. R^n; 
        t1 = 𝒟ʳʳ * p1;
        @test maximum(t1) ≈ factorial(n) atol=1.0e-5
        @test minimum(t1) ≈ factorial(n) atol=1.0e-5
    end

    R[R .== 0.0] .= 1.0e-6
    R⁻¹ = diagm(   1.0 ./ R    )
    R⁻² = diagm(  1.0 ./ R.^2  )
    # R₀   = diagm(   1.0 .* R   )
    # R₀²  = diagm(   1.0 .* R²  )

    # diffusivity operator
    Op.𝒟² = @. -1.0params.E * (  1.0 * 𝒟ʳʳ 
                            + 1.0 * R⁻¹ * Op.𝒟ʳ 
                            - 1.0 * params.m^2 * R⁻² * I⁰
                            + 1.0/params.ε^2 * 𝒟ᶻᶻ );
    return nothing
end


function meanflow!(mf, Op, params, grid, T)
    # file = matopen("eddy_structure_nd.mat");
    # rn   = transpose( read(file, "r" ) )[:,1];
    # zn   = transpose( read(file, "z" ) )[:,1];
    # Un   = read(file, "U"  ); Bn   = read(file, "B"  );
    # Ur   = read(file, "Ur" ); Uz   = read(file, "Uz" );
    # Br   = read(file, "Br" ); Bz   = read(file, "Bz" );
    # ζn   = read(file, "Vor"); Ro   = read(file, "Ro" )
    # close(file)

    file = jldopen("eddy_structure_nd_72hrs.jld2")
    rn   = file["r"];   zn   = file["z"]
    Un   = file["U"];   Bn   = file["B"]
    Ur   = file["drU"]; Uz   = file["dzU"]
    Ωn   = file["Ω"];   Ωz   = file["dzΩ"]
    Br   = file["drB"]; Bz   = file["dzB"]
    ζn   = file["ζ"]
    Ro   = file["Ro"]
    close(file)

    rn = vec(rn)
    zn = vec(zn)

    @assert maximum(rn) ≥ maximum(grid.r) "asking for a domain larger than allowed!"

    println(size(Un))
    println(size(rn))
    println(size(zn))
    @printf "max value of r in file: %f \n" maximum(rn)

    R, Z = ndgrid(grid.r, grid.z)
    t  = zeros(params.Nr, params.Nz); 
    t .= R;
    t[1,1:params.Nz] .= 1.0e-6
    R⁻¹ = @. 1.0/t

    params.Ro = 1.05Ro

    # interpolate U and Bz
    U₀     = Interp2D(rn, zn, Un, grid); 
    Ω₀     = Interp2D(rn, zn, Ωn, grid); 
    B₀     = Interp2D(rn, zn, Bn, grid); 

    ∂ʳU₀   = Interp2D(rn, zn, Ur, grid); 
    ∂ᶻU₀   = Interp2D(rn, zn, Uz, grid); 
    ∂ᶻΩ₀   = Interp2D(rn, zn, Ωz, grid); 

    ∂ʳB₀   = Interp2D(rn, zn, Br, grid); 
    ∂ᶻB₀   = Interp2D(rn, zn, Bz, grid);
    
    ζ₀     = Interp2D(rn, zn, ζn, grid);
    
    println(size(U₀))


    U₀   = transpose( U₀ ); mf.U₀   =   U₀[:];
    Ω₀   = transpose( Ω₀ ); mf.Ω₀   =   Ω₀[:];
    B₀   = transpose( B₀ ); mf.B₀   =   B₀[:];

    ∂ʳU₀ = transpose(∂ʳU₀); mf.∂ʳU₀ = ∂ʳU₀[:];
    ∂ᶻU₀ = transpose(∂ᶻU₀); mf.∂ᶻU₀ = ∂ᶻU₀[:];
    ∂ᶻΩ₀ = transpose(∂ᶻΩ₀); mf.∂ᶻΩ₀ = ∂ᶻΩ₀[:];
    
    ∂ʳB₀ = transpose(∂ʳB₀); mf.∂ʳB₀ = ∂ʳB₀[:];
    ∂ᶻB₀ = transpose(∂ᶻB₀); mf.∂ᶻB₀ = ∂ᶻB₀[:];
    
    ζ₀   = transpose( ζ₀ ); mf.ζ₀   =   ζ₀[:];

    @printf "min/max values of U:  %f %f \n" minimum(U₀ )  maximum(U₀ )
    @printf "min/max values of ζ:  %f %f \n" minimum(ζ₀ )  maximum(ζ₀ )

    @printf "min/max value of ∂zU: %f %f \n" minimum(∂ᶻU₀) maximum(∂ᶻU₀)
    @printf "min/max value of ∂rU: %f %f \n" minimum(∂ʳU₀) maximum(∂ʳU₀)

    @printf "min/max value of ∂zB: %f %f \n" minimum(∂ᶻB₀) maximum(∂ᶻB₀)
    @printf "min/max value of ∂rB: %f %f \n" minimum(∂ʳB₀ ) maximum(∂ʳB₀ )

    return nothing
end

function construct_lhs_matrix(params)
    T::Type = Float64
    N       = params.Nr * params.Nz
    grid    = Grid{params.Nr, params.Nz, T}() 
    Op      = Operator{N, T}()
    mf      = MeanFlow{N, T}()
    ChebMatrix!(grid, params)
    ChebDiff_Matrix!(Op, params, grid, T)
    meanflow!(mf, Op, params, grid, T)

    Ω    = sparse( diagm(mf.Ω₀) )
    ζ    = sparse( diagm(mf.ζ₀) )
    ∇ʳU  = sparse(diagm(mf.∂ʳU₀))
    ∇ᶻU  = sparse(diagm(mf.∂ᶻU₀))
    ∇ᶻΩ  = sparse(diagm(mf.∂ᶻΩ₀))
    ∇ʳB  = sparse(diagm(mf.∂ʳB₀))
    ∇ᶻB  = sparse(diagm(mf.∂ᶻB₀))

    @printf "Rossby number: %f \n" params.Ro

    R, Z = ndgrid(grid.r, grid.z)
    R  = transpose(R); 
    R  = R[:]; 
    R² = @. R^2

    R₀   = sparse(diagm(   1.0 .* R   ))
    R₀²  = sparse(diagm(   1.0 .* R²  ))
    R[R .== 0.0] .= 1.0e-6
    R⁻¹::Array{T, 2} = SparseMatrixCSC(Zeros(N, N))
    R⁻²::Array{T, 2} = SparseMatrixCSC(Zeros(N, N))
    R⁻¹ = sparse( diagm( 1.0 ./ R ) )
    R⁻² = sparse( diagm( 1.0 ./ R²) )
    
    I⁰  = sparse(1.0I, N, N) 
    im_m = 1.0im * params.m
    tmp = sparse(1.0 * Op.𝒟² + 1.0im_m * params.Ro * Ω * I⁰)
    
    # -------- stuff required for boundary conditions -------------
    ri, zi = ndgrid(1:1:params.Nr, 1:1:params.Nz)
    ri     = transpose(ri); 
    zi     = transpose(zi);
    ri     = ri[:]; zi   = zi[:];
    bcʳ₁   = findall( x -> (x==1),                  ri );
    bcʳ₂   = findall( x -> (x==params.Nr),          ri );
    bcʳ    = findall( x -> (x==1) | (x==params.Nr), ri );
    bcᶻ    = findall( x -> (x==1) | (x==params.Nz), zi );

    Tc::Type = ComplexF64

    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2);
    𝓛₁ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₂ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₃ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₄ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₅ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    B  = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));

    ε² = params.ε^2
    @printf "ε²: %f \n" ε²
    
    # lhs of the matrix (size := 5 × 5)
    # eigenvectors: [ur uθ w p b]ᵀ
    # ur-momentum equation 
    𝓛₁[:,    1:1s₂] = 1.0 * tmp + 1.0params.E * R⁻² * I⁰
    𝓛₁[:,1s₂+1:2s₂] = (-1.0 * I⁰ 
                    - 2.0params.Ro * Ω * I⁰
                    + 2.0im_m * params.E * R⁻² * I⁰)
    𝓛₁[:,3s₂+1:4s₂] = 1.0 * Op.𝒟ʳ
    # bc for `ur' in r-direction 
    if params.m == 0.0
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₁[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₁[bcʳ₂, :] = B[bcʳ₂, :]
    elseif params.m == 1.0
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * Op.𝒟ʳ; 𝓛₁[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₁[bcʳ₂, :] = B[bcʳ₂, :]
    else
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₁[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₁[bcʳ₂, :] = B[bcʳ₂, :]
    end
    # bc for `ur' in z-directon
    B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * Op.𝒟ᶻ;     𝓛₁[bcᶻ, :] = B[bcᶻ, :]

    # uθ-momentum equation
    𝓛₂[:,    1:1s₂] = (1.0params.Ro * ζ * I⁰
                    + 1.0 * I⁰ 
                    - 2.0im_m * params.E * R⁻² * I⁰)
    𝓛₂[:,1s₂+1:2s₂] = 1.0 * tmp + 1.0params.E * R⁻² * I⁰
    𝓛₂[:,2s₂+1:3s₂] = 1.0params.Ro * R₀ * ∇ᶻΩ * I⁰
    𝓛₂[:,3s₂+1:4s₂] = 1.0im_m * R⁻¹ * I⁰
    # bc for `uθ' in r-direction
    if params.m == 0.0
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰;    𝓛₂[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰;    𝓛₂[bcʳ₂, :] = B[bcʳ₂, :]
    elseif params.m == 1.0
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₂[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0im * I⁰;  𝓛₂[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰;    𝓛₂[bcʳ₂, :] = B[bcʳ₂, :]
    else
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰;    𝓛₂[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰;    𝓛₂[bcʳ₂, :] = B[bcʳ₂, :]
    end
    # bc for `uθ' in z-direction
    B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * Op.𝒟ᶻ;     𝓛₂[bcᶻ, :]  = B[bcᶻ, :]

    # w-momentum equation 
    𝓛₃[:,2s₂+1:3s₂] =  1.0    * tmp
    𝓛₃[:,3s₂+1:4s₂] =  1.0/ε² * Op.𝒟ᶻ
    𝓛₃[:,4s₂+1:5s₂] = -1.0/ε² * I⁰
    # bc for `w' in r-direction 
    if params.m == 0.0
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * Op.𝒟ʳ; 𝓛₃[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰;    𝓛₃[bcʳ₂, :] = B[bcʳ₂, :]
    else
        B .= 0.0; B = sparse(B); B[:,2s₂+1:3s₂] = 1.0 * I⁰;    𝓛₃[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,2s₂+1:3s₂] = 1.0 * I⁰;    𝓛₃[bcʳ₂, :] = B[bcʳ₂, :]
    end
    # bc for `w' in z-direction
    B .= 0.0; B = sparse(B); B[:,2s₂+1:3s₂] = 1.0 * I⁰; 𝓛₃[bcᶻ, :]  = B[bcᶻ, :]

    # ∇⋅u⃗ = 0 
    𝓛₄[:,    1:1s₂] = 1.0 * I⁰ + 1.0 * R₀ * Op.𝒟ʳ 
    𝓛₄[:,1s₂+1:2s₂] = 1.0im_m * I⁰
    𝓛₄[:,2s₂+1:3s₂] = 1.0 * R₀ * Op.𝒟ᶻ
    # bc for `p' in r-direction 
    if params.m == 0.0
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * Op.𝒟ʳ; 𝓛₄[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * Op.𝒟ʳ; 𝓛₄[bcʳ₂, :] = B[bcʳ₂, :]
    else
        B .= 0.0; B = sparse(B); B[:,3s₂+1:4s₂] = 1.0 * I⁰;    𝓛₄[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,3s₂+1:4s₂] = 1.0 * I⁰;    𝓛₄[bcʳ₂, :] = B[bcʳ₂, :]
    end
    # bc for `p' in z-direction
    B .= 0.0; B = sparse(B); B[:,3s₂+1:4s₂] = 1.0 * Op.𝒟ᶻ;     𝓛₄[bcᶻ, :]  = B[bcᶻ, :]

    # buoyancy equation
    𝓛₅[:,    1:1s₂] = 1.0params.Ro * ∇ʳB * I⁰
    𝓛₅[:,2s₂+1:3s₂] = 1.0params.Ro * ∇ᶻB * I⁰
    𝓛₅[:,4s₂+1:5s₂] = 1.0 * tmp
    # bc for `b' in r-direction 
    B .= 0.0; B = sparse(B); B[:,4s₂+1:5s₂] = 1.0 * I⁰;    𝓛₅[bcʳ₁, :] = B[bcʳ₁, :]
    B .= 0.0; B = sparse(B); B[:,4s₂+1:5s₂] = 1.0 * I⁰;    𝓛₅[bcʳ₂, :] = B[bcʳ₂, :]
    # bc for `b' in z-direction
    B .= 0.0; B = sparse(B); B[:,4s₂+1:5s₂] = 1.0 * Op.𝒟ᶻ; 𝓛₅[bcᶻ, :]  = B[bcᶻ, :]

    𝓛 = sparse([𝓛₁; 𝓛₂; 𝓛₃; 𝓛₄; 𝓛₅]);
    ℳ = construct_rhs_matrix(params, grid)

    return grid.r, grid.z, 𝓛, ℳ
end


function construct_rhs_matrix(params, grid, T::Type=Float64)
    N  = params.Nr*params.Nz
    I₀ = sparse(T, 1.0I, N, N) 

    ri, zi = ndgrid(1:1:params.Nr, 1:1:params.Nz)
    ri  = transpose(ri); 
    zi  = transpose(zi)
    ri  = ri[:]; zi = zi[:]
    bcʳ = findall( x -> (x==1) | (x==params.Nr), ri )
    bcᶻ = findall( x -> (x==1) | (x==params.Nz), zi )

    s₁ = size(I₀, 1); s₂ = size(I₀, 2)
    ℳ₁ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₂ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₃ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₄ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₅ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));

    #       |1   0   0   0   0|
    #       |0   1   0   0   0|
    # M = - |0   0   1   0   0| * m
    #       |0   0   0   0   0|
    #       |0   0   0   0   1|

    ℳ₁[:,    1:1s₂] = -1.0I₀; 
    ℳ₂[:,1s₂+1:2s₂] = -1.0I₀; 
    ℳ₃[:,2s₂+1:3s₂] = -1.0I₀; 
    ℳ₅[:,4s₂+1:5s₂] = -1.0I₀; 

    ℳ₁[bcʳ, :] .= 0.0;  ℳ₁[bcᶻ, :] .= 0.0;
    ℳ₂[bcʳ, :] .= 0.0;  ℳ₂[bcᶻ, :] .= 0.0;
    ℳ₃[bcʳ, :] .= 0.0;  ℳ₃[bcᶻ, :] .= 0.0;
    ℳ₅[bcʳ, :] .= 0.0;  ℳ₅[bcᶻ, :] .= 0.0;

    ℳ = sparse([ℳ₁; ℳ₂; ℳ₃; ℳ₄; ℳ₅])

    return ℳ
end


@with_kw mutable struct Params{T<:Real} @deftype T
    R::T     = 3.0
    H::T     = 1.0   
    E::T     = 1.0e-8
    ε::T     = 0.1
    Nr::Int  = 120
    Nz::Int  = 40
    m::T     = 2.0
    Ro::T    = 1.0
end

struct ShiftAndInvert{TA,TB,TT}
    A_lu::TA
    B::TB
    temp::TT
end

function (M::ShiftAndInvert)(y, x)
    mul!(M.temp, M.B, x)
    ldiv!(y, M.A_lu, M.temp)
end

function construct_linear_map(A, B)
    a = ShiftAndInvert(factorize(A), B, Vector{eltype(A)}(undef, size(A,1)))
    LinearMap{eltype(A)}(a, size(A,1), ismutating=true)
end

function construct_linear_map(H, S, num_thread)
    ps = MKLPardisoSolver()
    set_nprocs!(ps, num_thread) 
    set_matrixtype!(ps, Pardiso.COMPLEX_NONSYM)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    H_pardiso = get_matrix(ps, H, :N)
    b = rand(ComplexF64, size(H, 1))
    set_phase!(ps, Pardiso.ANALYSIS)
    #set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    pardiso(ps, H_pardiso, b)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, H_pardiso, b)
    return (LinearMap{ComplexF64}(
            (y, x) -> begin
                set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
                pardiso(ps, y, H_pardiso, S * x)
            end,
            size(H, 1);
            ismutating=true), ps)
end

function solve_AAI2d(m, λref, ra, rb) #,λₛ)
    params = Params{Float64}(m=1.0m)
    @printf "Ekman number: %1.1e \n" params.E
    @info("Start matrix constructing ...")

    r, z, 𝓛, ℳ = construct_lhs_matrix(params)
    @info("Matrix construction done ...")

    @. 𝓛 *= 1.0/params.m 

    @printf "Matrix size is: %d × %d \n" size(𝓛, 1) size(𝓛, 2)

    N = params.Nr * params.Nz
    MatSize = 5N 

    # mmwrite("60_360_julia/systemA_" * string(trunc(Int32, m+1e-2)) * ".mtx", sparse(𝓛))
    # mmwrite("60_360_julia/systemB_" * string(trunc(Int32, m+1e-2)) * ".mtx", sparse(ℳ))

    #* Method: 1
    @info("Eigensolver using `implicitly restarted Arnoldi method' ...")
    decomp, history = partialschur(
        construct_linear_map(𝓛, ℳ), nev=5000, tol=0.0, restarts=300, which=LR()
        )
    λₛ⁻¹, Χ = partialeigen(decomp)  
    λₛ = @. 1.0 / λₛ⁻¹ #* -1.0*im

    ## Method: 2
    # lm, ps  = construct_linear_map(𝓛, ℳ, 40)
    # @info("Construction of linear map is done!")
    # λₛ⁻¹, Χ = eigs(lm, tol=1e-6, maxiter=100, nev=2000, which=:LR)
    # # Release all internal memory for all matrices
    # set_phase!(ps, Pardiso.RELEASE_ALL) 
    # pardiso(ps)
    # ## Eigenvalues have to be inverted to find the smallest eigenvalues of the non-inverted problem.
    # λₛ = @. 1.0 / λₛ⁻¹ #* -1.0*im
   
    #* Method: 3
    # σ = 3e-3 + 0.21im
    # printstyled("Eigensolver using Arpack eigs with shift and invert method ...\n"; 
    #                 color=:red)
    # σᵢ = 0.15:0.02:0.35
    # λₐ = []
    # for σᵢ⁰ in σᵢ
    #     try
    #         λₛ, Χ = EigSolver_shift_invert(𝓛, ℳ, σ₀=σ.re + 1im * σᵢ⁰)
    #         @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
    #         error = 1.0
    #         Iter::Int = 0
    #         λₜ = []
    #         while error > 5e-3 && Iter < 5
    #             λₛ_old = λₛ
    #             λₛ, Χ = EigSolver_shift_invert1(𝓛, ℳ, σ₀=λₛ[1])
    #             @printf "found eigenvalue: %f + im %f \n" λₛ[1].re λₛ[1].im
    #             error = abs(λₛ[1] - λₛ_old[1])/abs(λₛ[1])
    #             Iter += 1
    #             append!(λₜ, λₛ)
    #         end
    #         λₘₐₓ = maximum(real(λₜ))
    #         loc::Int = 0
    #         for it in 1:length(λₜ)
    #             if real(λₜ[it]) == λₘₐₓ
    #                 loc = it
    #             end
    #         end
    #         append!(λₐ, 1.0λₜ[loc])
    #     catch error
    #        append!(λₐ, 0.0)
    #     end
    # end
    # println(λₐ)

    # λₘₐₓ = maximum(real(λₐ))
    # loc  = 0
    # for it in 1:length(λₐ)
    #     if real(λₐ[it]) == λₘₐₓ
    #         loc = it
    #         println(loc)
    #     end
    # end
    # λₛ = 1.0λₐ[loc]  
    # #λₛ = 0.0041579 + 0.22866im
    # λₛ, Χ = Arpack.eigs(𝓛, ℳ,
    #                     nev     = 1, 
    #                     tol     = 1.0e-10, 
    #                     maxiter = 30, 
    #                     which   = :LR,
    #                     sigma   = 0.99λₛ,)
    
    
    ###FEAST parameters
    T::Type             = Float64
    emid::ComplexF64    = λref #complex(0.1, 0.0)   #contour center
    ra::T               = ra                        #contour radius 1
    rb::T               = rb                        #contour radius 2
    nc::Int64           = 100                       #number of contour points
    m₀::Int64           = 40                        #subspace dimension
    ε::T                = 1.0e-5                    #residual convergence tolerance
    maxit::Int64        = 100                       #maximum FEAST iterations
    x₀                  = sprand(ComplexF64, MatSize, m₀, 0.1)   #eigenvector initial guess
    # @info("Standard FEAST!")
    # @printf "size of the subspace: %d \n" m₀
    # @printf "no. of contour points: %d \n" nc
    # @printf "contour radius: %f \n" ra
    # λₛ, Χ = feast_linear(𝓛, ℳ, x₀, nc, emid, ra, rb, ε, 0.0, 0.0+0.0im, maxit)

    #contour    = circular_contour_trapezoidal(emid, ra, 50)
    #λₛ, Χ, res = gen_feast!(x₀, 𝓛, ℳ, contour, iter=maxit, debug=true, ϵ=ε)

#################

    cnst = 1.0params.m
    λₛ = @. λₛ * cnst
    @assert length(λₛ) ≥ 1 "No eigenvalue(s) found!"

    ## Post Process egenvalues
    ## removes the magnitude of eigenvalue ≥ min value and ≤ max value
    # Option: "M" : magnitude, "R" : real, "I" : imaginary 
    λₛ, Χ = remove_evals(λₛ, Χ, 0.0, 1.0e2, "I") 
    # sorting the eignevalues λₛ (real part: growth rate) based on maximum value 
    # and corresponding eigenvectors Χ
    λₛ, Χ = sort_evals(λₛ, Χ, "R", "lm")

    #= 
    this removes any further spurious eigenvalues based on norm 
    if you don't need it, just `comment' it!
    =#
    @show norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) 
    # while norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) > 1.0e-4 #|| imag(λₛ[1]) < 0.0
    #     @printf "norm: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) 
    #     λₛ, Χ = remove_spurious(λₛ, Χ)
    #     println(λₛ[1])        
    # end

    print_evals(λₛ/cnst, length(λₛ))

    𝓛 = nothing
    ℳ = nothing

    #return nothing #λₛ #[1:5] #[1:3], Χ #, uᵣ, uₜ, w, b
    return r, z, λₛ #[1] #, Χ, N #uᵣ, uₜ, w, b
end


function EigsSolver()
    m = collect(LinRange(1, 25, 25))
    λₛ = Array{ComplexF64}(undef, length(m))

    λref = complex(0.06, 0.16)
    ra = 0.06
    rb = ra 
    
    for it ∈ 7:1:7
        @printf "value of m: %f \n" m[it]
        
        # if it == 1
        #     λ = 0.0
        # else
        #     λ = λₛ[it-1]/m[it-1]
        # end

        @time r, z, λₛ =  #[it] = #, Χ, N = 
                    solve_AAI2d(m[it], λref, ra, rb) #, it, λ)

        # twoDContour(r, z, Χ[:,1], N, it)
        # jldsave("pertubation_60360_m" * string(it) * ".jld2"; 
        #                             r=r, 
        #                             z=z,
        #                             N=N, 
        #                             m=m[it],
        #                             λₛ=λₛ[it],
        #                             Xre=Χ[:,1], )

        #@time solve_AAI2d(m[it], λref, ra, rb)

        # which::Int = 1
        # uᵣ = cat( real(Χ[   1:1N, which]), imag(Χ[   1:1N, which]), dims=2 ); 
        # uₜ = cat( real(Χ[1N+1:2N, which]), imag(Χ[1N+1:2N, which]), dims=2 ); 
        # w  = cat( real(Χ[2N+1:3N, which]), imag(Χ[2N+1:3N, which]), dims=2 ); 
        # p  = cat( real(Χ[3N+1:4N, which]), imag(Χ[3N+1:4N, which]), dims=2 ); 
        # b  = cat( real(Χ[4N+1:5N, which]), imag(Χ[4N+1:5N, which]), dims=2 ); 
        # save("matrices_60/first_eigenfun_m" * string(trunc(Int32, m[it]+1e-2)) *".jld", 
        #    "r", r, "z", z, "ur", uᵣ, "ut", uₜ, "w", w, "b", b)

        # which = 2
        # uᵣ = cat( real(Χ[   1:1N, which]), imag(Χ[   1:1N, which]), dims=2 ); 
        # uₜ = cat( real(Χ[1N+1:2N, which]), imag(Χ[1N+1:2N, which]), dims=2 ); 
        # w  = cat( real(Χ[2N+1:3N, which]), imag(Χ[2N+1:3N, which]), dims=2 ); 
        # p  = cat( real(Χ[3N+1:4N, which]), imag(Χ[3N+1:4N, which]), dims=2 ); 
        # b  = cat( real(Χ[4N+1:5N, which]), imag(Χ[4N+1:5N, which]), dims=2 );
        # save("matrices_60/second_eigenfun_m" * string(trunc(Int32, m[it]+1e-2)) *".jld", 
        #     "r", r, "z", z, "ur", uᵣ, "ut", uₜ, "w", w, "b", b)

        @printf("=================================================================== \n")
    end
    #save("GrowthRate_1_20_60_360.jld", "m", m, "ev", λₛ)
    save("GrowthRate_40_m7.jld", "ev", λₛ)
end

EigsSolver()

# file = matopen("AmS_structure.mat");
# rn   = transpose( read(file, "r" ) )[:,1];
# zn   = transpose( read(file, "z" ) )[:,1];
# AmS  = transpose( read(file, "AmS") );
# close(file)

# @printf "min/max of A-S: %f %f \n" minimum(AmS) maximum(AmS)
# levels  = LinRange(minimum(AmS), maximum(AmS), 12) 
# levels₋ = levels[findall( x -> (x ≤ 0.0), levels )]
# levels₊ = levels[findall( x -> (x > 0.0), levels )]

# m = 1.0
# λref = complex(0.08, 0.2)
# ra = 0.08
# rb = ra 
# @time U, r, z, λₛ, uᵣ, uₜ, w, b = solve_AAI2d(m, λref, ra, rb)
# #@time solve_AAI2d(m, λref, ra, rb)

# # save("eigenfun_m" * string(trunc(Int32, m+1e-2)) *".jld", 
# #     "r", r, "z", z, "ur", uᵣ, "ut", uₜ, "w", w, "b", b)

# U = diag(U)
# #B = diag(B)


