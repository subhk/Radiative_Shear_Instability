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
using JLD
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

using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod

include("feast.jl")
using ..feastLinear

# include("FEASTSolver/src/FEASTSolver.jl")
# using Main.FEASTSolver

include("dmsuite.jl")
include("transforms.jl")
include("utils.jl")

function Interp2D(rn, zn, An, grid)
    itp = BicubicInterpolator(rn, zn, An)
    #A₀ = zeros(Float64, length(grid.r), length(grid.z))
    A₀ = [itp(rᵢ, zᵢ) for rᵢ in grid.r, zᵢ in grid.z]
    # A₀ = transpose(A₀)
    # A₀ = A₀[:]
    return A₀
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
    U::Array{T, 2}    = SparseMatrixCSC(Zeros(N, N))
    Ω::Array{T, 2}    = SparseMatrixCSC(Zeros(N, N))
    ζ::Array{T, 2}    = SparseMatrixCSC(Zeros(N, N))
    ∇ᶻU::Array{T, 2}  = SparseMatrixCSC(Zeros(N, N))
    ∇ʳB::Array{T, 2}  = SparseMatrixCSC(Zeros(N, N))
    ∇ᶻB::Array{T, 2}  = SparseMatrixCSC(Zeros(N, N))
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
    N = params.Nr * params.Nz
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

    ## chebyshev in z-direction
    grid.z, grid.Dᶻ  = chebdif(params.Nz, 1)
    grid.z, grid.Dᶻᶻ = chebdif(params.Nz, 2)
    # Transform the domain and derivative operators from [-1, 1] → [-H, H]
    grid.z, grid.Dᶻ, grid.Dᶻᶻ = chebder_transform(grid.z, 
                                                grid.Dᶻ, 
                                                grid.Dᶻᶻ, 
                                                MinusLtoPlusL_transform, 
                                                params.H)

    grid.r      = collect(range(0.0, stop=params.R, length=params.Nr))
    grid.Dʳ     = ddz(  grid.r )
    grid.Dʳʳ    = ddz2( grid.r )

    grid.z      = collect(range(-params.H, stop=params.H, length=params.Nz))
    grid.Dᶻ     = ddz(  grid.z )
    grid.Dᶻᶻ    = ddz2( grid.z )

    @printf "grid.r[1], grid.r[2]: %f %f \n" grid.r[1] grid.r[2]

    @assert maximum(grid.r) ≈ params.R
    @assert maximum(grid.z) ≈ params.H && minimum(grid.z) ≈ -params.H

    return nothing

end


function ChebDiff_Matrix!(Op, params, grid, T)
    N       = params.Nr * params.Nz
    Iʳ      = sparse(1.0I, params.Nr, params.Nr) 
    Iᶻ      = sparse(1.0I, params.Nz, params.Nz) 
    I⁰      = Eye{T}(N)

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
        tol = 1.0e-6
        t1  = Op.𝒟ᶻ * Z;
        @test maximum(t1) ≈ 1.0 atol=tol
        @test minimum(t1) ≈ 1.0 atol=tol
        t1  = Op.𝒟ʳ * R;
        @test maximum(t1) ≈ 1.0 atol=tol
        @test minimum(t1) ≈ 1.0 atol=tol
        n::Int32 = 2
        p1  = @. Z^n; 
        t1  = 𝒟ᶻᶻ * p1;
        @test maximum(t1) ≈ factorial(n) atol=tol
        @test minimum(t1) ≈ factorial(n) atol=tol
        p1  = @. R^n; 
        t1  = 𝒟ʳʳ * p1;
        @test maximum(t1) ≈ factorial(n) atol=tol
        @test minimum(t1) ≈ factorial(n) atol=tol
    end

    R[R .== 0.0] .= 1.0e-5
    R⁻¹ = diagm(   1.0 ./ R    )
    R⁻² = diagm(  1.0 ./ R.^2  )

    # diffusivity operator
    Ek = params.Ro / params.Re
    Op.𝒟² = @. -1.0Ek * ( 1.0 * 𝒟ʳʳ 
                        + 1.0 * R⁻¹ * Op.𝒟ʳ 
                        - 1.0 * params.m^2 * R⁻² * I⁰
                        + 1.0/params.α^2 * 𝒟ᶻᶻ );

    return nothing
end


function meanflow!(mf, Op, params, grid, T)

    R, Z = ndgrid(grid.r, grid.z)

    fₛ   = 1.0/params.Ro  # f/Ω₀
    Ω₀   = @. 1.0 * exp(-R^2.0 - Z^2.0)
    Ω₀²  = @. 1.0 * exp(-2.0*R^2.0 - 2.0*Z^2.0)
    ∂rΩ₀ = @. -2.0 * R * Ω₀
    ∂zΩ₀ = @. -2.0 * Z * Ω₀

    U₀   = @. 1.0 * R * Ω₀
    ζ₀   = @. 2.0 * Ω₀ + 1.0 * R * ∂rΩ₀
    ∂ᶻU₀ = @. 1.0 * R * ∂zΩ₀  

    ∂ʳB₀ = @. -1.0params.Ro * Z * (2.0 * ∂rΩ₀ * Ω₀ + 1.0 * fₛ * ∂rΩ₀) 
    ∂ᶻB₀ = @. ( -1.0params.Ro * (1.0 * Ω₀ * Ω₀ + 1.0 * fₛ * Ω₀) 
            - 1.0params.Ro * Z * (2.0 * Ω₀ * ∂zΩ₀ + 1.0 * fₛ * ∂zΩ₀) )

    Ω₀   = transpose( Ω₀ ); Ω₀   =   Ω₀[:];
    U₀   = transpose( U₀ ); U₀   =   U₀[:];
    ζ₀   = transpose( ζ₀ ); ζ₀   =   ζ₀[:];
    ∂ᶻU₀ = transpose(∂ᶻU₀); ∂ᶻU₀ = ∂ᶻU₀[:];
    ∂ʳB₀ = transpose(∂ʳB₀); ∂ʳB₀ = ∂ʳB₀[:];
    ∂ᶻB₀ = transpose(∂ᶻB₀); ∂ᶻB₀ = ∂ᶻB₀[:];
    
    mf.U    = sparse( diagm(U₀) )
    mf.Ω    = sparse( diagm(Ω₀) )
    mf.ζ    = sparse( diagm(ζ₀) )

    mf.∇ᶻU  = sparse(diagm(∂ᶻU₀))
    mf.∇ʳB  = sparse(diagm(∂ʳB₀))
    mf.∇ᶻB  = sparse(diagm(∂ᶻB₀))

    @printf "min/max of U: %f %f \n" minimum(mf.U) maximum(mf.U)
    @printf "min/max of ζ: %f %f \n" minimum(mf.ζ) maximum(mf.ζ)

    @printf "min/max of ∇ᶻU: %f %f \n" minimum(mf.∇ᶻU) maximum(mf.∇ᶻU)
    @printf "min/max of ∇ᶻB: %f %f \n" minimum(mf.∇ᶻB) maximum(mf.∇ᶻB)
    @printf "min/max of ∇ʳB: %f %f \n" minimum(mf.∇ʳB) maximum(mf.∇ʳB)

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

    R, Z = ndgrid(grid.r, grid.z)
    R = transpose(R); 
    R  = R[:]; 
    R² = @. R^2

    R₀   = sparse(diagm(   1.0 .* R   ))
    R₀²  = sparse(diagm(   1.0 .* R²  ))
    R[R .== 0.0] .= 1.0e-5
    R⁻¹  = sparse( diagm( 1.0 ./ R ) )
    R⁻²  = sparse( diagm( 1.0 ./ R² ) )
    
    I⁰  = sparse(1.0I, N, N) 
    im_m = 1.0im * params.m 
    tmp = sparse(1.0 * Op.𝒟² + 1.0im_m * params.Ro * mf.Ω * I⁰)
    
    # -------- stuff required for boundary conditions -------------
    ri, zi = ndgrid(1:1:params.Nr, 1:1:params.Nz)
    ri   = transpose(ri); 
    zi   = transpose(zi)
    ri   = ri[:]; zi   = zi[:];
    bcʳ₁ = findall( x -> (x==1),                  ri );
    bcʳ₂ = findall( x -> (x==params.Nr),          ri );
    bcʳ  = findall( x -> (x==1) | (x==params.Nr), ri );
    bcᶻ  = findall( x -> (x==1) | (x==params.Nz), zi );

    Tc::Type = ComplexF64

    s₁ = size(I⁰, 1); 
    s₂ = size(I⁰, 2);
    𝓛₁ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₂ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₃ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₄ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    𝓛₅ = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));
    B  = SparseMatrixCSC(Zeros{Tc}(s₁, 5s₂));

    α² = 1.0params.α^2
    @printf "α²: %f \n" α²
    Ek = params.Ro / params.Re
    @printf "Ek: %f \n" Ek
    
    # lhs of the matrix (size := 5 × 5)
    # eigenvectors: [ur uθ w p b]ᵀ
    # ur-momentum equation 
    𝓛₁[:,    1:1s₂] = 1.0 * tmp + 1.0Ek * R⁻² * I⁰
    𝓛₁[:,1s₂+1:2s₂] = (-2.0params.Ro * mf.Ω * I⁰ 
                    - 1.0 * I⁰
                    + 2.0im_m * Ek * R⁻² * I⁰)
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
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₁[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;    𝓛₁[bcʳ₂, :] = B[bcʳ₂, :]
    end
    # bc for `ur' in z-directon
    B .= 0.0; B = sparse(B); B[:,    1:1s₂] = 1.0 * I⁰;        𝓛₁[bcᶻ, :] = B[bcᶻ, :]

    # uθ-momentum equation
    𝓛₂[:,    1:1s₂] = (1.0params.Ro * mf.ζ * I⁰
                    + 1.0 * I⁰
                    - 2.0im_m * Ek * R⁻² * I⁰)
    𝓛₂[:,1s₂+1:2s₂] = 1.0 * tmp + 1.0Ek * R⁻² * I⁰
    𝓛₂[:,2s₂+1:3s₂] = 1.0params.Ro * mf.∇ᶻU * I⁰
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
    B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰;        𝓛₂[bcᶻ, :]  = B[bcᶻ, :]

    # w-momentum equation 
    𝓛₃[:,2s₂+1:3s₂] = 1.0 * tmp
    𝓛₃[:,3s₂+1:4s₂] = 1.0/α² * Op.𝒟ᶻ
    𝓛₃[:,4s₂+1:5s₂] = 1.0/α² * I⁰
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
    B .= 0.0; B = sparse(B); B[:,2s₂+1:3s₂] = 1.0 * I⁰;        𝓛₃[bcᶻ, :]  = B[bcᶻ, :]

    # ∇⋅u⃗ = 0 
    𝓛₄[:,    1:1s₂] = 1.0 * I⁰ + 1.0 * R₀ * Op.𝒟ʳ 
    𝓛₄[:,1s₂+1:2s₂] = 1.0im_m * I⁰
    𝓛₄[:,2s₂+1:3s₂] = 1.0 * R₀ * Op.𝒟ᶻ
    # bc for `p' in r-direction 
    if params.m == 0.0
        @printf "m = %f \n" params.m
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰; 𝓛₄[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,1s₂+1:2s₂] = 1.0 * I⁰; 𝓛₄[bcʳ₂, :] = B[bcʳ₂, :]
    else
        B .= 0.0; B = sparse(B); B[:,3s₂+1:4s₂] = 1.0 * I⁰; 𝓛₄[bcʳ₁, :] = B[bcʳ₁, :]
        B .= 0.0; B = sparse(B); B[:,3s₂+1:4s₂] = 1.0 * I⁰; 𝓛₄[bcʳ₂, :] = B[bcʳ₂, :]
    end
    # bc for `p' in z-direction
    B .= 0.0; B = sparse(B); B[:,3s₂+1:4s₂] = 1.0 * I⁰;     𝓛₄[bcᶻ, :]  = B[bcᶻ, :]

    # buoyancy equation
    𝓛₅[:,    1:1s₂] = 1.0params.Ro * mf.∇ʳB * I⁰
    𝓛₅[:,2s₂+1:3s₂] = 1.0params.Ro * mf.∇ᶻB * I⁰ - 1.0params.Ro^2/params.Fr^2 * params.α^2 * I⁰
    𝓛₅[:,4s₂+1:5s₂] = 1.0 * tmp
    # bc for `b' in r-direction 
    B .= 0.0; B = sparse(B); B[:,4s₂+1:5s₂] = 1.0 * I⁰; 𝓛₅[bcʳ₁, :] = B[bcʳ₁, :]
    B .= 0.0; B = sparse(B); B[:,4s₂+1:5s₂] = 1.0 * I⁰; 𝓛₅[bcʳ₂, :] = B[bcʳ₂, :]
    # bc for `b' in z-direction
    B .= 0.0; B = sparse(B); B[:,4s₂+1:5s₂] = 1.0 * I⁰; 𝓛₅[bcᶻ, :]  = B[bcᶻ, :]

    𝓛 = sparse([𝓛₁; 𝓛₂; 𝓛₃; 𝓛₄; 𝓛₅]);
    ℳ = construct_rhs_matrix(params, grid)

    return grid.r, grid.z, 𝓛, ℳ
end


function construct_rhs_matrix(params, grid, T::Type=Float64)
    N  = params.Nr*params.Nz
    I₀ = sparse(Float64, 1.0I, N, N) 

    ri, zi = ndgrid(1:1:params.Nr, 1:1:params.Nz)
    ri  = transpose(ri); ri = ri[:];
    zi  = transpose(zi); zi = zi[:];
    bcʳ = findall( x -> (x==1) | (x==params.Nr), ri )
    bcᶻ = findall( x -> (x==1) | (x==params.Nz), zi )

    s₁ = size(I₀, 1); s₂ = size(I₀, 2)
    ℳ₁ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₂ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₃ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₄ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));
    ℳ₅ = SparseMatrixCSC(Zeros{T}(s₁, 5s₂));

    #      |1   0   0   0   0|
    #      |0   1   0   0   0|
    # M = -|0   0   1   0   0|
    #      |0   0   0   0   0|
    #      |0   0   0   0   1|

    ℳ₁[:,    1:1s₂] = -1.0params.Ro * I₀; 
    ℳ₂[:,1s₂+1:2s₂] = -1.0params.Ro * I₀; 
    ℳ₃[:,2s₂+1:3s₂] = -1.0params.Ro * I₀; 
    ℳ₅[:,4s₂+1:5s₂] = -1.0params.Ro * I₀; 

    ℳ₁[bcʳ, :] .= 0.0; ℳ₁[bcᶻ, :] .= 0.0;
    ℳ₂[bcʳ, :] .= 0.0; ℳ₂[bcᶻ, :] .= 0.0;
    ℳ₃[bcʳ, :] .= 0.0; ℳ₃[bcᶻ, :] .= 0.0;
    ℳ₅[bcʳ, :] .= 0.0; ℳ₅[bcᶻ, :] .= 0.0;

    ℳ = sparse([ℳ₁; ℳ₂; ℳ₃; ℳ₄; ℳ₅])
    
    return ℳ
end


@with_kw mutable struct Params{T<:Real} @deftype T
    R::T        = 10.0
    H::T        = 5.0   
    Re::T       = 1.0e4
    Sc::T       = 1.0
    Fr::T       = 0.5
    Ro::T       = 10.0/2.0
    α::T        = 0.5
    Nr::Int     = 200
    Nz::Int     = 200
    m::T        = 2.0
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


function solve_AAI2d(m, λref, ra, rb)
    params = Params{Float64}(m=m)
    @info("Start matrix constructing ...")
    @printf "azimuthal wavenumber, m: %f \n" params.m
    r, z, 𝓛, ℳ = construct_lhs_matrix(params)
    @info("Matrix construction done ...")

    mmwrite("pancake/m_1/Ro_10/systemA_" * string(trunc(Int32, m+1e-2)) * ".mtx", 𝓛)
    mmwrite("pancake/m_1/Ro_10/systemB_" * string(trunc(Int32, m+1e-2)) * ".mtx", ℳ)

    @printf "Matrix size is: %d × %d \n" size(𝓛, 1) size(𝓛, 2)

    N = params.Nr * params.Nz
    MatSize = 5N 

    #* Method: 1
    # @info("Eigensolver using `implicitly restarted Arnoldi method' ...")
    # decomp, history = partialschur(
    #     construct_linear_map(𝓛, ℳ), nev=2, tol=1.0e-8, restarts=300, which=SI()
    #     )
    # λₛ⁻¹, Χ = partialeigen(decomp)  
    # λₛ = @. 1.0 / λₛ⁻¹ #* -1.0*im
   
    #* Method: 3
    @info("Eigensolver using `Arpack eigs with shift and invert method' ...")
    λₛ, Χ = Arpack.eigs(𝓛, ℳ,
                        nev     = 2, 
                        tol     = 1.0e-8, 
                        maxiter = 40, 
                        which   = :LR,
                        sigma   = 0.005,)
    
    ###FEAST parameters
    # T::Type             = Float64
    # emid::ComplexF64    = λref #complex(0.1, 0.0)   #contour center
    # ra::T               = ra                        #contour radius 1
    # rb::T               = rb                        #contour radius 2
    # nc::Int             = 120                       #number of contour points
    # m₀::Int             = 30                        #subspace dimension
    # ε::T                = 1.0e-6                    #residual convergence tolerance
    # maxit::Int          = 100                       #maximum FEAST iterations
    # x₀                  = sprand(ComplexF32, MatSize, m₀, 0.1)   #eigenvector initial guess
    # @info("Standard FEAST!")
    # λₛ, Χ = feast_linear(𝓛, ℳ, x₀, nc, emid, ra, rb, ε, 0.0, 0.0+0.0im, maxit)

#################

    cnst = 1.0 #im #params.m
    @. λₛ *= cnst
    @assert length(λₛ) ≥ 1 "No eigenvalue(s) found!"
    # @printf "\n"

    ## Post Process egenvalues
    ## removes the magnitude of eigenvalue ≥ min value and ≤ max value
    # Option: "M" : magnitude, "R" : real, "I" : imaginary 
    λₛ, Χ = remove_evals(λₛ, Χ, 0.0, 1.0e1, "M") 
    # sorting the eignevalues λₛ (real part: growth rate) based on maximum value 
    # and corresponding eigenvectors Χ
    λₛ, Χ = sort_evals(λₛ, Χ, "R", "lm")

    #= 
    this removes any further spurious eigenvalues based on norm 
    if you don't need it, just `comment' it!
    =#
    @show norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) 
    while norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) > 1.0e-5 #|| imag(λₛ[1]) < 0.0
        @printf "norm: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1]/cnst * ℳ * Χ[:,1]) 
        λₛ, Χ = remove_spurious(λₛ, Χ)
        println(λₛ[1])        
    end

    print_evals(λₛ, length(λₛ))
    # @printf "largest eigenvalue: %1.5e%+1.5eim\n"  real(λₛ[1]) imag(λₛ[1])
    # @printf "norm: %f \n" norm(𝓛 * Χ[:,1] - λₛ[1]/(-im*params.m) * ℳ * Χ[:,1]) 

    which::Int = 1
    uᵣ = cat( real(Χ[   1:1N, which]), imag(Χ[   1:1N, which]), dims=2 ); 
    uₜ = cat( real(Χ[1N+1:2N, which]), imag(Χ[1N+1:2N, which]), dims=2 ); 
    w  = cat( real(Χ[2N+1:3N, which]), imag(Χ[2N+1:3N, which]), dims=2 ); 
    p  = cat( real(Χ[3N+1:4N, which]), imag(Χ[3N+1:4N, which]), dims=2 ); 
    b  = cat( real(Χ[4N+1:5N, which]), imag(Χ[4N+1:5N, which]), dims=2 ); 

    #return λₛ[1:5] #[1:3], Χ #, uᵣ, uₜ, w, b
    return r, z, λₛ, uᵣ, uₜ, w, b
end

m = 1.0
λref = complex(0.25, -0.25)
ra = 0.25
rb = ra 
@time r, z, λₛ, uᵣ, uₜ, w, b = solve_AAI2d(m, λref, ra, rb)

save("eigen_fun_m" * string(trunc(Int32, m+1e-2)) *".jld", 
    "r", r, "z", z, "ur", uᵣ, "ut", uₜ, "w", w, "b", b)

#U = diag(U)
#B = diag(B)

#### plotting the eigenfunction
function Interp2D_eigenFun(yn, zn, An, y0, z0)
    itp = BicubicInterpolator(yn, zn, transpose(An))
    A₀ = zeros(Float64, length(y0), length(z0))
    A₀ = [itp(yᵢ, zᵢ) for yᵢ ∈ y0, zᵢ ∈ z0]
    return A₀
end

uᵣ = reshape( uᵣ[:,1], (length(z), length(r)) )
uₜ = reshape( uₜ[:,1], (length(z), length(r)) )
w  = reshape( w[:,1],  (length(z), length(r)) )
b  = reshape( b[:,1],  (length(z), length(r)) )

#U  = reshape( U,  (length(z), length(r)) )
#B  = reshape( B,  (length(z), length(r)) )

r_interp = collect(LinRange(minimum(r), maximum(r), 1000))
z_interp = collect(LinRange(minimum(z), maximum(z), 100) )

#U_interp = Interp2D_eigenFun(r, z, U, r_interp, z_interp)
#B_interp = Interp2D_eigenFun(r, z, B, r_interp, z_interp)

fig = Figure(fontsize=30, resolution = (1800, 580), )

ax1 = Axis(fig[1, 1], xlabel=L"$r/R$", xlabelsize=30, ylabel=L"$z/H$", ylabelsize=30)

interp_  = Interp2D_eigenFun(r, z, uᵣ, r_interp, z_interp)
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
xlims!(0., maximum(r))
ylims!(minimum(z), maximum(z))

ax2 = Axis(fig[1, 3], xlabel=L"$r/R$", xlabelsize=30, ylabel=L"$z/H$", ylabelsize=30)

interp_ = Interp2D_eigenFun(r, z, uₜ, r_interp, z_interp)
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
xlims!(0., maximum(r))
ylims!(minimum(z), maximum(z))

ax1.title = L"Re$(\hat{u}_r)$"
ax2.title = L"Re$(\hat{u}_\theta)$"

fig
save("AAI2d.png", fig, px_per_unit=4)
