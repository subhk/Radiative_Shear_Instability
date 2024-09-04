using LazyGrids
using BlockArrays
using Printf
using StaticArrays
#using Interpolations
using SparseArrays
using SparseMatrixDicts
using SpecialFunctions
using FillArrays
using Parameters
using Test
using MAT
using BenchmarkTools
using BasicInterpolators: BicubicInterpolator

using Serialization
#using Pardiso
using Arpack
using LinearMaps
using ArnoldiMethod

function Eigs(𝓛, ℳ; σ::ComplexF64, maxiter)
    λₛ, Χ = Arpack.eigs(𝓛, ℳ,
                        nev     = 1, 
                        tol     = 1e-10, 
                        maxiter = maxiter, 
                        which   = :LR,
                        sigma   = σ)
    return λₛ, Χ
end

function EigSolver_shift_invert1(𝓛, ℳ; σ₀::ComplexF64)
    maxiter::Int = 30
    try 
        σ = complex(1.10σ₀.re, σ₀.im) 
        @printf "sigma: %f \n" σ.re
        λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
        return λₛ, Χ
    catch error
        try 
            σ = complex(1.05σ₀.re, σ₀.im)
            @printf "(first didn't work) sigma: %f \n" real(σ) 
            λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
            return λₛ, Χ
        catch error
            try 
                σ = complex(1.02σ₀.re, σ₀.im)
                @printf "(second didn't work) sigma: %f \n" real(σ) 
                λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                return λₛ, Χ
            catch error
                try
                    σ = complex(0.98σ₀.re, σ₀.im)
                    @printf "(third didn't work) sigma: %f \n" real(σ) 
                    λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                    return λₛ, Χ
                catch error
                    try
                        σ = complex(0.94σ₀.re, σ₀.im)
                        @printf "(third didn't work) sigma: %f \n" real(σ) 
                        λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                        return λₛ, Χ
                    catch error
                        σ = complex(0.92σ₀.re, σ₀.im)
                        @printf "(third didn't work) sigma: %f \n" real(σ) 
                        λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                        return λₛ, Χ                        
                    end
                end
            end
        end    
    end
end

function EigSolver_shift_invert(𝓛, ℳ; σ₀::ComplexF64)
    maxiter = 30
    try 
        σ = complex(2.0σ₀.re, σ₀.im) 
        @printf "sigma: (%f %f)\n" σ.re σ.im
        λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
        return λₛ, Χ
    catch error
        try
            σ.re = complex(1.8σ₀.re, σ₀.im) #1.4σ₀.re
            @printf "(first didn't work) sigma: (%f %f)\n" σ.re σ.im
            λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
            return λₛ, Χ
        catch error
            try
                σ = complex(1.6σ₀.re, σ₀.im) #1.3σ₀.re
                @printf "(second didn't work) sigma: (%f %f) \n" σ.re σ.im
                λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                return λₛ, Χ
            catch error
                try
                    σ = complex(1.4σ₀.re, σ₀.im) #1.2σ₀.re
                    @printf "(third didn't work) sigma: (%f %f) \n" σ.re σ.im
                    λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                    return λₛ, Χ
                catch error
                    try
                        σ = complex(1.2σ₀.re, σ₀.im) #1.15σ₀
                        @printf "(fourth didn't work) sigma: (%f %f) \n" σ.re σ.im
                        λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                        return λₛ, Χ 
                    catch error
                        try
                            σ = complex(1.0σ₀.re, σ₀.im) #1.1σ₀
                            @printf "(fifth didn't work) sigma: (%f %f) \n" σ.re σ.im
                            λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                            return λₛ, Χ
                        catch error
                            try
                                σ = complex(0.8σ₀.re, σ₀.im) #1.0σ₀
                                @printf "(sixth didn't work) sigma: (%f %f) \n" σ.re σ.im
                                λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                return λₛ, Χ
                            catch error
                                try
                                    σ = complex(0.6σ₀.re, σ₀.im) #0.95σ₀
                                    @printf "(seventh didn't work) sigma: (%f %f) \n" σ.re σ.im
                                    λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                    return λₛ, Χ
                                catch error
                                    try
                                        σ = complex(0.4σ₀.re, σ₀.im) #0.90σ₀
                                        @printf "(eighth didn't work) sigma: (%f %f) \n" σ.re σ.im
                                        λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                        return λₛ, Χ
                                    catch error
                                        try
                                            σ = complex(0.3σ₀.re, σ₀.im) #0.85σ₀
                                            @printf "(ninth didn't work) sigma: (%f %f) \n" σ.re σ.im
                                            λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                            return λₛ, Χ
                                        catch error
                                            try
                                                σ = complex(0.2σ₀.re, σ₀.im) #0.80σ₀
                                                @printf "(tenth didn't work) sigma: (%f %f) \n" σ.re σ.im
                                                λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                                return λₛ, Χ
                                            catch error
                                                σ = complex(0.1σ₀.re, σ₀.im) #0.75σ₀
                                                @printf "(eleventh didn't work) sigma: (%f %f) \n" σ.re σ.im
                                                λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
                                                return λₛ, Χ
                                            end    
                                        end   
                                    end
                                end    
                            end
                        end                    
                    end          
                end    
            end
        end
    end
end


# function EigSolver_shift_invert(𝓛, ℳ; σ₀::ComplexF64)
#     maxiter = 150
#     err::Bool = true
#     it = 2.0
#     Ctr::Int = 0
#     while err
#         try 
#             σ = complex((it-0.1Ctr)*σ₀.re, σ₀.im) 
#             @printf "sigma: (%f %f)\n" σ.re σ.im
#             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#             err = false
#             return λₛ, Χ
#         catch error
#             σ.re = complex((it-0.1Ctr)*σ₀.re, σ₀.im) #1.4σ₀.re
#             @printf "(previous didn't work) sigma: (%f %f)\n" σ.re σ.im
#             λₛ, Χ = Eigs(𝓛, ℳ; σ=σ, maxiter=maxiter)
#             err = true
#             return λₛ, Χ
#         end
#         Ctr += 1
#     end
# end