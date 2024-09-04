using LinearAlgebra
using Printf
using BSplineKit #: interpolate
using Tullio
using Statistics

function myfindall(condition, x)
    results = Int[]
    for i in 1:length(x)
        if condition(x[i])
            push!(results, i)
        end
    end
    return results
end

function findnearest(a, x)
    length(a) > 0 || return 0:-1
    r = searchsorted(a,x)
    length(r) > 0 && return r
    last(r) < 1 && return searchsorted(a,a[first(r)])
    first(r) > length(a) && return searchsorted(a,a[last(r)])
    x-a[last(r)] < a[first(r)]-x && return searchsorted(a,a[last(r)])
    x-a[last(r)] > a[first(r)]-x && return searchsorted(a,a[first(r)])
    return first(searchsorted(a,a[last(r)])):last(searchsorted(a,a[first(r)]))
end

# print the eigenvalues
function print_evals(λs, n)
    @printf "%i largest eigenvalues: \n" n
    for p in n:-1:1
        if imag(λs[p]) >= 0
            @printf "%i: %1.4e+%1.4eim\n" p real(λs[p]) imag(λs[p])
        end
        if imag(λs[p]) < 0
            @printf "%i: %1.4e%1.4eim\n" p real(λs[p]) imag(λs[p])
        end
    end
end

# sort the eigenvalues
function sort_evals(λs, χ, which, sorting="lm")
    @assert which ∈ ["M", "I", "R"]

    if sorting == "lm"
        if which == "I"
            idx = sortperm(λs, by=imag, rev=true) 
        end
        if which == "R"
            idx = sortperm(λs, by=real, rev=true) 
        end
        if which == "M"
            idx = sortperm(λs, by=abs, rev=true) 
        end
    else
        if which == "I"
            idx = sortperm(λs, by=imag, rev=false) 
        end
        if which == "R"
            idx = sortperm(λs, by=real, rev=false) 
        end
        if which == "M"
            idx = sortperm(λs, by=abs, rev=false) 
        end
    end

    return λs[idx], χ[:,idx]
end


function remove_evals(λs, χ, lower, higher, which)
    @assert which ∈ ["M", "I", "R"]
    if which == "I" # imaginary part
        arg = findall( (lower .≤ imag(λs)) .& (imag(λs) .≤ higher) )
    end
    if which == "R" # real part
        arg = findall( (lower .≤ real(λs)) .& (real(λs) .≤ higher) )
    end
    if which == "M" # absolute magnitude 
        arg = findall( abs.(λs) .≤ higher )
    end
    
    χ  = χ[:,arg]
    λs = λs[arg]
    return λs, χ
end

function remove_spurious(λₛ, X)
    #p = findall(x->x>=abs(item), abs.(real(λₛ)))  
    deleteat!(λₛ, 1)
    X₁ = X[:, setdiff(1:end, 1)]
    return λₛ, X₁
end

# function ∇f(f, x)
#     @assert ndims(f) == ndims(x)
#     # @tullio dx[i] := x[i+1] - x[i]
#     # @assert std(dx) ≤ 1.0e-8
#     N = length(x); #Assume >= 3
#     ∂f_∂x = similar(f);
#     Δx    = x[2]-x[1]; #assuming evenly spaced points
#     ∂f_∂x[1] = (-3.0/2.0*f[1] + 2.0*f[2] - 1.0/2.0*f[3]);
#     ∂f_∂x[2] = (-3.0/2.0*f[2] + 2.0*f[3] - 1.0/2.0*f[4]);
#     for k ∈ 3:N-2
#         ∂f_∂x[k] = (1.0/12.0*f[k-2] - 2.0/3.0*f[k-1] + 2.0/3.0*f[k+1] - 1.0/12.0*f[k+2]);
#     end
#     ∂f_∂x[N-1] = (3.0/2.0*f[N-1] - 2.0*f[N-2] + 1.0/2.0*f[N-3]);
#     ∂f_∂x[N]   = (3.0/2.0*f[N]   - 2.0*f[N-1] + 1.0/2.0*f[N-2]);
#     return ∂f_∂x ./ Δx
# end

function ∇f(f, x)
    @assert ndims(f) == ndims(x)
    @tullio dx[i] := x[i+1] - x[i]
    @assert std(dx) ≤ 1.0e-8
    @assert length(f) == length(x)
    N = length(x); #Assume >= 3
    ∂f_∂x = similar(f);
    ∂f_∂x .= 0.0;
    Δx    = x[2]-x[1]; #assuming evenly spaced points

    c₄₊ = (-25.0/12.0, 4.0, -3.0, 4.0/3.0, -1.0/4.0);
    c₄₋ = @. -1.0 * c₄₊;
    c₈  = (1.0/280.0, -4.0/105.0, 1.0/5.0, 
        -4.0/5.0, 0.0, 4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0);
    for k ∈ 1:4
        ∂f_∂x[k] = (c₄₊[1]*f[k] + c₄₊[2]*f[k+1] + 
        c₄₊[3]*f[k+2] + c₄₊[4]*f[k+3] + c₄₊[5]*f[k+4]);
    end
    for k ∈ 5:N-4
        ∂f_∂x[k]  = c₈[1]*f[k-4] + c₈[2]*f[k-3] + c₈[3]*f[k-2] + c₈[4]*f[k-1];
        ∂f_∂x[k] += c₈[5]*f[k];
        ∂f_∂x[k] += c₈[6]*f[k+1] + c₈[7]*f[k+2] + c₈[8]*f[k+3] + c₈[9]*f[k+4];
    end
    for k ∈ N-3:N
        ∂f_∂x[k] = (c₄₋[1]*f[k] + c₄₋[2]*f[k-1] 
                + c₄₋[3]*f[k-2] + c₄₋[4]*f[k-3] + c₄₋[5]*f[k-4]);
    end
    return ∂f_∂x ./Δx
end

function gradient(f, x; dims::Int=1)
    n   = size(f)
    sol = similar(f)
    if ndims(f) == 1
        sol = ∇f(f, x)
    end
    if ndims(f) == 2
        @assert ndims(f) ≥ dims 
        if dims==1
            for it ∈ 1:n[dims+1]
                sol[:,it] = ∇f(f[:,it], x)
            end
        else
            for it ∈ 1:n[dims-1]
                sol[it,:] = ∇f(f[it,:], x)
            end
        end
    end
    if ndims(f) == 3
        @assert ndims(f) ≥ dims 
        if dims==1
            for it ∈ 1:n[dims+1], jt ∈ 1:n[dims+2]
                sol[:,it,jt] = ∇f(f[:,it,jt], x)
            end
        elseif dims==2
            for it ∈ 1:n[dims-1], jt ∈ 1:n[dims+1]
                sol[it,:,jt] = ∇f(f[it,:,jt], x)
            end
        else
            for it ∈ 1:n[dims-2], jt ∈ 1:n[dims-1]
                sol[it,jt,:] = ∇f(f[it,jt,:], x)
            end    
        end
    end
    return sol
end

# trapezoidal integration for a uniform or nonuniform grids 
# ∫ₐᵇ f(x) dx = ∑ₖ₌₁ⁿ 0.5(f(xₖ) + f(xₖ₊₁)) Δxₖ; Δxₖ = xₖ₊₁ - xₖ; n = N-1
function Trapz_integ(f, x, interp::Bool=true)
    @assert ndims(f) == ndims(x)
    if interp
        itp = BSplineKit.interpolate(x, f, BSplineOrder(4))
        x1  = collect(range(minimum(x), stop=maximum(x), length=3length(x)))
        f1  = zeros(eltype(f), length(x1))
        for it in 1:length(x1)
            f1[it] = itp(x1[it])
        end
        f = f1; x = x1;
    end
    N = length(x); 
    sum_ = 0.0
    for it in 1:N-1
        Δx = x[it+1] - x[it]
        sum_ += (f[it] + f[it+1])*Δx/2.0
    end
    return sum_
end
