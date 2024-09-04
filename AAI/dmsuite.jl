using ToeplitzMatrices
using LinearAlgebra
using Printf
using FFTW

#@inline diagonal(A::AbstractMatrix, k::Integer=0) = view(A, diagind(A, k))

function toeplitz(x::AbstractVector{T}) where T
    n = length(x)
    A = zeros(T, n, n)
    for i = 1:n
        for j = 1:n-i+1
            A[i,i+j-1] = x[j]
        end
        for j = n-i+2:n
            A[i, j-(n-i+1)] = x[j]
        end
    end
    return A
end


function cheb(N)
    @assert N > 0
    x = @. cos(π*(0:N)/N)' / 2 + 0.5; 
    c = @. [2; ones(N-1, 1); 2] * (-1)^(0:N)';
    X = repeat(x, N+1, 1)';
    dX = @. X - X';                  
    #D  = (c .* (1.0 ./ c)') ./ (dX .+ Eye{Float64}(N+1));      # off-diagonal entries
    D  = (c .* (1.0 ./ c)') ./ (dX .+ Matrix(1.0I, N+1, N+1));      # off-diagonal entries
    L  = similar(D); fill!(L, 0.0); 
    L[diagind(L)] = dropdims(sum(D, dims=2), dims=2);
    D  = @. D - L;                                              # diagonal entries
    reverse!(x)
    reverse!(D)
    # x = transpose(x);
    return x[1,:], D
end

function chebdif(ncheb, mder)
    """
    Calculate differentiation matrices using Chebyshev collocation.
    Returns the differentiation matrices D1, D2, .. Dmder corresponding to the
    mder-th derivative of the function f, at the ncheb Chebyshev nodes in the
    interval [-1,1].
    Parameters
    ----------
    ncheb : int, polynomial order. ncheb + 1 collocation points
    mder   : int
          maximum order of the derivative, 0 < mder <= ncheb - 1
    Returns
    -------
    x  : ndarray
         (ncheb + 1) x 1 array of Chebyshev points
    DM : ndarray
         mder x ncheb x ncheb  array of differentiation matrices
    Notes
    -----
    This function returns  mder differentiation matrices corresponding to the
    1st, 2nd, ... mder-th derivates on a Chebyshev grid of ncheb points. The
    matrices are constructed by differentiating ncheb-th order Chebyshev
    interpolants.
    The mder-th derivative of the grid function f is obtained by the matrix-
    vector multiplication
    .. math::
    f^{(m)}_i = D^{(m)}_{ij}f_j
    The code implements two strategies for enhanced accuracy suggested by
    W. Don and S. Solomonoff :
    (a) the use of trigonometric  identities to avoid the computation of
    differences x(k)-x(j)
    (b) the use of the "flipping trick"  which is necessary since sin t can
    be computed to high relative precision when t is small whereas sin (pi-t)
    cannot.
    It may, in fact, be slightly better not to implement the strategies
    (a) and (b). Please consult [3] for details.
    This function is based on code by Nikola Mirkov
    http://code.google.com/p/another-chebpy
    References
    ----------
    ..[1] B. Fornberg, Generation of Finite Difference Formulas on Arbitrarily
    Spaced Grids, Mathematics of Computation 51, no. 184 (1988): 699-706.
    ..[2] J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix
    Suite, ACM Transactions on Mathematical Software, 26, (2000) : 465-519
    ..[3] R. Baltensperger and M. R. Trummer, Spectral Differencing With A
    Twist, SIAM Journal on Scientific Computing 24, (2002) : 1465-1487
    Examples
    --------
    The derivatives of functions is obtained by multiplying the vector of
    function values by the differentiation matrix.
    """

    if mder ≥ ncheb + 1
        throw("number of nodes must be greater than mder")
    end

    if mder ≤ 0
        throw("derivative order must be at least 1")
    end
    
    DM = zeros(mder, ncheb+1, ncheb+1);
    ℒ  = Matrix(I, ncheb, ncheb);

    # indices used for flipping trick
    nn1 = Int32(floor(ncheb/2));
    nn2 = Int32(ceil(ncheb/2));
    k = 0:1:ncheb-1;

    # compute theta vector
    θ = k * π / (ncheb-1);

    # Compute the Chebyshev points
    x = sin.(π * (ncheb-1 .- 2 * range(ncheb-1, 0, length=ncheb)) / (2 * (ncheb-1)));
    reverse!(x);

    # Assemble the differentiation matrices
    T = repeat(θ/2, 1, ncheb);
    # trigonometric identity
    Dₓ = 2 * sin.(T' + T) .* sin.(T' - T);
    
    # flipping trick
    Dₓ[nn1+1:end, 1:end] = -reverse(reverse(Dₓ[1:nn2, 1:end], dims=2), dims=1);
        
    # diagonals of Dₓ
    Dₓ[ℒ] .= ones(ncheb);

    # matrix with entries c(k)/c(j)
    C = toeplitz( (-1.0) .^ k);
    C[1, :] = C[1, :] * 2.0; C[end, :] = C[end, :] * 2.0;
    C[:, 1] = C[:, 1] * 0.5; C[:, end] = C[:, end] * 0.5;

    # Z contains entries 1/(x(k)-x(j))
    Z = 1.0 ./ Dₓ;
    # with zeros on the diagonal.
    Z[ℒ] .= zeros(ncheb);

    # initialize differentiation matrices.
    D = Matrix(1.0I, ncheb, ncheb);

    Dᵐ = zeros(Float64, mder, ncheb, ncheb);

    for ell = 1:mder
        # off-diagonals
        D = ell * Z .* (C .* repeat(diag(D), 1, ncheb) - D);
        # negative sum trick
        D[ℒ] .= -sum(D, dims=2);
        # store current D in Dᵐ
        Dᵐ[ell, 1:end, 1:end] = D;
    end

    # return only one differntiation matrix
    D = Dᵐ[mder, 1:end, 1:end];
    # mirror x ∈ [1,-1] to x ∈ [-1,1] (for convenience)
    reverse!(x);
    reverse!(D);

    return x, D

end


function FourierDiff(nfou, mder)
    """
    Fourier spectral differentiation.
    Spectral differentiation matrix on a grid with nfou equispaced points in [0, 2π)
    INPUT
    -----
    nfou: Size of differentiation matrix.
    mder: Derivative required (non-negative integer)
    OUTPUT
    -------
    xxt: Equispaced points 0, 2pi/nfou, 4pi/nfou, ... , (nfou-1)2pi/nfou
    ddm: mder'th order differentiation matrix
    Explicit formulas are used to compute the matrices for m=1 and 2.
    A discrete Fouier approach is employed for m>2. The program
    computes the first column and first row and then uses the
    toeplitz command to create the matrix.
    For mder=1 and 2 the code implements a "flipping trick" to
    improve accuracy suggested by W. Don and A. Solomonoff in
    SIAM J. Sci. Comp. Vol. 6, pp. 1253--1268 (1994).
    The flipping trick is necesary since sin t can be computed to high
    relative precision when t is small whereas sin (pi-t) cannot.
    S.C. Reddy, J.A.C. Weideman 1998.  Corrected for MATLAB R13
    by JACW, April 2003.
    """

    # grid points
    range_ = 0:1:nfou-1
    x₀ = collect(2π * range_ / nfou)

    # grid spacing
    dx₀ = 2π/nfou

    # indices used for flipping trick
    nn1 = Int32(floor((nfou-2)/2));
    nn2 = Int32(ceil(((nfou-2))/2));

    if mder == 0
        # compute first column of zeroth derivative matrix, which is identity
        col1 = zeros(nfou)
        col1[1] = 1
        row1 = copy(col1)

    elseif mder == 1
        # compute first column of 1st derivative matrix
        col1 = 0.5 .* [(-1)^k for k ∈ 1:nfou-1]
        if nfou % 2 == 0
            topc = 1.0 ./ tan.( (1:nn2+1) * 0.5dx₀ )
            col1 = col1 .* vcat( topc, -reverse(topc[1:nn1]) )
            col1 = vcat( 0, col1 )
        else
            topc = 1.0 ./ sin.( (1:nn2) * 0.5dx₀ )
            col1 = vcat( 0, col1 .* vcat( topc, reverse(topc[1:nn1+1]) ) )
        end
        # first row
        row1 = -1.0 .* col1

    elseif mder == 2
        # compute first column of 1st derivative matrix
        col1 = -0.5 .* [(-1)^k for k ∈ 1:nfou-1]   
        if nfou % 2 == 0 # corresponds to odd number of grid points
            topc = 1.0 ./ sin.( (1:nn2+1) * 0.5dx₀ ).^2
            col1 = col1 .* vcat( topc, reverse(topc[1:nn1]) )
            col1 = vcat( -π^2 / 3.0 / dx₀^2 - 1.0/6.0, col1 )
        else  # corresponds to even number of grid points
            topc = @. ( 1.0/ tan((1:nn2) * 0.5dx₀)/ sin((1:nn2) * 0.5dx₀) )
            col1 = col1 .* vcat( topc, -reverse(topc[1:nn1+1]) )
            col1 = vcat( -π^2 / 3 / dx₀^2 + 1/12, col1 ) 
        end
        # first row
        row1 = 1.0 .* col1
    else
        # employ FFT to compute 1st column of matrix for mder > 2
        nfo1  = floor((nfou-2)/2);
        nfo2  = @. -nfou/2.0 * (rem(mder,2)==0) * ones(rem(nfou,2)==0)
        mwave = 1.0 * im .* vcat(0:1:nfo1, nfo2, nfo1:-1:1)
        col1  = real( ifft( 
                    mwave.^mder .* fft( vcat(1, zeros(nfou-1)) ) 
                    ) 
                )
        if mder % 2 == 0
            row1 = 1.0 .* col1
        else
            col1 = vcat(0, col1[2:nfou+1])
            row1 = -1.0 .* col1
        end
    end

    𝒟 = Toeplitz(col1, row1)
    return x₀, 𝒟
end
