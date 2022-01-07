say(what) = run(`osascript -e "say \"$(what)\""`, wait=false)

lastval(X::SamplePath) = X.yy[end]

lastval(ℐ::PathInnovation) = lastval(ℐ.X)

function mergepaths(ℐs)
    tt = map(x->x.X.tt, ℐs)
    yy = map(x->x.X.yy, ℐs)
    SamplePath(vcat(tt...),vcat(yy...))
end


"""
    kernelrk4(f, t, y, dt, ℙ)

    solver for Runge-Kutta 4 scheme
"""
function kernelrk4(f, t, y, dt, ℙ)
    k1 = f(t, y, ℙ)
    k2 = f(t + 0.5*dt, y + 0.5*k1*dt, ℙ)
    k3 = f(t + 0.5*dt, y + 0.5*k2*dt, ℙ)
    k4 = f(t + dt, y + k3*dt, ℙ)
    y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
end


HFC(obs::Observation) = (obs.H, obs.F, obs.C)

vectorise(P,ν, C) = vcat(SVector(P), ν, SVector(C))

"""
    init_HFC(v, L, d; ϵ=0.01)

    d = dimension of the diffusion
    First computes xT = L^(-1) * vT (Moore-Penrose inverse), a reasonable guess for the full state based on the partial observation vT
    Then convert artifical observation v ~ N(xT, ϵ^(-1) * I)
    to triplet  (H, F, C)
"""
function init_HFC(v, L, d::Int64; ϵ=0.01)
    P = ϵ^(-1)*SMatrix{d,d}(1.0I)
    xT = L\v
    z = zero(xT)
    C = logpdf(Bridge.Gaussian(z, P), z) 
    convert_PνC_to_HFC(P, xT ,C)
end


"""
    observation_HFC(v, L, Σ)

    Convert observation v ~ N(Lx, Σ)
    to triplet  (H, F, C)
"""
function observation_HFC(v, L, Σ)
    A = L' * inv(Σ)
    A*L, A*v, logpdf(Bridge.Gaussian(zero(v), Σ), v)
end



"""
    static_accessor_HFc(u::SVector, ::Val{T}) where T
Access data stored in the container `u` so that it matches the shapes of H,F,c
and points to the correct points in `u`. `T` is the dimension of the stochastic
process.

implemented by M. Mider in GuidedProposals.jl
"""
function static_accessor_HFc(u::K, ::Val{T}) where {K<:Union{SVector,MVector},T}
    Hidx = SVector{T*T,Int64}(1:T*T)
    Fidx = SVector{T,Int64}((T*T+1):(T*T+T))
    reshape(u[Hidx], Size(T,T)), u[Fidx], u[T*T+T+1]
end


τ(S,T) = (x) ->  S + (x-S) * (2-(x-S)/(T-S))

timegrid(S, T, M) = τ(S,T).(collect(range(S, T, length=M)))

set_timegrids(obs, M) =  [timegrid(obs[i-1].t, obs[i].t, M) for i ∈ 2:length(obs)]