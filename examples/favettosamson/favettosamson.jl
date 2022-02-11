######  spedific model definitions

# target process
struct FV{T,Tδ} <: ContinuousTimeProcess{ℝ{2}}
    α::T
    λ::T
    β::T
    μ::T
    σ1::T
    σ2::T
    δ::Tδ
end

# auxiliary process 
struct FVAux{Tpar,Tδ} <: ContinuousTimeProcess{ℝ{2}}
    α::Tpar
    λ::Tpar
    β::Tpar
    μ::Tpar
    σ1::Tpar
    σ2::Tpar
    δ::Tδ
end

FVAux(P::FV) = FVAux(P.α, P.λ, P.β, P.μ, P.σ1, P.σ2, P.δ)

Bridge.b(t, x, P::FV) =  SA[P.α * P.δ(t) - (P.λ +P.β)*x[1] + P.μ*x[2],  P.λ*x[1] - P.μ*x[2] ]
Bridge.σ(t, x, P::FV) = SDiagonal(P.σ1, P.σ2)        

Bridge.constdiff(::FV) = true
Bridge.constdiff(::FVAux) = true
dim(::FV) = 2
dim(::FVAux) = 2

Bridge.B(t, P::FVAux) = @SMatrix [-(P.λ +P.β) P.μ;  P.λ  -P.μ ]
Bridge.β(t, P::FVAux) = SA[P.α * P.δ(t), 0.0]
Bridge.σ(t, P::FVAux) = SDiagonal(P.σ1, P.σ2)        
Bridge.a(t, P::FVAux) = SDiagonal(P.σ1^2, P.σ2^2)    
Bridge.b(t, x, P::FVAux) = Bridge.B(t,P) * x + Bridge.β(t,P)

wienertype(::FV) = Wiener{ℝ{2}}()


####### simulate some data 

ℙ0 = FV(217.0, 5.0, 6.0, 3.0, 0.3, 1.4, t-> t/(1+0.5*t)^2)

T = 10.0
x0 = SA[0.0, 0.0]
W = sample(0.0:0.0001:T, wienertype(ℙ0))                        #  sample(tt, Wiener{ℝ{1}}())
Xf = solve(Euler(), x0, W, ℙ0)
p = plot(Xf.tt, first.(Xf.yy))
plot!(p, Xf.tt, last.(Xf.yy))

L = @SMatrix [1.0 1.0]
skipobs = 10_000
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
X=Xf
p = plot(X.tt, getindex.(X.yy,1), label="1")
plot!(p, X.tt, getindex.(X.yy,2), label="2")
plot!(p, obstimes, map(x->x[1], obsvals), seriestype=:scatter, markersize=2.5, label="obs")
    

####### BFFG

# process observations
Σdiagel = 10e-6
Σ = SMatrix{1,1}(Σdiagel*I)
obs = [Observation(obstimes[1],  x0,  SMatrix{2,2}(1.0I), SMatrix{2,2}(Σdiagel*I))]
for i in 2:length(obstimes)
  push!(obs, Observation(obstimes[i], obsvals[i], L, Σ));
end

auxprocesses(ℙ::FV, obs) = [FVAux(ℙ) for i in 2:length(obs)]

ℙ̃s = auxprocesses(ℙ0, obs)

# set the ODE solver
S = DE(Vern7())  

# set timegrids and auxiliary processes
timegrids = set_timegrids(obs, 0.005)
ℙ̃s = auxprocesses(ℙ0, obs)

# backward filter
B = BackwardFilter(S, ℙ̃s, obs, timegrids)
# compute innovations
Z = Innovations(timegrids, ℙ0);
# forward guide
XX, ll = forwardguide(B, ℙ0)(x0, Z);

function plot_(::FV, tt, XX, comp::Int)
    p = plot(tt[1], ec(XX[1],comp), label="")
    for k in 2:length(XX)
      plot!(p, tt[k], ec(XX[k],comp), label="")
    end
    p
end



p1 = plot_(ℙ, timegrids, XX, 1)
plot!(p1, Xf.tt, first.(Xf.yy),label="forward", color=:black)
plot!(p1, obstimes, 0.5*map(x->x[1], obsvals), seriestype=:scatter, markersize=2.5, label="obs")
p2 = plot_(ℙ, timegrids, XX, 2)
plot!(p2, Xf.tt, last.(Xf.yy),label="forward", color=:black)
plot!(p2, obstimes, 0.5*map(x->x[1], obsvals), seriestype=:scatter, markersize=2.5, label="obs")
plot(p1, p2, l= @layout [a b])


Zbuffer = deepcopy(Z)
Zᵒ = deepcopy(Z)
ρs = zeros(length(timegrids))




moveλ = ParMove([:λ], parameterkernel((short=[.2], long=[1.0]); s=0.0), Uniform(0.0, 1000.0), true)
moveλ = ParMove([:α], parameterkernel((short=[.2], long=[1.0]); s=0.2), Uniform(0.0, 1000.0), true)
λinit = 55.0
θ = [copy(λinit)] # initial value for parameter
ℙ = setproperties(ℙ0, λ=λinit)
ℙ̃s = auxprocesses(ℙ, obs)


B = BackwardFilter(S, ℙ̃s, obs, timegrids)
Z = Innovations(timegrids, ℙ);
XX, ll = forwardguide(B, ℙ)(x0, Z);
XXsave = [XX]

iterations = 1_000  
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

samples = [State(x0, copy(Z), copy(θ), copy(ll))]
for i in 1:iterations
  (i % 500 == 0) && println(i)
  
#  ll, B, ℙ, accpar_ = parupdate!(B, ℙ, XX, moveλ, obs, S, timegrids; verbose=verbose)(x0, θ, Z, ll);
  ll, B, ℙ, accpar_ = parupdate!(B, ℙ, x0, θ, Z, ll, XX, moveλ, obs, S,  timegrids)

  ll, accinnove_ = pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs)(x0, Z, ll); 
  push!(samples, State(x0, copy(Z), copy(θ), copy(ll)))   # collection of samples from exploring chain
  (i in subsamples) && push!(XXsave, deepcopy(XX))
end

θs = getfield.(samples, :θ)
plot(first.(θs))    
hline!([ℙ0.α])
hline!([ℙ0.λ])


path = vcat(XXsave[end]...)
pp = plot(first.(path))
path1 = vcat(XXsave[1]...)
plot!(pp, first.(path1))




function parupdate!(B, ℙ::FV, x0, θ, Z, ll, XX, move, obs, S, timegrids; verbose=true)
    accpar_ = false
    θᵒ = propose(move)(θ)   
    ℙᵒ = setpar(move)(θᵒ, ℙ)    
    if move.recomputeguidingterm        
        Bᵒ =BackwardFilter(S, auxprocesses(ℙᵒ, obs), obs, timegrids)
    else 
        Bᵒ = B
    end
    XXᵒ, llᵒ = forwardguide(Bᵒ, ℙᵒ)(x0, Z)
    !verbose && printinfo(ll, llᵒ, "par") 

    if log(rand()) < llᵒ-ll + logpriordiff(move)(θ, θᵒ)
      @. XX = XXᵒ
      ll = llᵒ
      @. θ = θᵒ
      B = Bᵒ
      ℙ = ℙᵒ
      accpar_ = true
      !verbose && print("✓")  
    end
    ll, B, ℙ, accpar_
end

parupdate!(B, ℙ::FV, XX, move, obs, S, timegrids; verbose=true)  = (x0, θ, Z, ll) -> parupdate!(B, ℙ, x0, θ, Z, ll, XX, move, obs, S,  timegrids; verbose=verbose)

