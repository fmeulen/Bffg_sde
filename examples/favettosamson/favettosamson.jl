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

ℙ0 = FV(117.0, 5.0, 6.0, 3.0, 0.3, 1.4, t-> t/(1+0.5*t)^2)

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

savefig(joinpath(outdir,"forward_and_observations.png"))

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



function parupdate!(B, ℙ::FV, x0, Z, ll, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=true)
  accpar_ = false
  θ =  getpar(move)(ℙ)  #move.par(ℙ)
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
#      @. θ = θᵒ
    B = Bᵒ
    ℙ = ℙᵒ
    accpar_ = true
    !verbose && print("✓")  
  end
  ll, B, ℙ, accpar_
end

parupdate!(B, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=true)  = (x0, ℙ, Z, ll) -> parupdate!(B, ℙ, x0, Z, ll, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)


     
