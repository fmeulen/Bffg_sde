wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")
include("favettosamson.jl")


allparnames =[:β, :μ, :α, :σ1]
move = ParMove([:β], parameterkernel((short=[.2], long=[1.0]); s=0.0), Uniform(0.0, 1000.0), true)
move = ParMove([:μ], parameterkernel((short=[.2], long=[1.0]); s=0.0), Uniform(0.0, 1000.0), true)
move = ParMove([:α], parameterkernel((short=[.2], long=[2.0]); s=.5), Uniform(0.0, 1000.0), true)
U = Uniform(0.0, 1000.0)
move = ParMove([:μ, :α, :σ1], parameterkernel((short=[.2, .2, .2], long=[2.0, 2.0, 1.0]); s=.5), product_distribution([U,U, U]), true)

#moveλ = ParMove([:β], parameterkernel((short=[.2], long=[1.0]); s=0.2), Uniform(0.0, 1000.0), true)
θinit = [15.0, 15.0, 5.0]
θ = [copy(θinit)] # initial value for parameter
ℙ = setproperties(ℙ0, β=θinit)
ℙ = setproperties(ℙ0, μ=θinit)
ℙ = setproperties(ℙ0, α=θinit)

ℙ = setproperties(ℙ0, μ=θinit[1], α=θinit[2], σ1=θinit[3])

ℙ̃s = auxprocesses(ℙ, obs)


B = BackwardFilter(S, ℙ̃s, obs, timegrids)
Z = Innovations(timegrids, ℙ);
XX, ll = forwardguide(B, ℙ)(x0, Z);
XXsave = [XX]

iterations = 23_000  
skip_it = 200
subsamples = 0:skip_it:iterations # for saving paths

samples = [State(x0, copy(Z), getpar(allparnames, ℙ), copy(ll))]
for i in 1:iterations
  (i % 500 == 0) && println(i)
  
  ll, B, ℙ, accpar_ = parupdate!(B, XX, move, obs, obsvals, S, AuxType, timegrids; verbose=verbose)(x0, ℙ, Z, ll);
  ll, accinnove_ = pcnupdate!(B, ℙ, XX, Zbuffer, Zᵒ, ρs)(x0, Z, ll); 
  push!(samples, State(x0, copy(Z), getpar(allparnames, ℙ), copy(ll)))  
  (i in subsamples) && push!(XXsave, copy(XX))
end

θs = getfield.(samples, :θ)
#plot(first.(θs),label="")  

pμ = plot(getindex.(θs, 2), label="μ")
hline!([ℙ0.μ],label="")

pα = plot(getindex.(θs, 3),label="α")  
hline!(pα, [ℙ0.α],label="")

pσ1 = plot(getindex.(θs, 4),label="σ1")  
hline!(pσ1, [ℙ0.σ1],label="")


plot(pα, pμ, pσ1, layout = @layout [a; b; c])
savefig(joinpath(outdir,"traces.png"))

scatter(getindex.(θs, 2), getindex.(θs, 3))

#hline!([ℙ0.β])

hline!([ℙ0.μ],label="")
hline!([ℙ0.α],label="")
hline!([ℙ0.λ],label="")

savefig(joinpath(outdir,"trace_alpha.png"))

path = vcat(XXsave[end]...)
pp = plot(first.(path))
path1 = vcat(XXsave[1]...)
plot!(pp, first.(path1))



