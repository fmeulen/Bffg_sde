θ0 =[3.25, 100.0, 22.0, 50.0, 190.0, 0.8, 0.25, 5.0, 6.0, 0.56, 200.0, 6.0]  # except for μy as in Buckwar/Tamborrino/Tubikanec#
ℙ0 = JansenRitDiffusion(θ0...)
@show properties(ℙ0)
AuxType = JansenRitDiffusionAux

T =5.0
x00 = @SVector zeros(6)
W = sample((-1.0):0.0001:T, wienertype(ℙ0))                        #  sample(tt, Wiener{ℝ{1}}())
Xf_prelim = solve(Euler(), x00, W, ℙ0)
# drop initial nonstationary behaviour
Xf = SamplePath(Xf_prelim.tt[10001:end], Xf_prelim.yy[10001:end])
x0 = Xf.yy[1]
dt = Xf.tt[2]-Xf.tt[1]

skipobs = 40# I took 400  all the time
obstimes =  Xf.tt[1:skipobs:end]
obsvals = map(x -> L*x, Xf.yy[1:skipobs:end])
pF = plot_all(ℙ0,  Xf, obstimes, obsvals)


#------- process observations, assuming x0 known
obs = [Observation(obstimes[1],  x0,  SMatrix{6,6}(1.0I), SMatrix{6,6}(Σdiagel*I))]
for i in 2:length(obstimes)
  push!(obs, Observation(obstimes[i], obsvals[i], L, Σ));
end


timegrids = set_timegrids(obs, 0.00005)
B = BackwardFilter(S, ℙ0, AuxType, obs, obsvals, timegrids) ;
Z = Innovations(timegrids, ℙ0);
XX, ll = forwardguide(B, ℙ0)(x0, Z);
plot_all(ℙ0,Xf,  obstimes, obsvals, timegrids, XX)

llC =[]
Cgrid = 5.0:2.5:400.0
for C ∈ Cgrid
  ℙ = setproperties(ℙ0, C=C)
  println(ℙ.C)
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llC, copy(ll))
end
plot(Cgrid, llC, label="loglik"); vline!([ℙ0.C], label="true val of C")

llμ =[]
μgrid = 5.0:2.0:400.0
for μ ∈ μgrid
  ℙ = setproperties(ℙ0, μy=μ)
  B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llμ, copy(ll))
end
plot(μgrid, llμ, label="loglik"); vline!([ℙ0.μy], label="true val of μ")


llA =[]
Agrid = .2:.2:10.0
for A ∈ Agrid
  ℙ = setproperties(ℙ0, A=A)
  B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llA, copy(ll))
end
plot(Agrid, llA, label="loglik"); vline!([ℙ0.A], label="true val of A")
plot(Agrid[10:end], llA[10:end], label="loglik"); vline!([ℙ0.A], label="true val of A")


llσ =[]
σgrid = 1.0:.2:42.0
for σ ∈ σgrid
  ℙ = setproperties(ℙ0, σy=σ)
  B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
  _, ll = forwardguide(B, ℙ)(x0, Z);
  push!(llσ, copy(ll))
end
plot(σgrid, llσ, label="loglik")
 vline!([ℙ0.σy], label="true val of σy")

 llνmax = []
 νmaxgrid = .1:.2:8.0
 for νmax ∈ νmaxgrid
   ℙ = setproperties(ℙ0, νmax=νmax)
   B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
   _, ll = forwardguide(B, ℙ)(x0, Z);
   push!(llνmax, copy(ll))
 end
 plot(νmaxgrid, llνmax, label="loglik")
  vline!([ℙ0.νmax], label="true val of νmax")
 


  llα1 = []
  α1grid = .01:.02:0.99
  for α1 ∈ α1grid
    ℙ = setproperties(ℙ0, α1=α1)
    B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
    _, ll = forwardguide(B, ℙ)(x0, Z);
    push!(llα1, copy(ll))
  end
  plot(α1grid, llα1, label="loglik")
   vline!([ℙ0.α1], label="true val of α1")
  

   llα2 = []
   α2grid = .01:.02:0.99
   for α2 ∈ α2grid
     ℙ = setproperties(ℙ0, α2=α2)
     B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
     _, ll = forwardguide(B, ℙ)(x0, Z);
     push!(llα2, copy(ll))
   end
   plot(α2grid, llα2, label="loglik")
    vline!([ℙ0.α2], label="true val of α2")
 


Cgrid = 5.0:5.0:400.0
σgrid = 100.0:100.0: 2000.0
llCσ = zeros(length(Cgrid), length(σgrid))
for i ∈ eachindex(Cgrid)
    println(i)
    for j ∈ eachindex(σgrid)
        ℙ = setproperties(ℙ0, C=Cgrid[i], σy = σgrid[j])
        B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
        #println(ℙ.C)
        _, ll = forwardguide(B, ℙ)(x0, Z);
        llCσ[i,j] = ll
    end
end
plot(Cgrid, llC, label="loglik"); vline!([ℙ0.C], label="true val of C")

heatmap(σgrid, Cgrid,  llCσ)
heatmap(llCσ)
ℙ0.C
ℙ0.σy

plot(σgrid, llCσ[30,:])





# should be possible to compute the MLE
parnames = [:A, :α1, :C]


function loglik(ℙ0, θ, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids)
    tup = (; zip(parnames, θ)...) # try copy here
    ℙ = setproperties(ℙ0, tup)
    B = BackwardFilter(S, ℙ, AuxType, obs, obsvals, timegrids) ;
    _, ll = forwardguide(B, ℙ)(x0, Z);
    ll
end

loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids) = (θ) -> loglik(ℙ0, θ, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids)
 
using Optim
getpar(parnames, ℙ0)

θ = [4.0, 0.4, 220.0]
lower = [0.5, 0.1, 50.0]
upper = [10.0, 0.9, 250.0]

Optim.optimize(loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids), lower, upper, θ, SimulatedAnnealing())

ForwardDiff.gradient(loglik(ℙ0, x0, Z, parnames,  S, AuxType, obs, obsvals, timegrids), lower, upper, θ)