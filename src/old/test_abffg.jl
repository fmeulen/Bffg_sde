



################################  TESTING  ################################################
# settings sampler
iterations = 7_000 # 5*10^4
skip_it = 500  #1000
subsamples = 0:skip_it:iterations

T = 0.5
dt = 1/50000
τ(T) = (x) ->  x * (2-x/T)
tt = τ(T).(0.:dt:T)

sk = 0 # skipped in evaluating loglikelihood
const d=3

easy_conditioning = true
obs_scheme =["full","firstcomponent"][2]
ρ = obs_scheme=="full" ? 0.85 : 0.95
if obs_scheme=="full"
    LT = SMatrix{3,3}(1.0I)
    vT = easy_conditioning ?  ℝ{3}(1/32,1/4,1) :  ℝ{3}(5/128,3/8,2)
end
if obs_scheme=="firstcomponent"
    LT = @SMatrix [1. 0. 0.]
    vT = easy_conditioning ? ℝ{1}(1/32) : ℝ{1}(5/128)
end


ρ = 0.0

Σdiagel = 10e-9
m,  = size(LT)
ΣT = SMatrix{m,m}(Σdiagel*I)

# specify target process
struct NclarDiffusion <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.b(t, x, P::NclarDiffusion) = ℝ{3}(x[2],x[3],-P.α * sin(P.ω * x[3]))
Bridge.σ(t, x, P::NclarDiffusion) = ℝ{3}(0.0, 0.0, P.σ)
Bridge.constdiff(::NclarDiffusion) = true

jacobianb(t, x, P::NclarDiffusion) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 -P.α * P.ω * cos(P.ω * x[3])]

P = NclarDiffusion(6.0, 2pi, 1.0)
x0 = ℝ{3}(0.0, 0.0, 0.0)


# specify auxiliary process
struct NclarDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.B(t, P::NclarDiffusionAux) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 0.0]
Bridge.β(t, P::NclarDiffusionAux) = ℝ{3}(0.0,0.0,0)
Bridge.σ(t,  P::NclarDiffusionAux) = ℝ{3}(0.0,0.0, P.σ)
Bridge.constdiff(::NclarDiffusionAux) = true
Bridge.b(t, x, P::NclarDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)
Bridge.a(t, P::NclarDiffusionAux) = Bridge.σ(t,P) * Bridge.σ(t,  P)'
Bridge.a(t, x, P::NclarDiffusionAux) = Bridge.a(t,P) 

P̃ = NclarDiffusionAux(P.α, P.ω, P.σ)


# Solve Backward Recursion
ϵ = 10e-2  # choice not too important for bridges
PT_ = ϵ^(-1)*SMatrix{3,3}(1.0I)

FT_, HT_, CT_ = initFHC(PT_,vT, LT)
FT, HT, CT = fusion(FT_, HT_, CT_, vT, LT, ΣT)
𝒫 = PBridge(P, P̃, tt, FT, HT, CT)

####################### MH algorithm ###################
W = sample(tt, Wiener())
X = solve(Euler(), x0, W, P)
Xᵒ = copy(X)
solve!(Euler(),Xᵒ, x0, W, 𝒫)
solve!(Euler(),X, x0, W, 𝒫)
ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)


# Fold = 𝒫.F
# Hold = 𝒫.H
# # new constructor, adaptive
#𝒫 = PBridge(P, P̃, tt, FT, HT, CT, X)
 solve!(Euler(),Xᵒ, x0, W, 𝒫)
 solve!(Euler(),X, x0, W, 𝒫)
 ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)

# further initialisation
Wᵒ = copy(W)
W2 = copy(W)
XX = Any[]
if 0 in subsamples
    push!(XX, copy(X))
end

acc = 0

for iter in 1:iterations
    # Proposal
    global ll, acc, 𝒫
    sample!(W2, Wiener())
    #ρ = rand(Uniform(0.95,1.0))
    Wᵒ.yy .= ρ*W.yy + sqrt(1-ρ^2)*W2.yy
    solve!(Euler(),Xᵒ, x0, Wᵒ, 𝒫)


    llᵒ = llikelihood(Bridge.LeftRule(), Xᵒ, 𝒫,skip=sk)
    print("ll $ll $llᵒ, diff_ll: ",round(llᵒ-ll;digits=3))

    if log(rand()) <= llᵒ - ll
        X.yy .= Xᵒ.yy
        W.yy .= Wᵒ.yy
        ll = llᵒ
        print("✓")
        acc +=1
        # 𝒫 = PBridge(P, P̃, tt, FT, HT, CT, X)
        # ll = llikelihood(Bridge.LeftRule(), X, 𝒫, skip=sk)

    end

    println()
    if iter in subsamples
        push!(XX, copy(X))
    end
end

@info "Done."*"\x7"^6

# write mcmc iterates to csv file
extractcomp(v,i) = map(x->x[i], v)

iterates = [Any[s, tt[j], d, XX[i].yy[j][d]] for d in 1:3, j in 1:length(X), (i,s) in enumerate(subsamples) ][:]
df_iterates = DataFrame(iteration=extractcomp(iterates,1),time=extractcomp(iterates,2), component=extractcomp(iterates,3), value=extractcomp(iterates,4))
CSV.write(outdir*"iterates-"*obs_scheme*".csv",df_iterates)

ave_acc_perc = 100*round(acc/iterations;digits=2)

if true
    # write info to txt file
    fn = outdir*"info-"*obs_scheme*".txt"
    f = open(fn,"w")
    write(f, "Choice of observation schemes: ",obs_scheme,"\n")
    write(f, "Easy conditioning (means going up to 1 for the rough component instead of 2): ",string(easy_conditioning),"\n")
    write(f, "Number of iterations: ",string(iterations),"\n")
    write(f, "Skip every ",string(skip_it)," iterations, when saving to csv","\n\n")
    write(f, "Starting point: ",string(x0),"\n")
    write(f, "End time T: ", string(T),"\n")
    write(f, "Endpoint v: ",string(vT),"\n")
    write(f, "Noise Sigma: ",string(ΣT),"\n")
    write(f, "L: ",string(LT),"\n\n")
    write(f,"Mesh width: ",string(dt),"\n")
    write(f, "rho (Crank-Nicholsen parameter: ",string(ρ),"\n")
    write(f, "Average acceptance percentage: ",string(ave_acc_perc),"\n\n")
    #write(f, "Backward type parametrisation in terms of nu and H? ",string(νHparam),"\n")
    close(f)


    println("Average acceptance percentage: ",ave_acc_perc,"\n")
    println(obs_scheme)
end




