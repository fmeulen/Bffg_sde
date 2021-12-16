# target process

struct FitzhughDiffusion <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
end

Bridge.b(t, x, ℙ::FitzhughDiffusion) = ℝ{2}((x[1]-x[2]-x[1]^3+ℙ.s)/ℙ.ϵ, ℙ.γ*x[1]-x[2] +ℙ.β)
Bridge.σ(t, x, ℙ::FitzhughDiffusion) = ℝ{2}(0.0, ℙ.σ)
Bridge.constdiff(::FitzhughDiffusion) = true
dim(::FitzhughDiffusion) = 2

#  auxiliary process

struct FitzhughDiffusionAux <: ContinuousTimeProcess{ℝ{2}}
    ϵ::Float64
    s::Float64
    γ::Float64
    β::Float64
    σ::Float64
    t::Float64
    u::Float64
    T::Float64
    v::Float64
end

function uv(t, ℙ::FitzhughDiffusionAux)
    λ = (t - ℙ.t)/(ℙ.T - ℙ.t)
    ℙ.v*λ + ℙ.u*(1-λ)
end

        
if aux_choice=="linearised_end"
    Bridge.B(t, ℙ::FitzhughDiffusionAux) = @SMatrix [1/ℙ.ϵ-3*ℙ.v^2/ℙ.ϵ  -1/ℙ.ϵ; ℙ.γ -1.0]
    Bridge.β(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(ℙ.s/ℙ.ϵ+2*ℙ.v^3/ℙ.ϵ, ℙ.β)
#    ρ = endpoint=="extreme" ? 0.9 : 0.0
elseif aux_choice=="linearised_startend"
    Bridge.B(t, ℙ::FitzhughDiffusionAux) = @SMatrix [1/ℙ.ϵ-3*uv(t, ℙ)^2/ℙ.ϵ  -1/ℙ.ϵ; ℙ.γ -1.0]
    Bridge.β(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(ℙ.s/ℙ.ϵ+2*uv(t, ℙ)^3/ℙ.ϵ, ℙ.β)
 #   ρ = endpoint=="extreme" ? 0.98 : 0.0
else
    Bridge.B(t, ℙ::FitzhughDiffusionAux) = @SMatrix [1/ℙ.ϵ  -1/ℙ.ϵ; ℙ.γ -1.0]
    Bridge.β(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(ℙ.s/ℙ.ϵ-(ℙ.v^3)/ℙ.ϵ, ℙ.β)
 #   ρ = 0.99
end
Bridge.σ(t, ℙ::FitzhughDiffusionAux) = ℝ{2}(0.0, ℙ.σ)

Bridge.constdiff(::FitzhughDiffusionAux) = true
dim(::FitzhughDiffusionAux) = 2

# standardfunctions (no adjustment needed)
Bridge.b(t, x, ℙ::FitzhughDiffusionAux) = Bridge.B(t,ℙ) * x + Bridge.β(t,ℙ)
Bridge.a(t, ℙ::FitzhughDiffusionAux) = Bridge.σ(t,ℙ) * Bridge.σ(t, ℙ)'
Bridge.a(t, x, ℙ::FitzhughDiffusionAux) = Bridge.a(t,ℙ) 





