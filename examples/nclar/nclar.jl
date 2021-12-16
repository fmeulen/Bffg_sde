# target process
struct NclarDiffusion <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.b(t, x, ℙ::NclarDiffusion) = ℝ{3}(x[2],x[3],-ℙ.α * sin(ℙ.ω * x[3]))
Bridge.σ(t, x, ℙ::NclarDiffusion) = ℝ{3}(0.0, 0.0, ℙ.σ)
Bridge.constdiff(::NclarDiffusion) = true
dim(::NclarDiffusion) = 3

jacobianb(t, x, ℙ::NclarDiffusion) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 -ℙ.α * ℙ.ω * cos(ℙ.ω * x[3])]

#  auxiliary process
struct NclarDiffusionAux <: ContinuousTimeProcess{ℝ{3}}
    α::Float64
    ω::Float64
    σ::Float64
end

Bridge.B(t, ℙ::NclarDiffusionAux) = @SMatrix [0.0 1.0 0.0 ; 0.0 0.0 1.0 ; 0.0 0.0 0.0]
Bridge.β(t, ℙ::NclarDiffusionAux) = ℝ{3}(0.0,0.0,0)
Bridge.σ(t,  ℙ::NclarDiffusionAux) = ℝ{3}(0.0,0.0, ℙ.σ)

Bridge.constdiff(::NclarDiffusionAux) = true
dim(::NclarDiffusionAux) = 3 


# standardfunctions (no adjustment needed)
Bridge.b(t, x, ℙ::NclarDiffusionAux) = Bridge.B(t,ℙ) * x + Bridge.β(t,ℙ)
Bridge.a(t, ℙ::NclarDiffusionAux) = Bridge.σ(t,ℙ) * Bridge.σ(t,  ℙ)'
Bridge.a(t, x, ℙ::NclarDiffusionAux) = Bridge.a(t,ℙ) 

