# target process
struct JansenRitDiffusion{T} <: ContinuousTimeProcess{ℝ{6}}
    A::T
    a::T
    B::T 
    b::T 
    C::T
    α1::T 
    α2::T
    νmax::T
    v::T
    r::T
    μy::T
    σy::T
end

#  auxiliary process
struct JansenRitDiffusionAux{T,Tobs} <: ContinuousTimeProcess{ℝ{6}}
    A::T
    a::T
    B::T 
    b::T 
    C::T
    α1::T 
    α2::T
    νmax::T
    v::T
    r::T
    μy::T
    σy::T
    T::Float64  # observation time
    vT::Tobs # observation value
end

JansenRitDiffusionAux(T, vT, ℙ::JansenRitDiffusion) = JansenRitDiffusionAux(ℙ.A, ℙ.a, ℙ.B, ℙ.b, ℙ.C, ℙ.α1, ℙ.α2, ℙ.νmax, ℙ.v, ℙ.r, ℙ.μy, ℙ.σy, T, vT)


sigm(x, ℙ::Union{JansenRitDiffusion, JansenRitDiffusionAux}) = ℙ.νmax / (1.0 + exp(ℙ.r*(ℙ.v - x)))
μy(t, ℙ::Union{JansenRitDiffusion, JansenRitDiffusionAux}) =  ℙ.a * ℙ.A * ℙ.μy #constant
C1(ℙ::JansenRitDiffusion) = ℙ.C
C2(ℙ::JansenRitDiffusion) = ℙ.α1*ℙ.C
C3(ℙ::JansenRitDiffusion) = ℙ.α2*ℙ.C
C4(ℙ::JansenRitDiffusion) = ℙ.α2*ℙ.C


function Bridge.b(t, x, ℙ::JansenRitDiffusion)
    @SVector [
        x[4],
        x[5],
        x[6],
        ℙ.A*ℙ.a*(sigm(x[2] - x[3], ℙ)) - 2ℙ.a*x[4] - ℙ.a*ℙ.a*x[1],
        μy(t, ℙ) + ℙ.A*ℙ.a*C2(ℙ)*sigm(C1(ℙ)*x[1], ℙ) - 2ℙ.a*x[5] - ℙ.a*ℙ.a*x[2],
        ℙ.B*ℙ.b*(C4(ℙ)*sigm(C3(ℙ)*x[1], ℙ)) - 2ℙ.b*x[6] - ℙ.b*ℙ.b*x[3]]
end

Bridge.σ(t, x, ℙ::JansenRitDiffusion) =ℝ{6}(0.0, 0.0, 0.0, 0.0, ℙ.σy, 0.0)
    
wienertype(::JansenRitDiffusion) = Wiener()

Bridge.constdiff(::JansenRitDiffusion) = true
dim(::JansenRitDiffusion) = 6


# I would think this works nice, but this matrix is very ill-conditioned
function Bridge.B(t, ℙ::JansenRitDiffusionAux)      
    @SMatrix [  0.0 0.0 0.0 1.0 0.0 0.0;
                0.0 0.0 0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 0.0 0.0 1.0;
                -ℙ.a^2 0.0 0.0 -2.0*ℙ.a 0.0 0.0;
                0.0 -ℙ.a^2 0.0 0.0 -2.0*ℙ.a 0.0;
                0.0 0.0 -ℙ.b^2 0.0 0.0 -2.0*ℙ.b]
end

# hence try (this gives way worse bridges, but the numerical problems disappear)
# function Bridge.B(t, ℙ::JansenRitDiffusionAux)      
#     @SMatrix [  0.0 0.0 0.0 1.0 0.0 0.0;
#                 0.0 0.0 0.0 0.0 1.0 0.0;
#                 0.0 0.0 0.0 0.0 0.0 1.0;
#                 0.0 0.0 0.0 0.0 0.0 0.0;
#                 0.0 0.0 0.0 0.0 0.0 0.0;
#                 0.0 0.0 0.0 0.0 0.0 0.0]
# end

Bridge.β(t, ℙ::JansenRitDiffusionAux) = @SVector [0.0, 0.0, 0.0, ℙ.A * ℙ.a * sigm(ℙ.vT, ℙ), μy(t,ℙ), 0.0]

# Bridge.β(t, P::JansenRitDiffusionAux) = @SVector [0.0, 0.0, 0.0, P.A*P.a*0.5*P.νmax, 
#             P.μy +  P.A*P.a*P.α1*P.C*0.5 , P.B*P.b*P.α2*P.C*0.5*P.νmax]



Bridge.σ(t,  ℙ::JansenRitDiffusionAux) = ℝ{6}(0.0, 0.0, 0.0, 0.0, ℙ.σy, 0.0)

Bridge.σ(t, x,  ℙ::JansenRitDiffusionAux) = Bridge.σ(t,  ℙ)

Bridge.constdiff(::JansenRitDiffusionAux) = true
dim(::JansenRitDiffusionAux) = 6 


# standardfunctions (no adjustment needed)
Bridge.b(t, x, ℙ::JansenRitDiffusionAux) = Bridge.B(t,ℙ) * x + Bridge.β(t,ℙ)
Bridge.a(t, ℙ::JansenRitDiffusionAux) = Bridge.σ(t,ℙ) * Bridge.σ(t,  ℙ)'
Bridge.a(t, x, ℙ::JansenRitDiffusionAux) = Bridge.a(t,ℙ) 


