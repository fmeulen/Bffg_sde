# just like JansenRit, except that there is diffusivity on components 4, 5 and 6

# target process
struct JansenRitDiffusion3{T} <: ContinuousTimeProcess{ℝ{6}}
    A::T
    a::T
    B::T 
    b::T 
    C::T
    α1::T 
    α2::T
    e0::T
    v0::T
    r::T
    μy::T
    σx::T
    σy::T
    σz::T
end

#  auxiliary process
struct JansenRitDiffusionAux3{T,Tobs} <: ContinuousTimeProcess{ℝ{6}}
    A::T
    a::T
    B::T 
    b::T 
    C::T
    α1::T 
    α2::T
    e0::T
    v0::T
    r::T
    μy::T
    σx::T
    σy::T
    σz::T
    T::Float64  # observation time
    vT::Tobs # observation value
end

JansenRitDiffusionAux3(T, vT, ℙ::JansenRitDiffusion3) = JansenRitDiffusionAux3(ℙ.A, ℙ.a, ℙ.B, ℙ.b, ℙ.C, ℙ.α1, ℙ.α2, 
ℙ.e0, ℙ.v0, ℙ.r, ℙ.μy, ℙ.σx, ℙ.σy, ℙ.σz, T, vT)

sigm(x, ℙ::Union{JansenRitDiffusion3, JansenRitDiffusionAux3}) = 2.0ℙ.e0 / (1.0 + exp(ℙ.r*(ℙ.v0 - x)))
μy(t, ℙ::Union{JansenRitDiffusion3, JansenRitDiffusionAux3}) =  ℙ.a * ℙ.A * ℙ.μy #constant
C1(ℙ::JansenRitDiffusion3) = ℙ.C
C2(ℙ::JansenRitDiffusion3) = ℙ.α1*ℙ.C
C3(ℙ::JansenRitDiffusion3) = ℙ.α2*ℙ.C
C4(ℙ::JansenRitDiffusion3) = ℙ.α2*ℙ.C


function Bridge.b(t, x, ℙ::JansenRitDiffusion3)
    @SVector [
        x[4],
        x[5],
        x[6],
        ℙ.A*ℙ.a*(sigm(x[2] - x[3], ℙ)) - 2ℙ.a*x[4] - ℙ.a*ℙ.a*x[1],
        μy(t, ℙ) + ℙ.A*ℙ.a*C2(ℙ)*sigm(C1(ℙ)*x[1], ℙ) - 2ℙ.a*x[5] - ℙ.a*ℙ.a*x[2],
        ℙ.B*ℙ.b*(C4(ℙ)*sigm(C3(ℙ)*x[1], ℙ)) - 2ℙ.b*x[6] - ℙ.b*ℙ.b*x[3]]
end

Bridge.σ(t, x, ℙ::JansenRitDiffusion3) =@SMatrix [0.0 0.0 0.0; 0.0 0.0 0.0;
                                            0.0 0.0 0.0;  ℙ.σx 0.0 0.0; 
                                        0.0 ℙ.σy 0.0;  0.0  0.0 ℙ.σz ]

    
wienertype(::JansenRitDiffusion3) = Wiener{ℝ{3}}()



Bridge.constdiff(::JansenRitDiffusion3) = true
dim(::JansenRitDiffusion3) = 6


# I would think this works nice, but this matrix is very ill-conditioned
function Bridge.B(t, ℙ::JansenRitDiffusionAux3)      
    @SMatrix [  0.0 0.0 0.0 1.0 0.0 0.0;
                0.0 0.0 0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 0.0 0.0 1.0;
                -ℙ.a^2 0.0 0.0 -2.0*ℙ.a 0.0 0.0;
                0.0 -ℙ.a^2 0.0 0.0 -2.0*ℙ.a 0.0;
                0.0 0.0 -ℙ.b^2 0.0 0.0 -2.0*ℙ.b]
end

# hence try (this gives way worse bridges, but the numerical problems disappear)
# function Bridge.B(t, ℙ::JansenRitDiffusionAux3)      
#     @SMatrix [  0.0 0.0 0.0 1.0 0.0 0.0;
#                 0.0 0.0 0.0 0.0 1.0 0.0;
#                 0.0 0.0 0.0 0.0 0.0 1.0;
#                 0.0 0.0 0.0 0.0 0.0 0.0;
#                 0.0 0.0 0.0 0.0 0.0 0.0;
#                 0.0 0.0 0.0 0.0 0.0 0.0]
# end

Bridge.β(t, ℙ::JansenRitDiffusionAux3) = @SVector [0.0, 0.0, 0.0, ℙ.A * ℙ.a * sigm(ℙ.vT, ℙ), μy(t,ℙ), 0.0]

# Bridge.β(t, P::JansenRitDiffusionAux3) = @SVector [0.0, 0.0, 0.0, P.A*P.a*0.5*P.νmax, 
#             P.μy +  P.A*P.a*P.α1*P.C*0.5 , P.B*P.b*P.α2*P.C*0.5*P.νmax]



Bridge.σ(t,  ℙ::JansenRitDiffusionAux3) = @SMatrix [0.0 0.0 0.0; 0.0 0.0 0.0;
0.0 0.0 0.0;  ℙ.σx 0.0 0.0; 
0.0 ℙ.σy 0.0;  0.0  0.0 ℙ.σz ]

Bridge.σ(t, x,  ℙ::JansenRitDiffusionAux3) = Bridge.σ(t,  ℙ)

Bridge.constdiff(::JansenRitDiffusionAux3) = true
dim(::JansenRitDiffusionAux3) = 6 


# standardfunctions (no adjustment needed)
Bridge.b(t, x, ℙ::JansenRitDiffusionAux3) = Bridge.B(t,ℙ) * x + Bridge.β(t,ℙ)
Bridge.a(t, ℙ::JansenRitDiffusionAux3) = Bridge.σ(t,ℙ) * Bridge.σ(t,  ℙ)'
Bridge.a(t, x, ℙ::JansenRitDiffusionAux3) = Bridge.a(t,ℙ) 


