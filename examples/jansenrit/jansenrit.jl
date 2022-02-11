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
struct JansenRitDiffusionAux{T,Tobs,Tx1} <: ContinuousTimeProcess{ℝ{6}}
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
    x1::Tx1  # LinearInterpolation object of deterministic solution of x1 on interpolating time grid
    guidingterm_with_x1::Bool
end

JansenRitDiffusionAux(T, vT, x1, guidingterm_with_x1, P::JansenRitDiffusion) = 
        JansenRitDiffusionAux(P.A, P.a, P.B, P.b, P.C, P.α1, P.α2, P.νmax, P.v, P.r, P.μy, P.σy, T, vT, x1, guidingterm_with_x1)


sigm(x, P::Union{JansenRitDiffusion, JansenRitDiffusionAux}) = P.νmax / (1.0 + exp(P.r*(P.v - x)))
μy(t, P::Union{JansenRitDiffusion, JansenRitDiffusionAux}) =  P.a * P.A * P.μy #constant
C1(P::Union{JansenRitDiffusion, JansenRitDiffusionAux}) = P.C
C2(P::Union{JansenRitDiffusion, JansenRitDiffusionAux}) = P.α1*P.C
C3(P::Union{JansenRitDiffusion, JansenRitDiffusionAux}) = P.α2*P.C
C4(P::Union{JansenRitDiffusion, JansenRitDiffusionAux}) = P.α2*P.C


function Bridge.b(t, x, P::JansenRitDiffusion)
    SA[
        x[4],
        x[5],
        x[6],
        P.A*P.a*(sigm(x[2] - x[3], P)) - 2P.a*x[4] - P.a*P.a*x[1],
        μy(t, P) + P.A*P.a*C2(P)*sigm(C1(P)*x[1], P) - 2P.a*x[5] - P.a*P.a*x[2],
        P.B*P.b*(C4(P)*sigm(C3(P)*x[1], P)) - 2P.b*x[6] - P.b*P.b*x[3]]
end

Bridge.σ(t, x, P::JansenRitDiffusion) = SA[0.0, 0.0, 0.0, 0.0, P.σy, 0.0]

wienertype(::JansenRitDiffusion) = Wiener()

Bridge.constdiff(::JansenRitDiffusion) = true
Bridge.constdiff(::JansenRitDiffusionAux) = true
dim(::JansenRitDiffusion) = 6
dim(::JansenRitDiffusionAux) = 6 

Bridge.σ(t, P::JansenRitDiffusionAux) = SA[0.0, 0.0, 0.0, 0.0, P.σy, 0.0]

Bridge.a(t, P::JansenRitDiffusionAux) =   @SMatrix [  0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 P.σy^2 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0]

Bridge.a(t, x, P::JansenRitDiffusion) =   @SMatrix [  0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0;
                                                    0.0 0.0 0.0 0.0 P.σy^2 0.0;
                                                    0.0 0.0 0.0 0.0 0.0 0.0]


# This matrix is very ill-conditioned
function Bridge.B(t, P::JansenRitDiffusionAux)      
    @SMatrix [  0.0 0.0 0.0 1.0 0.0 0.0;
                0.0 0.0 0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 0.0 0.0 1.0;
                -P.a^2 0.0 0.0 -2.0*P.a 0.0 0.0;
                0.0 -P.a^2 0.0 0.0 -2.0*P.a 0.0;
                0.0 0.0 -P.b^2 0.0 0.0 -2.0*P.b]
end

Bridge.β(t, P::JansenRitDiffusionAux) = SA[0.0, 0.0, 0.0,  P.A * P.a * sigm(P.vT, P), μy(t,P), 0.0 ]
#Bridge.β(t, P::JansenRitDiffusionAux) = SA[0.0, 0.0, 0.0,  0.0, 0.0, 0.0 ]

Bridge.b(t, x, P::JansenRitDiffusionAux) = Bridge.B(t,P) * x + Bridge.β(t,P)



# adjust later, this version should be called if guidingter_with_x1 == true
# Bridge.β(t, P::JansenRitDiffusionAux) = SA[0.0, 0.0, 0.0, 
#                                     P.A * P.a * sigm(P.vT, P), 
#                                     μy(t,P) + P.guidingterm_with_x1 * P.A*P.a*C2(P)* sigm( C1(P)*P.x1(t) , P),
#                                     P.guidingterm_with_x1 * P.B*P.b*C4(P)* sigm( C3(P)*P.x1(t), P)]




# mulXσ(X,  P̃::JansenRitDiffusionAux) = P̃.σy * view(X, :, 5) #  P̃.σy * X[:,5]
# mulXσ(X,  P̃::JansenRitDiffusionAux) = [X[1,5], X[2,5], X[3,5], X[4,5],P̃.σy * X[5,5], X[6,5]]
# mulax(x,  P̃::JansenRitDiffusionAux) = (x[5] * P̃.σy) * Bridge.σ(0, P̃) 
# trXa(X,  P̃::JansenRitDiffusionAux) = X[5,5] * P̃.σy^2
# dotσx(x,  P̃::JansenRitDiffusionAux) = P̃.σy * x[5]



