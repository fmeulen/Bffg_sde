function HFcSolver(
            ::Val{:outofplace},
            tt,
            xT_plus,
            P,
            obs,
            choices
        )
        access = Val{DiffusionDefinition.dimension(P).process}()

        function HFc_update(u, P, t)
            H, F, c = static_accessor_HFc(u, Val{d}())
            _B, _β, _σ, _a = DD.B(t, P), DD.β(t, P), DD.σ(t, P), DD.a(t, P)

            dH = - (_B' * H) - (H * _B) + outer(H * _σ)
            dF = - (_B' * F) + (H * _a * F) + (H * _β)
            dc = dot(_β, F) + 0.5*outer(F' * _σ) - 0.5*tr(H*_a)
            vcat(SVector(dH), dF, SVector(dc))
        end

        el = choices.eltype
        TH, TF, Tc = prepare_static_saving_types(Val{:hfc}(), access, el)
        prob = ODEProblem{false}(
            HFc_update,   # increment
            update_HFc(xT_plus, obs, access), # starting val
            (tt[end], tt[1]),   # time interval
            P  # parameter
        )
        saved_values = SavedValues(Float64, Tuple{TH,TF,Tc})
        callback = SavingCallback(
            (u,t,integrator) -> static_accessor_HFc(u, access),
            saved_values;
            saveat=reverse(tt),
            tdir=-1
        )
        integrator = init(
            prob,
            choices.solver,
            callback=callback,
            save_everystep=false, # to prevent wasting memory allocations
        )
        sol = solve!(integrator)   # solve the ODEs
        HFc0 = MVector(sol.u[end])
        Tsv, Ti, T0 = typeof(saved_values), typeof(integrator), typeof(HFc0)
        Ta = typeof(access)
        new{:outofplace,Tsv,Ti,Nothing,T0,Ta}(
            saved_values,
            integrator,
            nothing,
            HFc0,
            deepcopy(access),
        )
    end
end





function solve_and_ll!(
    XX,
    WW,
    P::GuidProp,
    ::AbstractGuidingTermSolver{:outofplace},
    y1;
    skip=0
)
yy, ww, tt = XX.x, WW.x, XX.t
N = length(XX)

yy[1] = DD.value(y1)
x = y1
ll = 0.0
for i in 1:(N-1)
    add_to_ll = (i < N-skip)
    s = tt[i]
    dt = tt[i+1] - tt[i]
    dW = ww[i+1] - ww[i]

    r_i = ∇logρ(i, x, P)
    b_i = DD.b(s, x, P.P_target)
    btil_i = DD.b(s, x, P.P_aux)

    σ_i = DD.σ(s, x, P.P_target)
    a_i = σ_i*σ_i'

    add_to_ll && (ll += dot(b_i-btil_i, r_i) * dt)

    if !DD.constdiff(P)
        H_i = H(i, x, P)
        atil_i = DD.a(s, x, P.P_aux)
        add_to_ll && (ll += 0.5*tr( (a_i - atil_i)*(r_i*r_i'-H_i') ) * dt)
    end

    x = x + (a_i*r_i + b_i)*dt + σ_i*dW

    yy[i+1] = DD.value(x)

    DD.bound_satisfied(P, yy[i+1]) || return false, -Inf
end
true, ll
end
