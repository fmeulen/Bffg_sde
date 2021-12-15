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