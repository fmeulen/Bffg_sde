
"""
    update_Message(M, tup)

    Construct new instance of Message, with fields in ℙ and ℙ̃ adjusted according to tup
    
    M = Ms[3]
    tup = (C=3333333.1, A=3311.0)
    Mup = update_Message(M,tup)
"""
function update_Message(M::Message,tup)
    # adjust ℙ
    P_ = M.ℙ
    P_ = setproperties(P_, tup)
    @set! M.ℙ = P_
    # adjust ℙ̃
    P̃_ = M.ℙ̃
    P̃_ = setproperties(P̃_, tup)
    @set! M.ℙ̃ = P̃_
    M
end    


"""
    update_Messagees!(Ms, tup)

    Construct new instance of Message, with fields in ℙ and ℙ̃ adjusted according to tup
    Do this for each element of Ms and write into it

    tup = (C=3333333.1, A=3311.0)
    update_Messagees!(Ms,tup)
"""
function update_Messagees!(Ms, tup)
    for i ∈ eachindex(Ms)
        Ms[i] = update_Message(Ms[i], tup)
    end
end


