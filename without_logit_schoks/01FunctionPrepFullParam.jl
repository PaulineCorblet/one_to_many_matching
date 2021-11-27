
function pseudo_IPFP(mrkt, α, γ)

    Φ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    [Φ[j,k] = γ[j,k] + sum(mrkt.K[k,i]*α[i,j,k] for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>3))

    @variable(model, m_U[i=1:mrkt.nbX])
    @variable(model, m_V[j=1:mrkt.nbY])

    # @constraint(model, norm, sum(mrkt.K[1,i]*m_U[i] for i=1:mrkt.nbX) == 1.0)
    # @constraint(model, norm, m_U[1] == 1.0)

    # @NLexpression(model, m_log_D[j=1:mrkt.nbY, k=1:mrkt.nbK], (Φ[j,k] - sum(mrkt.K[k,i]*(m_U[i]-log(mrkt.massX[i]/mrkt.K[k,i])) for i=1:mrkt.nbX)- (m_V[j]-log(mrkt.massY[j])))/(sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1))
    @NLexpression(model, m_log_D[j=1:mrkt.nbY, k=1:mrkt.nbK], (Φ[j,k] - sum(mrkt.K[k,i]*m_U[i] for i=1:mrkt.nbX)- m_V[j])/(sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1))
    @NLexpression(model, m_D[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_D[j,k]))

    @NLconstraint(model, marg_x[i=1:mrkt.nbX], sum(mrkt.K[k,i]*m_D[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK) == mrkt.massX[i])
    @NLconstraint(model, marg_y[j=1:mrkt.nbY], sum(m_D[j,k] for k=1:mrkt.nbK) == mrkt.massY[j])

    @NLobjective(model, Max, 1.0)
    JuMP.optimize!(model)

    return termination_status(model), value.(m_D),  value.(m_U),  value.(m_V)
end

function simulate_matching_wage(mrkt, params, with_resid = true, N_obs = 1e3)

    α, γ = generate_surplus(mrkt, params)

    term, μ_obs, U, V = pseudo_IPFP(mrkt, α, γ)

    W = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    # [W[j,k] = .5*((V[j]-γ[j,k])+sum(mrkt.K[k,i]*(α[i,j,k]-U[i]) for i=1:mrkt.nbX)) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    [W[j,k] = (1/(sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1))*(sum(mrkt.K[k,i]*(γ[j,k]-α[i,j,k]+U[i]-V[j]) for i=1:mrkt.nbX)) for j=1:mrkt.nbY, k=1:mrkt.nbK]

    if with_resid
        N_ky = convert.(Int64,round.(N_obs*μ_obs))
        W_obs = Array{Vector{Float64},2}(undef, mrkt.nbY, mrkt.nbK)
        for j in 1:mrkt.nbY
            for k in 1:mrkt.nbK
                ε = rand(Normal(), N_ky[j,k])
                W_obs_jk = ε .+ W[j,k]
                W_obs[j,k] = W_obs_jk
            end
        end

    else
        W_obs = W
    end

    return term, μ_obs, W_obs, U, V

end


function max_log_likelihood(mrkt, μ_obs, W_obs, nbCoeff, observed_wage = false)

    Σk = sum(mrkt.K, dims = 2)

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>3))

    @variable(model, m_log_λ[i=1:mrkt.nbX, j=1:mrkt.nbY], start = log(true_params.λ[i,j]))
    @variable(model, m_log_ρ[j=1:mrkt.nbY],  start = log(-true_params.ρ[j]))
    @variable(model, m_β[i=1:mrkt.nbX, j=1:mrkt.nbY, x=1:nbCoeff], start = true_params.β[i,j,x])

    @NLexpression(model, m_λ[i=1:mrkt.nbX, j=1:mrkt.nbY], exp(m_log_λ[i,j]))
    @NLexpression(model, m_ρ[j=1:mrkt.nbY], -exp(m_log_ρ[j]))

    @NLconstraint(model, consλ[j=1:mrkt.nbY], sum(m_λ[i,j] for i=1:mrkt.nbX) == 1.0)

    @NLexpression(model, m_α[i=1:mrkt.nbX, j=1:mrkt.nbY, k=1:mrkt.nbK],  sum(m_β[i,j,x]*Σk[k]^(x-1) for x=1:nbCoeff))
    # @NLexpression(model, m_γ[j=1:mrkt.nbY, k=1:mrkt.nbK], sum(m_λ[i,j]*(mrkt.K[k,i]^m_ρ[j]) for i=1:mrkt.nbX)^(1/m_ρ[j]))

    @NLexpression(model, m_log_γ[j=1:mrkt.nbY, k=1:mrkt.nbK], (1/m_ρ[j])*log(sum(m_λ[i,j]*(mrkt.K[k,i]^m_ρ[j]) for i=1:mrkt.nbX)))
    @NLexpression(model, m_γ[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_γ[j,k]))

    @variable(model, m_U[i=1:mrkt.nbX])
    @variable(model, m_V[j=1:mrkt.nbY])

    # @constraint(model, normU2, m_U[1] == U_obs[1])
    # @constraint(model, normU, sum(mrkt.K[1,i]*m_U[i] for i=1:mrkt.nbX) == sum(mrkt.K[1,i]*U_obs[i] for i=1:mrkt.nbX))
    # @constraint(model, normV, m_V[1] == V_obs[1])

    # @constraint(model, normβ[i=1:mrkt.nbX, j=1:mrkt.nbY], m_β[i,j,1] == true_params.β[i,j,1])
    # @constraint(model, normβ[j=1:mrkt.nbY], m_β[1,j,1] == true_params.β[1,j,1])
    @constraint(model, normβ[i=1:mrkt.nbX], m_β[i,1,1] == true_params.β[i,1,1])
    # @constraint(model, normβ, m_β[1,1,1] == true_params.β[1,1,1])

    # @constraint(model, norm, sum(mrkt.K[1,i]*(m_U[i]-true_U[i]) for i=1:mrkt.nbX) == 0.0)

    @NLexpression(model, m_log_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], (m_γ[j,k]-m_V[j] + sum(mrkt.K[k,i]*(m_α[i,j,k]-m_U[i]) for i=1:mrkt.nbX))/(Σk[k]+1))
    @NLexpression(model, m_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_μ[j,k]))

    @NLconstraint(model, marg_x[i=1:mrkt.nbX], sum(mrkt.K[k,i]*m_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK) == mrkt.massX[i])
    @NLconstraint(model, marg_y[j=1:mrkt.nbY], sum(m_μ[j,k] for k=1:mrkt.nbK) == mrkt.massY[j])

    @NLexpression(model, m_W[j=1:mrkt.nbY, k=1:mrkt.nbK], (1/(Σk[k]+1))*(sum(mrkt.K[k,i]*(m_γ[j,k]-m_α[i,j,k]+m_U[i]-m_V[j]) for i=1:mrkt.nbX)))

    # @variable(model, m_s)
    @variable(model, m_s >= 1e-8)
    @NLexpression(model, m_s2, (m_s)^2)

    # @NLexpression(model, m_L1, sum(μ_obs[j,k]*m_log_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK))
    @NLexpression(model, m_L1, sum((Σk[k]+1)*μ_obs[j,k]*m_log_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK))
    # @NLexpression(model, m_L1, sum(μ_obs[j,k]*(m_γ[j,k]-m_V[j] + sum(mrkt.K[k,i]*(m_α[i,j,k]-m_U[i]) for i=1:mrkt.nbX)) for j=1:mrkt.nbY, k=1:mrkt.nbK))
    # @NLexpression(model, m_L2, sum((W_obs[j,k]-m_W[j,k])^2/(2*m_s^2) - mrkt.nbY*mrkt.nbK*log(m_s) for j=1:mrkt.nbY, k=1:mrkt.nbK))

    if observed_wage
        @NLexpression(model, m_L2, -.5*sum(sum((Σk[k]+1)*(W_obs[j,k][l]-m_W[j,k])^2/m_s2+(Σk[k]+1)*log(m_s2) for l=1:length(W_obs[j,k]))  for j=1:mrkt.nbY, k=1:mrkt.nbK))
    else
        @NLexpression(model, m_L2, -.5*sum(μ_obs[j,k]*(W_obs[j,k]-m_W[j,k])^2 for j=1:mrkt.nbY, k=1:mrkt.nbK))
    end
    # @NLexpression(model, m_L2, -.5*sum((W_obs[j,k]-m_W[j,k])^2 for j=1:mrkt.nbY, k=1:mrkt.nbK))

    @NLobjective(model, Max, m_L1+m_L2)
    # @NLobjective(model, Max, m_L1)
    JuMP.optimize!(model)

    # return termination_status(model), SurplusParams(value.(m_λ),  value.(m_ρ),  value.(m_β)), value(m_s), value(m_L1), value(m_L2), value.(m_W), value.(m_μ), value.(m_U), value.(m_V)
    return termination_status(model), SurplusParams(value.(m_λ),  value.(m_ρ),  value.(m_β)), value(m_s)

end


    # [(value(m_V[j])-value(m_γ[j,k]))+sum(mrkt.K[k,i]*(value(m_α[i,j,k])-value(m_U[i])) for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # (W_obs-[(value(m_V[j])-value(m_γ[j,k]))+sum(mrkt.K[k,i]*(value(m_α[i,j,k])-value(m_U[i])) for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]).^2
    #
    # true_α, true_γ = generate_surplus(mrkt, true_params)
    #
    # model_α = Array{Float64,3}(undef, mrkt.nbX, mrkt.nbY, mrkt.nbK)
    # [model_α[i,j,k] =  sum(true_params.β[i,j,x]*Σk[k]^(x-1) for x=1:nbCoeff) for i=1:mrkt.nbX, j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # model_log_γ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    # [model_log_γ[j,k] = (1/true_params.ρ[j])*log(sum(true_params.λ[i,j]*(mrkt.K[k,i]^true_params.ρ[j]) for i=1:mrkt.nbX)) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # model_γ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    # [model_γ[j,k] = exp(model_log_γ[j,k]) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # model_term, model_μ, model_U, model_V = pseudo_IPFP(mrkt, model_α, model_γ)
    # term, μ_obs, true_U, true_V = pseudo_IPFP(mrkt, true_α, true_γ)
    #
    # model_W = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    # [model_W[j,k] = (1/(Σk[k]+1))*(sum(mrkt.K[k,i]*(model_γ[j,k]-model_α[i,j,k]+model_U[i]-model_V[j]) for i=1:mrkt.nbX)) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # sum((W_obs[j,k]-model_W[j,k]) for j=1:mrkt.nbY, k=1:mrkt.nbK)
    # [(W_obs[j,k]-model_W[j,k])^2 for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # W = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    # # [W[j,k] = .5*((V[j]-γ[j,k])+sum(mrkt.K[k,i]*(α[i,j,k]-U[i]) for i=1:mrkt.nbX)) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    # [W[j,k] = (1/(sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1))*(sum(mrkt.K[k,i]*(model_γ[j,k]-model_α[i,j,k]+model_U[i]-model_V[j]) for i=1:mrkt.nbX)) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # true_Φ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    # [true_Φ[j,k] = true_γ[j,k] + sum(mrkt.K[k,i]*true_α[i,j,k] for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #
    # m_Φ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    # [m_Φ[j,k] = value(m_γ[j,k]) + sum(mrkt.K[k,i]*value(m_α[i,j,k]) for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    #

    # while cont
    #     iter = iter+1
    #
    #     term, μ_t, W_t = simulate_matching_wage(mrkt, params, with_resid = false)
    #
    #     max_error =
    #
    #     println("Max error is ", max_error)
    #     if max_error <= tol
    #         cont = false
    #         println("Gradient descent converged")
    #     end
    #
    #     if iter>=maxiter
    #         cont = false
    #         println("Max number of iterations reached")
    #     end
    # end
