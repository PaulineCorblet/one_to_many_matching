
function IPFP_knownhet(mrkt, α, γ, massXL)

    Φ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    [Φ[j,k] = γ[j,k] + sum(sum(mrkt.K[k,(i-1)*nbL+l] for l=1:nbL)*α[i,j] for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>3))

    # @variable(model, m_U[l=1:mrkt.nbL, i=1:mrkt.nbX])
    @variable(model, m_U[il=1:(mrkt.nbX*mrkt.nbL)])
    @variable(model, m_V[j=1:mrkt.nbY])

    # @constraint(model, norm, sum(mrkt.K[1,i]*m_U[i] for i=1:mrkt.nbX) == 1.0)
    # @constraint(model, norm, m_U[1] == 1.0)

    # @NLexpression(model, m_log_D[j=1:mrkt.nbY, k=1:mrkt.nbK], (Φ[j,k] - sum(mrkt.K[k,i]*(m_U[i]-log(mrkt.massX[i]/mrkt.K[k,i])) for i=1:mrkt.nbX)- (m_V[j]-log(mrkt.massY[j])))/(sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1))
    @NLexpression(model, m_log_D[j=1:mrkt.nbY, k=1:mrkt.nbK], (Φ[j,k] - sum(mrkt.K[k,(i-1)*nbL+l]*m_U[(i-1)*nbL+l] for l=1:mrkt.nbL, i=1:mrkt.nbX)- m_V[j])/(sum(mrkt.K[k,il] for il=1:(mrkt.nbX*mrkt.nbL))+1))
    @NLexpression(model, m_D[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_D[j,k]))

    @NLexpression(model, m_D_x0[il=1:(mrkt.nbX*mrkt.nbL)], exp(-m_U[il]))
    @NLexpression(model, m_D_0y[j=1:mrkt.nbY], exp(-m_V[j]))

    @NLconstraint(model, marg_x[il=1:(mrkt.nbX*mrkt.nbL)], sum(mrkt.K[k,il]*m_D[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK)+m_D_x0[il] == massXL[il])
    @NLconstraint(model, marg_y[j=1:mrkt.nbY], sum(m_D[j,k] for k=1:mrkt.nbK)+m_D_0y[j] == mrkt.massY[j])

    @NLobjective(model, Max, 1.0)
    JuMP.optimize!(model)

    return termination_status(model), value.(m_D), value.(m_D_x0), value.(m_D_0y), value.(m_U),  value.(m_V)
end

function min_wage_distance_MPEC(mrkt, params, obs_total_wage)

    α, γ = generate_surplus(mrkt, params, nbL)

    Φ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    [Φ[j,k] = γ[j,k] + sum(sum(mrkt.K[k,(i-1)*nbL+l] for l=1:nbL)*α[i,j] for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))

    @variable(model, m_U[il=1:(mrkt.nbX*mrkt.nbL)])
    @variable(model, m_V[j=1:mrkt.nbY])

    @NLexpression(model, m_log_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], (Φ[j,k] - sum(mrkt.K[k,(i-1)*nbL+l]*m_U[(i-1)*nbL+l] for l=1:mrkt.nbL, i=1:mrkt.nbX)- m_V[j])/(sum(mrkt.K[k,il] for il=1:(mrkt.nbX*mrkt.nbL))+1))
    @NLexpression(model, m_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_μ[j,k]))

    @NLexpression(model, m_μ_x0[il=1:(mrkt.nbX*mrkt.nbL)], exp(-m_U[il]))
    @NLexpression(model, m_μ_0y[j=1:mrkt.nbY], exp(-m_V[j]))

    @variable(model, m_log_massXL[il=1:(mrkt.nbX*mrkt.nbL)])
    @NLexpression(model, m_massXL[il=1:(mrkt.nbX*mrkt.nbL)], exp(m_log_massXL[il]))

    @NLconstraint(model, marg_l[i=1:mrkt.nbX], sum(m_massXL[(i-1)*nbL+l] for l=1:mrkt.nbL) == mrkt.massX[i])

    @NLconstraint(model, marg_x[il=1:(mrkt.nbX*mrkt.nbL)], sum(mrkt.K[k,il]*m_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK)+m_μ_x0[il] == m_massXL[il])
    @NLconstraint(model, marg_y[j=1:mrkt.nbY], sum(m_μ[j,k] for k=1:mrkt.nbK)+m_μ_0y[j] == mrkt.massY[j])

    @NLexpression(model, m_total_wage[j=1:mrkt.nbY, k=1:mrkt.nbK], log(mrkt.massY[j])+γ[j,k]-m_V[j]-m_log_μ[j,k])

    @NLobjective(model, Min, sum((m_total_wage[j,k] - obs_total_wage[j,k])^2 for j=1:mrkt.nbY, k=1:mrkt.nbK))
    JuMP.optimize!(model)

    return termination_status(model), value.(m_massXL)


end


function max_log_likelihood_MPEC(mrkt, massXL, obs_μ, obs_μ_x0, obs_μ_0y)

    # Warning I assume right away that α is everywhere 0

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))

    @variable(model, m_log_λ[j=1:mrkt.nbY])
    @variable(model, m_log_δ[j=1:mrkt.nbY])
    @variable(model, m_log_ρ)
    @variable(model, m_log_τ)

    @variable(model, m_log_ξ[l=1:mrkt.nbL, i=1:mrkt.nbX])

    @NLexpression(model, m_λ[j=1:mrkt.nbY], exp(m_log_λ[j]))
    @NLexpression(model, m_δ[j=1:mrkt.nbY], exp(m_log_δ[j]))
    @NLexpression(model, m_ρ, -exp(m_log_ρ))
    @NLexpression(model, m_τ, -exp(m_log_τ))

    @NLexpression(model, m_ξ[l=1:mrkt.nbL, i=1:mrkt.nbX], exp(m_log_ξ[l,i]))
    @NLconstraint(model, consξ[i=1:mrkt.nbX], sum(m_ξ[l,i] for l=1:mrkt.nbL) == 1.0)

    # @NLexpression(model, m_log_intγ[j=1:mrkt.nbY, k=1:mrkt.nbK],  (1/m_τ)*log(m_λ[j]*sum(true_params.ξ[l,2]*mrkt.K[k,mrkt.nbL+l] for l=1:mrkt.nbL)^m_τ + (1-m_λ[j])*sum(true_params.ξ[l,1]*mrkt.K[k,l] for l=1:mrkt.nbL)^m_τ))
    @NLexpression(model, m_log_intγ[j=1:mrkt.nbY, k=1:mrkt.nbK],  (1/m_τ)*log(m_λ[j]*sum(m_ξ[l,2]*mrkt.K[k,mrkt.nbL+l] for l=1:mrkt.nbL)^m_τ + (1-m_λ[j])*sum(m_ξ[l,1]*mrkt.K[k,l] for l=1:mrkt.nbL)^m_τ))

    @NLexpression(model, m_intγ[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_intγ[j,k]))

    # @NLexpression(model, m_log_γ[j=1:mrkt.nbY, k=1:mrkt.nbK], (1/m_ρ)*log(m_δ[j]*sum(true_params.ξ[l,3]*mrkt.K[k,2*mrkt.nbL+l] for l=1:mrkt.nbL)^m_ρ+ (1-m_δ[j])*m_intγ[j,k]^m_ρ))
    @NLexpression(model, m_log_γ[j=1:mrkt.nbY, k=1:mrkt.nbK], (1/m_ρ)*log(m_δ[j]*sum(m_ξ[l,3]*mrkt.K[k,2*mrkt.nbL+l] for l=1:mrkt.nbL)^m_ρ+ (1-m_δ[j])*m_intγ[j,k]^m_ρ))
    @NLexpression(model, m_γ[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_γ[j,k]))

    @variable(model, m_U[il=1:(mrkt.nbX*mrkt.nbL)])
    @variable(model, m_V[j=1:mrkt.nbY])

    @NLexpression(model, m_log_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], (m_γ[j,k] - sum(mrkt.K[k,il]*m_U[il] for il=1:(mrkt.nbX*mrkt.nbL))- m_V[j])/(sum(mrkt.K[k,il] for il=1:(mrkt.nbX*mrkt.nbL))+1))
    @NLexpression(model, m_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_μ[j,k]))

    @NLexpression(model, m_μ_x0[il=1:(mrkt.nbX*mrkt.nbL)], exp(-m_U[il]))
    @NLexpression(model, m_μ_0y[j=1:mrkt.nbY], exp(-m_V[j]))

    @NLconstraint(model, marg_x[il=1:(mrkt.nbX*mrkt.nbL)], sum(mrkt.K[k,il]*m_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK)+m_μ_x0[il] == massXL[il])
    @NLconstraint(model, marg_y[j=1:mrkt.nbY], sum(m_μ[j,k] for k=1:mrkt.nbK)+m_μ_0y[j] == mrkt.massY[j])

    @NLexpression(model, m_L1, sum((sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1)*obs_μ[j,k]*m_log_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK)+sum(log(m_μ_x0[i])*obs_μ_x0[i] for i=1:mrkt.nbX)+sum(log(m_μ_0y[j])*obs_μ_0y[j] for j=1:mrkt.nbY))

    @NLobjective(model, Max, m_L1)
    JuMP.optimize!(model)

    m_β = zeros(mrkt.nbX, mrkt.nbY)

    return termination_status(model), SurplusParams(value.(m_λ), value.(m_δ), value(m_τ), value(m_ρ), value.(m_ξ), m_β), objective_value(model)

end

function next_max_log_likelihood_MPEC(mrkt, params, obs_μ, obs_μ_x0, obs_μ_0y)

    # Warning I assume right away that α is everywhere 0

    α, γ = generate_surplus(mrkt, params, nbL)

    Φ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    [Φ[j,k] = γ[j,k] + sum(sum(mrkt.K[k,(i-1)*nbL+l] for l=1:nbL)*α[i,j] for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>0))

    @variable(model, m_log_prop_massL[l=1:mrkt.nbL, i=1:mrkt.nbX])
    @NLexpression(model, m_prop_massL[l=1:mrkt.nbL, i=1:mrkt.nbX], exp(m_log_prop_massL[l,i]))

    @NLconstraint(model, cons_prop_massL[i=1:mrkt.nbX], sum(m_prop_massL[l,i] for l=1:mrkt.nbL) == 1.0)

    @variable(model, m_U[il=1:(mrkt.nbX*mrkt.nbL)])
    @variable(model, m_V[j=1:mrkt.nbY])

    @NLexpression(model, m_log_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], (γ[j,k] - sum(mrkt.K[k,il]*m_U[il] for il=1:(mrkt.nbX*mrkt.nbL))- m_V[j])/(sum(mrkt.K[k,il] for il=1:(mrkt.nbX*mrkt.nbL))+1))
    @NLexpression(model, m_μ[j=1:mrkt.nbY, k=1:mrkt.nbK], exp(m_log_μ[j,k]))

    @NLexpression(model, m_μ_x0[il=1:(mrkt.nbX*mrkt.nbL)], exp(-m_U[il]))
    @NLexpression(model, m_μ_0y[j=1:mrkt.nbY], exp(-m_V[j]))

    @NLconstraint(model, marg_x[l=1:mrkt.nbL, i=1:mrkt.nbX], sum(mrkt.K[k,(i-1)*nbL+l]*m_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK)+m_μ_x0[(i-1)*nbL+l] == m_prop_massL[l,i]*mrkt.massX[i])
    @NLconstraint(model, marg_y[j=1:mrkt.nbY], sum(m_μ[j,k] for k=1:mrkt.nbK)+m_μ_0y[j] == mrkt.massY[j])

    @NLexpression(model, m_L1, sum((sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1)*obs_μ[j,k]*m_log_μ[j,k] for j=1:mrkt.nbY, k=1:mrkt.nbK)+sum(log(m_μ_x0[i])*obs_μ_x0[i] for i=1:mrkt.nbX)+sum(log(m_μ_0y[j])*obs_μ_0y[j] for j=1:mrkt.nbY))

    @NLobjective(model, Max, m_L1)
    JuMP.optimize!(model)

    m_β = zeros(mrkt.nbX, mrkt.nbY)

    return termination_status(model), value.(m_prop_massL), objective_value(model)

end
#
# function γ_func(mrkt, params, propL, nbL, j, k)
#     Ki = [sum(params.ξ[l,i]*propL[l,i]*mrkt.K[k ,i] for l=1:nbL) for i=1:mrkt.nbX]
#     return (params.δ[j]*Ki[3]^params.ρ+(1-params.δ[j])*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ))^(1/params.ρ)
# end
#
# function ∂γ_∂λ(mrkt, params, propL, nbL, j, k)
#     Ki = [sum(params.ξ[l,i]*propL[l,i]*mrkt.K[k ,i] for l=1:nbL) for i=1:mrkt.nbX]
#     return (1/params.τ)*(1-params.δ[j])*(Ki[2]^params.τ-Ki[1]^params.τ)*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ-1)*(params.δ[j]*Ki[3]^params.ρ+(1-params.δ[j])*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ))^(1/params.ρ-1)
# end
#
# function ∂γ_∂δ(mrkt, params, propL, nbL, j, k)
#     Ki = [sum(params.ξ[l,i]*propL[l,i]*mrkt.K[k ,i] for l=1:nbL) for i=1:mrkt.nbX]
#     return (1/params.ρ)*(Ki[3]^params.ρ-(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ))*(params.δ[j]*Ki[3]^params.ρ+(1-params.δ[j])*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ))^(1/params.ρ-1)
# end
#
# function ∂γ_∂ρ(mrkt, params, propL, nbL, j, k)
#     Ki = [sum(params.ξ[l,i]*propL[l,i]*mrkt.K[k ,i] for l=1:nbL) for i=1:mrkt.nbX]
#     return γ_func(mrkt, params, propL, nbL, j, k)*((-params.ρ^(-2))*log(γ_func(mrkt, params, propL, nbL, j, k)^params.ρ)+(1/params.ρ)*(params.δ[j]*log(Ki[3])*Ki[3]^params.ρ+(1-params.δ[j])*log((params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(1/params.τ))*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ))/(params.δ[j]*Ki[3]^params.ρ+(1-params.δ[j])*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ)))
# end
#
# function ∂γ_∂τ(mrkt, params, propL, nbL, j, k)
#     Ki = [sum(params.ξ[l,i]*propL[l,i]*mrkt.K[k ,i] for l=1:nbL) for i=1:mrkt.nbX]
#     return (1/params.ρ)*(1-params.δ[j])*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ)*γ_func(mrkt, params, propL, nbL, j, k)^(params.ρ*(1/params.ρ-1))*(-(1/params.τ^2)*(params.ρ*log(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ))+params.ρ*(params.λ[j]*log(Ki[2])*Ki[2]^params.τ+(1-params.λ[j])*log(Ki[1])*Ki[1]^params.τ)/(params.τ*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)))
# end
#
# function ∂γ_∂ξ(mrkt, params, propL, nbL, j, k, i, l)
#     Ki = [sum(params.ξ[l,i]*propL[l,i]*mrkt.K[k ,i] for l=1:nbL) for i=1:mrkt.nbX]
#     if i == 3
#         obj = params.δ[j]*propL[l,i]*mrkt.K[k,i]*Ki[i]^(params.ρ-1)*γ_func(mrkt, params, propL, nbL, j, k)^(params.ρ*(1/params.ρ-1))
#     elseif i == 2
#         obj = (1-params.δ[j])*params.λ[j]*propL[l,i]*mrkt.K[k,i]*Ki[i]^(params.τ-1)*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ-1)*γ_func(mrkt, params, propL, nbL, j, k)^(params.ρ*(1/params.ρ-1))
#     elseif i == 1
#         obj = (1-params.δ[j])*(1-params.λ[j])*propL[l,i]*mrkt.K[k,i]*Ki[i]^(params.τ-1)*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ-1)*γ_func(mrkt, params, propL, nbL, j, k)^(params.ρ*(1/params.ρ-1))
#     else println("i does not correspond to a worker type")
#     end
#
#     return obj
# end
#
# function ∂γ_∂propL(mrkt, params, propL, nbL, j, k, i, l)
#     Ki = [sum(params.ξ[l,i]*propL[l,i]*mrkt.K[k ,i] for l=1:nbL) for i=1:mrkt.nbX]
#     if i == 3
#         obj = params.δ[j]*params.ξ[l,i]*mrkt.K[k,i]*Ki[i]^(params.ρ-1)*γ_func(mrkt, params, propL, nbL, j, k)^(params.ρ*(1/params.ρ-1))
#     elseif i == 2
#         obj = (1-params.δ[j])*params.λ[j]*params.ξ[l,i]*mrkt.K[k,i]*Ki[i]^(params.τ-1)*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ-1)*γ_func(mrkt, params, propL, nbL, j, k)^(params.ρ*(1/params.ρ-1))
#     elseif i == 1
#         obj = (1-params.δ[j])*(1-params.λ[j])*params.ξ[l,i]*mrkt.K[k,i]*Ki[i]^(params.τ-1)*(params.λ[j]*Ki[2]^params.τ+(1-params.λ[j])*Ki[1]^params.τ)^(params.ρ/params.τ-1)*γ_func(mrkt, params, propL, nbL, j, k)^(params.ρ*(1/params.ρ-1))
#     else println("i does not correspond to a worker type")
#     end
#
#     return obj
# end
#
# # ϵ = zeros(nbL, mrkt.nbX)
# # ϵ[1,2] = 1e-12
# # A = γ_func(mrkt, true_params, propL, nbL, 1,1)
# # B = γ_func(mrkt, true_params, propL+ϵ, nbL, 1,1)
# #
# # (B-A)/1e-12
#
# function step_t(mrkt, params0, propL, nbL, obs_μ, obs_μ_x0, obs_μ_0y, β, maxiterML = 1e3, tolML = 1e-6, stepML = 1e-1)
#
#     params_t = SurplusParams(params0.λ, params0.δ, params0.τ, params0.ρ, params0.ξ, β)
#
#     cont = true
#     iter = 0
#
#     max_error = 0.0
#
#     while cont
#
#         iter = iter+1
#
#         α_t, γ_t = generate_surplus(mrkt, params_t, nbL, propL)
#         term_t, μ_t, μ_x0_t, μ_0y_t, U_t, V_t = IPFP_knownhet(mrkt, nbL, propL, α_t, γ_t)
#
#         likelihood = sum((sum(mrkt.K[k,i] for i=1:mrkt.nbX)+1)*obs_μ[j,k]*log(μ_t[j,k]) for j=1:mrkt.nbY, k=1:mrkt.nbK)+sum(log(μ_x0_t[i])*obs_μ_x0[i] for i=1:mrkt.nbX)+sum(log(μ_0y_t[j])*obs_μ_0y[j] for j=1:mrkt.nbY)
#
#         λnext = params_t.λ + stepML*[sum(∂γ_∂λ(mrkt, params_t, propL, nbL, j, k)*(obs_μ[j,k]-μ_t[j,k]) for k=1:mrkt.nbK) for j=1:mrkt.nbY]
#         δnext = params_t.δ + stepML*[sum(∂γ_∂δ(mrkt, params_t, propL, nbL, j, k)*(obs_μ[j,k]-μ_t[j,k]) for k=1:mrkt.nbK) for j=1:mrkt.nbY]
#
#         τnext = params_t.τ + stepML*sum(∂γ_∂τ(mrkt, params_t, propL, nbL, j, k)*(obs_μ[j,k]-μ_t[j,k]) for k=1:mrkt.nbK, j=1:mrkt.nbY)
#         ρnext = params_t.ρ + stepML*sum(∂γ_∂ρ(mrkt, params_t, propL, nbL, j, k)*(obs_μ[j,k]-μ_t[j,k]) for k=1:mrkt.nbK, j=1:mrkt.nbY)
#
#         ξnext = params_t.ξ + stepML*[sum(∂γ_∂ξ(mrkt, params_t, propL, nbL, j, k, i, l)*(obs_μ[j,k]-μ_t[j,k]) for k=1:mrkt.nbK, j=1:mrkt.nbY) for l=1:nbL, i=1:mrkt.nbX]
#
#         if iter>=maxiterML
#             cont = false
#             println("Max number of iterations reached")
#         end
#
#         max_error = findmax(abs.(vcat(λnext, δnext, τnext, ρnext)-vcat(params_t.λ, params_t.δ, params_t.τ, params_t.ρ)))[1]
#         # println("Max error is ", max_error)
#         if mod(iter, 100) == 0
#             println("Outer likelihood is ", likelihood)
#         end
#
#         if findmax(max_error)[1] <= tolML
#             cont = false
#             println("Gradient descent converged")
#         end
#
#         params_t = SurplusParams(λnext, δnext, τnext, ρnext, ξnext, β)
#
#     end
#
#     return max_error, params_t
#
# end
#
# function step_next_t_MPEC(next_mrkt, params, nbL, next_obs_μ, next_obs_μ_x0, next_obs_μ_0y)
#
#     model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>3))
#
#     # @variable(model, m_log_propL[l=1:nbL, i=1:next_mrkt.nbX] >= log(.1), start = log(propL[l,i]))
#     # @NLexpression(model, m_propL[l=1:nbL, i=1:next_mrkt.nbX], exp(m_log_propL[l,i]))
#
#     @variable(model, m_log_k[l=1:nbL, i=1:next_mrkt.nbX, k=1:mrkt.nbK])
#     @NLexpression(model, m_k[l=1:nbL, i=1:next_mrkt.nbX], exp(m_log_k[l,i]))
#
#     # @NLconstraint(model, cons_propL[i=1:next_mrkt.nbX], sum(m_propL[l,i] for l=1:nbL) == 1.0)
#     @NLconstraint(model, cons_k[i=1:next_mrkt.nbX], sum(m_k[l,i] for l=1:nbL) == 1.0)
#
#     @NLexpression(model, m_α[i=1:next_mrkt.nbX, j=1:next_mrkt.nbY],  params.β[i,j])
#
#     @NLexpression(model, m_log_intγ[j=1:next_mrkt.nbY, k=1:next_mrkt.nbK],  (1/params.τ)*log(params.λ[j]*sum(params.ξ[l,2]*m_propL[l,2]*next_mrkt.K[k,2] for l=1:nbL)^params.τ + (1-params.λ[j])*sum(params.ξ[l,1]*m_propL[l,1]*next_mrkt.K[k,1] for l=1:nbL)^params.τ))
#
#     @NLexpression(model, m_intγ[j=1:next_mrkt.nbY, k=1:next_mrkt.nbK], exp(m_log_intγ[j,k]))
#
#     @NLexpression(model, m_log_γ[j=1:next_mrkt.nbY, k=1:next_mrkt.nbK], (1/params.ρ)*log(params.δ[j]*sum(params.ξ[l,3]*m_propL[l,3]*next_mrkt.K[k,3] for l=1:nbL)^params.ρ+ (1-params.δ[j])*m_intγ[j,k]^params.ρ))
#     @NLexpression(model, m_γ[j=1:next_mrkt.nbY, k=1:next_mrkt.nbK], exp(m_log_γ[j,k]))
#
#     @NLexpression(model, m_Φ[j=1:next_mrkt.nbY, k=1:next_mrkt.nbK], 1e1*(m_γ[j,k] + sum(next_mrkt.K[k,i]*m_α[i,j] for i=1:next_mrkt.nbX)))
#
#     @variable(model, m_U[i=1:next_mrkt.nbX], start = next_obs_U[i])
#     @variable(model, m_V[j=1:next_mrkt.nbY], start = next_obs_V[j])
#
#     @NLexpression(model, m_log_μ[j=1:next_mrkt.nbY, k=1:next_mrkt.nbK], (m_Φ[j,k] - sum(next_mrkt.K[k,i]*m_U[i] for i=1:next_mrkt.nbX)- m_V[j])/(sum(next_mrkt.K[k,i] for i=1:next_mrkt.nbX)+1))
#     @NLexpression(model, m_μ[j=1:next_mrkt.nbY, k=1:next_mrkt.nbK], exp(m_log_μ[j,k]))
#
#     @NLexpression(model, m_μ_x0[i=1:next_mrkt.nbX], exp(-m_U[i]))
#     @NLexpression(model, m_μ_0y[j=1:next_mrkt.nbY], exp(-m_V[j]))
#
#     @NLconstraint(model, marg_x[i=1:next_mrkt.nbX], sum(next_mrkt.K[k,i]*m_μ[j,k] for j=1:next_mrkt.nbY, k=1:next_mrkt.nbK)+m_μ_x0[i] == next_mrkt.massX[i])
#     @NLconstraint(model, marg_y[j=1:next_mrkt.nbY], sum(m_μ[j,k] for k=1:next_mrkt.nbK)+m_μ_0y[j] == next_mrkt.massY[j])
#
#     @NLexpression(model, m_L1, sum((sum(next_mrkt.K[k,i] for i=1:next_mrkt.nbX)+1)*next_obs_μ[j,k]*m_log_μ[j,k] for j=1:next_mrkt.nbY, k=1:next_mrkt.nbK)+sum(log(m_μ_x0[i])*next_obs_μ_x0[i] for i=1:next_mrkt.nbX)+sum(log(m_μ_0y[j])*next_obs_μ_0y[j] for j=1:next_mrkt.nbY))
#
#     @NLobjective(model, Max, m_L1)
#     JuMP.optimize!(model)
#
#     return termination_status(model), value.(m_propL), objective_value(model), value.(m_μ)
# end
#
# function step_next_t(next_mrkt, params, propL0, nbL, next_obs_μ, next_obs_μ_x0, next_obs_μ_0y, maxiterML = 1e3, tolML = 1e-6, stepML = 1e-1)
#
#     propL_t = propL0
#
#     cont = true
#     iter = 0
#
#     max_error = 0.0
#
#     while cont
#
#         iter = iter+1
#
#         next_α_t, next_γ_t = generate_surplus(next_mrkt, params, nbL, propL_t)
#         term_t, next_μ_t, next_μ_x0_t, next_μ_0y_t, next_U_t, next_V_t = IPFP_knownhet(next_mrkt, nbL, propL_t, next_α_t, next_γ_t)
#
#         likelihood = sum((sum(next_mrkt.K[k,i] for i=1:next_mrkt.nbX)+1)*next_obs_μ[j,k]*log(next_μ_t[j,k]) for j=1:next_mrkt.nbY, k=1:next_mrkt.nbK)+sum(log(next_μ_x0_t[i])*next_obs_μ_x0[i] for i=1:next_mrkt.nbX)+sum(log(next_μ_0y_t[j])*next_obs_μ_0y[j] for j=1:next_mrkt.nbY)
#
#         propLnext = propL_t + stepML*[sum(∂γ_∂propL(next_mrkt, params, propL_t, nbL, j, k, i, l)*(next_obs_μ[j,k]-next_μ_t[j,k]) for k=1:next_mrkt.nbK, j=1:next_mrkt.nbY) for l=1:nbL, i=1:next_mrkt.nbX]
#         propLnext[nbL,:] = 1 .- propLnext[1:(nbL-1),:]
#
#         if iter>=maxiterML
#             cont = false
#             println("Max number of iterations reached")
#         end
#
#         max_error = findmax(abs.(propLnext[1:(nbL-1),:]-propL_t[1:(nbL-1),:]))[1]
#         # println("Max error is ", max_error)
#         if mod(iter, 100) == 0
#             println("Inner likelihood is ", likelihood)
#         end
#
#         if findmax(max_error)[1] <= tolML
#             cont = false
#             println("Gradient descent converged")
#         end
#
#         propL_t = propLnext
#
#     end
#
#     return max_error, propL_t
#
# end
