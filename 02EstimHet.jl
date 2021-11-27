using LinearAlgebra
using Random, Distributions
using DelimitedFiles
using JuMP, Ipopt

path = dirname(pwd())

include(path*"00DataPrepHet.jl")
include(path*"01FunctionsPrepHet.jl")

seed = 888
nbX = 3
nbY = 2
nbL = 2
massX = 2*ones(nbX)
next_massX = collect(1:nbX)
massY = 1*ones(nbY)
# true_mass_propL = rand(nbL, nbX)
# true_mass_propL = true_mass_propL./sum(true_mass_propL, dims =1)
# true_mass_propL = collect(Iterators.flatten(true_mass_propL))
cuts = 5

propL = [.25 .75; .5 .5; .75 .25]
true_massXL = kron(massX, ones(nbL))/nbL

mrkt = generate_data(nbX, nbY, nbL, massX, massY, cuts, propL)
next_mrkt = generate_data(nbX, nbY, nbL, next_massX, massY, cuts, propL)

true_params = generate_params(seed, nbX, nbY, nbL)
true_α, true_γ = generate_surplus(mrkt, true_params, nbL)

term, obs_μ, obs_μ_x0, obs_μ_0y, true_U, true_V = IPFP_knownhet(mrkt, true_α, true_γ, true_massXL)
obs_total_wage = [log(mrkt.massY[j])+true_γ[j,k]-true_V[j]-log(obs_μ[j,k]) for j=1:mrkt.nbY, k=1:mrkt.nbK]

next_term, next_obs_μ, next_obs_μ_x0, next_obs_μ_0y, next_obs_U, next_obs_V = IPFP_knownhet(mrkt, true_α, true_γ, true_massXL)

true_Φ = [true_γ[j,k] + sum(sum(mrkt.K[k,(i-1)*nbL+l] for l=1:nbL)*true_α[i,j] for i=1:mrkt.nbX) for j=1:mrkt.nbY, k=1:mrkt.nbK]

term, est_massXL = min_wage_distance_MPEC(mrkt, true_params, obs_total_wage)
term, est_params = max_log_likelihood_MPEC(mrkt, true_massXL, obs_μ, obs_μ_x0, obs_μ_0y)

# Using wage
params_t = generate_params(777, nbX, nbY, nbL)
# propL_t = rand(nbL)
# propL_t = propL_t./sum(propL_t, dims =1)
# est_massXL_t = kron(massX, propL_t)

maxiter = 1e2
tol = 1e-6
cont = true
iter = 0

obj = 0.0
massXL_next = []

while cont

    iter = iter+1

    term, massXL_next = min_wage_distance_MPEC(mrkt, params_t, obs_total_wage)
    term, params_next, obj_next = max_log_likelihood_MPEC(mrkt, massXL_next, obs_μ, obs_μ_x0, obs_μ_0y)

    if iter>=maxiter
            cont = false
            println("Max number of iterations reached")
    end

    max_error = findmax(abs.(obj_next-obj))[1]
    println("Max error is ", max_error)
    println("Likelihood is ", obj_next)

    if max_error <= tol
        cont = false
        println("Gradient descent converged")
    end

    params_t = params_next
    obj = obj_next

end

# Using next matching
params_t = generate_params(777, nbX, nbY, nbL)
# propL_t = rand(nbL)
# propL_t = propL_t./sum(propL_t, dims =1)
# est_massXL_t = kron(massX, propL_t)

maxiter = 1e2
tol = 1e-6
cont = true
iter = 0

obj2 = 0.0
massXL_next = []

while cont

    iter = iter+1

    term, massXL_next, obj1_next = next_max_log_likelihood_MPEC(next_mrkt, params_t, next_obs_μ, next_obs_μ_x0, next_obs_μ_0y)
    term, params_next, obj2_next = max_log_likelihood_MPEC(mrkt, massXL_next, obs_μ, obs_μ_x0, obs_μ_0y)

    if iter>=maxiter
        cont = false
        println("Max number of iterations reached")
    end

    max_error = findmax(abs.(obj2_next-obj2))[1]
    println("Max error is ", max_error)
    println("Likelihood is ", obj_next)

    if max_error <= tol
        cont = false
        println("Gradient descent converged")
    end

    params_t = params_next
    obj2 = obj2_next

end

est_α, est_γ = generate_surplus(mrkt, params_t, nbL)


# next_true_Φ = [true_γ[j,k] + sum(next_mrkt.K[k,i]*true_α[i,j] for i=1:next_mrkt.nbX) for j=1:next_mrkt.nbY, k=1:next_mrkt.nbK]

est_α, est_γ = generate_surplus(mrkt, est_params, nbL)

est_nbL = 2

ϵ = 1e-3
params0 = SurplusParams(true_params.λ.+ϵ, true_params.δ.+ϵ, true_params.τ.+ϵ, true_params.ρ.+ϵ, true_params.ξ.+ϵ, true_params.β)
max_error, params_t = step_t(mrkt, params0, propL, nbL, obs_μ, obs_μ_x0, obs_μ_0y, true_params.β, 2e3, 1e-7, 10)

propL0 = propL .+ [ϵ; -ϵ]
max_error, propL_t = step_next_t(next_mrkt, true_params, propL0, nbL, next_obs_μ, next_obs_μ_x0, next_obs_μ_0y, 2e3, 1e-6, 1e-2)

term, propL_t, obj, μ_t = step_next_t_MPEC(next_mrkt, true_params, nbL, next_obs_μ, next_obs_μ_x0, next_obs_μ_0y)



model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>3))

@variable(model, m_log_λ[j=1:obs_mrkt.nbY], start = log(true_params.λ[j]))
@variable(model, m_log_δ[j=1:obs_mrkt.nbY], start = log(true_params.δ[j]))
@variable(model, m_log_ρ, start = log(-true_params.ρ))
@variable(model, m_log_τ, start = log(-true_params.τ))

@variable(model, m_log_ξ[l=1:est_nbL, i=1:obs_mrkt.nbX], start = log(true_params.ξ[l,i]))

@variable(model, m_β[i=1:obs_mrkt.nbX, j=1:obs_mrkt.nbY], start = log(true_params.β[i,j]))

@NLexpression(model, m_λ[j=1:obs_mrkt.nbY], exp(m_log_λ[j]))
@NLexpression(model, m_δ[j=1:obs_mrkt.nbY], exp(m_log_δ[j]))
@NLexpression(model, m_ρ, -exp(m_log_ρ))
@NLexpression(model, m_τ, -exp(m_log_τ))

@NLexpression(model, m_ξ[l=1:est_nbL, i=1:obs_mrkt.nbX], exp(m_log_ξ[l,i]))
@NLconstraint(model, consξ[i=1:obs_mrkt.nbX], sum(m_ξ[l,i] for l=1:est_nbL) == 1.0)

# @variable(model, m_propL[l=1:est_nbL, i=1:obs_mrkt.nbX] >= 1e-8)
# @constraint(model, consL[i=1:obs_mrkt.nbX], sum(m_propL[l,i] for l=1:est_nbL) == 1.0)

# @constraint(model, true_propL[l=1:est_nbL, i=1:obs_mrkt.nbX], m_propL[l,i] == mrkt.propL[l,i])

@NLexpression(model, m_α[i=1:obs_mrkt.nbX, j=1:obs_mrkt.nbY],  m_β[i,j])

# @NLexpression(model, m_log_intγ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK],  (1/m_τ)*log(m_λ[j]*sum(m_ξ[l,2]*m_propL[l,2]*obs_mrkt.K[k,2] for l=1:est_nbL)^m_τ + (1-m_λ[j])*sum(m_ξ[l,3]*m_propL[l,3]*obs_mrkt.K[k,3] for l=1:est_nbL)^m_τ))
@NLexpression(model, m_log_intγ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK],  (1/m_τ)*log(m_λ[j]*sum(m_ξ[l,2]*mrkt.propL[l,2]*obs_mrkt.K[k,2] for l=1:est_nbL)^m_τ + (1-m_λ[j])*sum(m_ξ[l,3]*mrkt.propL[l,3]*obs_mrkt.K[k,3] for l=1:est_nbL)^m_τ))

@NLexpression(model, m_intγ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], exp(m_log_intγ[j,k]))

# @NLexpression(model, m_log_γ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], (1/m_ρ)*log(m_δ[j]*sum(m_ξ[l,1]*m_propL[l,1]*obs_mrkt.K[k,1] for l=1:est_nbL)^m_ρ+ (1-m_δ[j])*m_intγ[j,k]^m_ρ))
@NLexpression(model, m_log_γ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], (1/m_ρ)*log(m_δ[j]*sum(m_ξ[l,1]*mrkt.propL[l,1]*obs_mrkt.K[k,1] for l=1:est_nbL)^m_ρ+ (1-m_δ[j])*m_intγ[j,k]^m_ρ))
@NLexpression(model, m_γ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], exp(m_log_γ[j,k]))

@NLexpression(model, m_Φ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], m_γ[j,k] + sum(obs_mrkt.K[k,i]*m_α[i,j] for i=1:obs_mrkt.nbX))

# @variable(model, m_U[l=1:est_nbL,i=1:obs_mrkt.nbX])
@variable(model, m_U[i=1:obs_mrkt.nbX])
@variable(model, m_V[j=1:obs_mrkt.nbY])

# @NLconstraint(model, normΦ[k=1:3], m_Φ[1,k] == true_Φ[1,k])
# @NLconstraint(model, normΦ, m_Φ[1,mrkt.nbK] == true_Φ[1,mrkt.nbK])

# @NLconstraint(model, normV, m_V[1] == obs_V[1])

@constraint(model, normβ[i=1:obs_mrkt.nbX], m_β[i,1] == true_params.β[i,1])

# @NLexpression(model, m_log_μ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], (m_Φ[j,k] - sum(m_propL[l,i]*obs_mrkt.K[k,i]*m_U[l,i] for i=1:obs_mrkt.nbX, l=1:est_nbL)- m_V[j])/(sum(obs_mrkt.K[k,i] for i=1:obs_mrkt.nbX)+1))
@NLexpression(model, m_log_μ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], (m_Φ[j,k] - sum(obs_mrkt.K[k,i]*m_U[i] for i=1:obs_mrkt.nbX)- m_V[j])/(sum(obs_mrkt.K[k,i] for i=1:obs_mrkt.nbX)+1))
@NLexpression(model, m_μ[j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK], exp(m_log_μ[j,k]))

@NLexpression(model, m_μ_x0[i=1:mrkt.nbX], exp(-m_U[i]))
@NLexpression(model, m_μ_0y[j=1:mrkt.nbY], exp(-m_V[j]))

# @NLconstraint(model, marg_x[l=1:est_nbL, i=1:obs_mrkt.nbX], sum(m_propL[l,i]*obs_mrkt.K[k,i]*m_μ[j,k] for j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK) == m_propL[l,i]*obs_mrkt.massX[i])
@NLconstraint(model, marg_x[i=1:obs_mrkt.nbX], sum(obs_mrkt.K[k,i]*m_μ[j,k] for j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK)+m_μ_x0[i] == obs_mrkt.massX[i])
@NLconstraint(model, marg_y[j=1:obs_mrkt.nbY], sum(m_μ[j,k] for k=1:obs_mrkt.nbK)+m_μ_0y[j] == obs_mrkt.massY[j])

@NLexpression(model, m_L1, sum((sum(obs_mrkt.K[k,i] for i=1:obs_mrkt.nbX)+1)*obs_μ[j,k]*m_log_μ[j,k] for j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK)+sum(log(m_μ_x0[i])*obs_μ_x0[i] for i=1:obs_mrkt.nbX)+sum(log(m_μ_0y[j])*obs_μ_0y[j] for j=1:obs_mrkt.nbY))

@NLobjective(model, Max, m_L1)
JuMP.optimize!(model)


findmax(abs.(value.(m_μ)-obs_μ))

int_obs_μ = [(true_Φ[j,k] - sum(mrkt.K[k,(i-1)*nbL+l]*obs_U[l,i] for l=1:mrkt.nbL,  i=1:obs_mrkt.nbX)- obs_V[j])/(sum(mrkt.K[k,(i-1)*nbL+l] for l=1:mrkt.nbL, i=1:mrkt.nbX)+1) for j=1:mrkt.nbY, k=1:mrkt.nbK]
int_est_μ = [ (value(m_Φ[j,k]) - sum(obs_mrkt.K[k,i]*value(m_U[i]) for i=1:obs_mrkt.nbX)- value(m_V[j]))/(sum(obs_mrkt.K[k,i] for i=1:obs_mrkt.nbX)+1) for j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK]
findmax(abs.(int_obs_μ-int_est_μ))

[sum(obs_mrkt.K[k,i]*value(m_U[i]) for i=1:obs_mrkt.nbX) for k=1:mrkt.nbK]
[sum(mrkt.K[k,(i-1)*nbL+l]*obs_U[l,i] for i=1:mrkt.nbX, l=1:mrkt.nbL) for k=1:mrkt.nbK]

tx = transpose(sum(mrkt.propL.*obs_U, dims=1))-value.(m_U)
ty = obs_V - value.(m_V)

[sum(obs_mrkt.K[k,i]*tx[i] for i=1:obs_mrkt.nbX)+ty[j] for j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK]
diffΦ = true_Φ-value.(m_Φ)
[sum(obs_mrkt.K[k,i]*tx[i] for i=1:obs_mrkt.nbX)+ty[j] for j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK]-(true_Φ-value.(m_Φ))

tk = [sum(mrkt.K[k,(i-1)*mrkt.nbL+l]*obs_U[l,i] for l=1:mrkt.nbL, i=1:mrkt.nbX)-sum(obs_mrkt.K[k,i]*value(m_U[i]) for i=1:obs_mrkt.nbX) for k=1:obs_mrkt.nbK]
[tk[k]+ty[j] for j=1:obs_mrkt.nbY, k=1:obs_mrkt.nbK]-diffΦ
