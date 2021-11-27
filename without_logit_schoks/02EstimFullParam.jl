using LinearAlgebra
using Random, Distributions
using DelimitedFiles
using JuMP, Ipopt

path = dirname(pwd())

include(path*"00DataPrepFullParam.jl")
include(path*"01FunctionPrepFullParam.jl")

seed = 888
nbX = 3
nbY = 2
massX = 2*ones(nbX)
massY = 1*ones(nbY)
cuts = 10
nbCoeff = 2
N_obs = 1e6

mrkt, true_params = generate_data(seed, nbX, nbY, massX, massY, nbCoeff, cuts)

# No wage distribution
term, μ_obs, W_obs, U_obs, V_obs = simulate_matching_wage(mrkt, true_params, false, N_obs)
LL_term, est_params, est_s, est_L1, est_L2  =  max_log_likelihood(mrkt, μ_obs, W_obs, nbCoeff, false)

findmax(abs.(est_params.λ-true_params.λ))[1]
findmax(abs.(est_params.ρ-true_params.ρ))[1]
findmax(abs.(est_params.β-true_params.β))[1]

# Wage distribution
term, μ_obs, W_obs, U_obs, V_obs = simulate_matching_wage(mrkt, true_params, true, N_obs)
LL_term, est_params, est_s =  max_log_likelihood(mrkt, μ_obs, W_obs, nbCoeff, true)

findmax(abs.(est_params.λ-true_params.λ))[1]
findmax(abs.(est_params.ρ-true_params.ρ))[1]
findmax(abs.(est_params.β-true_params.β))[1]

# Note: surplus parameters are not exactly right, although estimated wage (and α, γ) is almost the same as true wage.
# Possibly comes from simulation of error

# true_α, true_γ = generate_surplus(mrkt, true_params)
# est_α, est_γ = generate_surplus(mrkt, est_params)
#
# W_true = simulate_matching_wage(mrkt, true_params, false, N_obs)[3]
#
# Σk = sum(mrkt.K, dims = 2)
# true_L1 = sum((Σk[k]+1)*μ_obs[j,k]*log(μ_obs[j,k]) for j=1:mrkt.nbY, k=1:mrkt.nbK)
# true_L2 = -.5*sum(sum((Σk[k]+1)*(W_obs[j,k][l]-W_true[j,k])^2/1.0+(Σk[k]+1)*log(1.0) for l=1:length(W_obs[j,k]))  for j=1:mrkt.nbY, k=1:mrkt.nbK)
#
# (1/(sum(mrkt.K[1,i] for i=1:mrkt.nbX)+1))*(sum(mrkt.K[1,i]*(est_γ[1,1]-est_α[i,1,1]+est_U[i]-est_V[1]) for i=1:mrkt.nbX))
