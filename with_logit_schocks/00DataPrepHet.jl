struct Market
    nbX::Int
    nbY::Int
    massX::Vector{Float64}
    massY::Vector{Float64}
    nbL::Int
    nbK::Int
    K::Matrix{Float64}
end

struct HetDistrib
    mpL::Vector{Float64}
    K::Matrix{Float64}
end

struct ObsMarket
    nbX::Int
    nbY::Int
    massX::Vector{Float64}
    massY::Vector{Float64}
    nbK::Int
    K::Matrix{Float64}
end

struct SurplusParams
    λ::Vector{Float64}
    δ::Vector{Float64}
    τ::Float64
    ρ::Float64
    ξ::Matrix{Float64}
    β::Matrix{Float64}
end


function generate_data(nbX, nbY, nbL, massX, massY, cuts, propL)

    # massXL = kron(massX, ones(nbL))
    # nbK = cuts^(nbX*nbL)
    #
    # grid = Array{Float64,2}(undef, cuts, nbX*nbL)
    # for il=1:(nbX*nbL)
    #     grid[:, il] = LinRange(1e-3, massXL[il], cuts)
    # end
    #
    # grid_combtup = reshape(collect(Iterators.product([grid[:,il] for il=1:(nbX*nbL)]...)), nbK,1)
    #
    # K_grid = Array{Float64,2}(undef, nbK, (nbX*nbL))
    # for k=1:nbK
    #     K_grid[k,:] = collect(grid_combtup[k,:][1])
    # end

    nbK_red = cuts^(nbX)
    nbK = cuts^(nbX*nbL)

    grid = Array{Float64,2}(undef, cuts, nbX)
    for i=1:nbX
        grid[:, i] = LinRange(1e-3, massX[i], cuts)
    end

    grid_kron = kron(grid, propL)

    max_k = cuts*size(propL)[1]
    K_grid = Matrix{Float64}(undef, max_k^nbX, nbX*nbL)

    iter = 0
    #Warning: works only if nbX = 3
    for k1=1:max_k
        for k2=1:max_k
            for k3=1:max_k
                iter = iter+1
                K_grid[iter,:] = vcat(grid_kron[k1, 1:nbL], grid_kron[k2, nbL+1:2*nbL], grid_kron[k3, 2*nbL+1:3*nbL])
            end
        end
    end

    return Market(nbX, nbY, massX, massY, nbL, max_k^nbX, K_grid)

end

function generate_params(seed, nbX, nbY, nbL)

    Random.seed!(seed)

    λ = rand(nbY)
    δ = rand(nbY)
    ρ = rand(-10:.1:1)
    τ = rand(-10:.1:1)
    ξ = rand(nbL, nbX)
    ξ = ξ./sum(ξ, dims =1)

    β = zeros(nbX, nbY)

    return  SurplusParams(λ, δ, τ, ρ, ξ, β)

end

function generate_hetdistrib(mrkt, mass_propL)

    massXL = kron(mrkt.massX, ones(mrkt.nbL))

    threshold = mass_propL.*massXL
    K_grid_red = transpose(mrkt.K[1,:])

    for k=2:mrkt.nbK
        if size(findall(mrkt.K[k,:] .> threshold))[1] == 0
            K_grid_red = vcat(K_grid_red, transpose(mrkt.K[k,:]))
        end
    end

    return HetDistrib(mass_propL, K_grid_red)

end

function generate_surplus(mrkt, params, nbL)

    α = params.β

    pre_γ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)
    γ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)

    [pre_γ[j,k] = (params.λ[j]*sum(params.ξ[l,2]*mrkt.K[k,nbL+l] for l=1:nbL)^params.τ + (1-params.λ[j])*sum(params.ξ[l,1]*mrkt.K[k,l] for l=1:nbL)^params.τ)^(1/params.τ) for j=1:mrkt.nbY, k=1:mrkt.nbK]
    [γ[j,k] = (params.δ[j]*sum(params.ξ[l,3]*mrkt.K[k,2*nbL+l] for l=1:nbL)^params.ρ + (1-params.δ[j])*pre_γ[j,k]^params.ρ)^(1/params.ρ) for j=1:mrkt.nbY, k=1:mrkt.nbK]

    return α, γ
end
