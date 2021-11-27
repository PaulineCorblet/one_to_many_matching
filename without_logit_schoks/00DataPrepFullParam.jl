struct Market
    nbX::Int
    nbY::Int
    massX::Vector{Float64}
    massY::Vector{Float64}
    nbK::Int
    K::Array{Float64,2}
end

struct SurplusParams
    λ::Array{Float64,2}
    ρ::Array{Float64,1}
    β::Array{Float64,3}
end


function generate_data(seed, nbX, nbY, massX, massY, nbCoeff, cuts)

    Random.seed!(seed)

    nbK = cuts^nbX

    grid = Array{Float64,2}(undef, cuts, nbX)
    for i=1:nbX
        grid[:, i] = LinRange(1e-3, massX[i], cuts)
    end

    grid_combtup = reshape(collect(Iterators.product([grid[:,i] for i=1:nbX]...)), (cuts^nbX,1))

    K_grid = Array{Float64,2}(undef, nbK, nbX)
    for k=1:cuts^nbX
        K_grid[k,:] = collect(grid_combtup[k,:][1])
    end

    λ = rand(nbX, nbY)
    λ = λ./sum(λ, dims =1)
    ρ = rand(-10:.1:1, nbY)

    β = rand(nbX, nbY, nbCoeff)

    return Market(nbX, nbY, massX, massY, nbK, K_grid), SurplusParams(λ, ρ, β)

end

function generate_surplus(mrkt, params)

    nbCoeff = size(params.β)[3]

    α = Array{Float64,3}(undef, mrkt.nbX, mrkt.nbY, mrkt.nbK)
    γ = Array{Float64,2}(undef, mrkt.nbY, mrkt.nbK)

    Σk = sum(mrkt.K, dims = 2)
    [α[i,j,k] = sum(params.β[i,j,x]*Σk[k]^(x-1) for x=1:nbCoeff) for i=1:mrkt.nbX, j=1:mrkt.nbY, k=1:mrkt.nbK]
    [γ[j,k] = sum(params.λ[i,j]*(mrkt.K[k,i] .^ params.ρ[j]) for i=1:mrkt.nbX)^(1/params.ρ[j]) for j=1:mrkt.nbY, k=1:mrkt.nbK]

    return α, γ
end
