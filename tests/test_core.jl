using PyCall
using Test
using BenchmarkTools

include("../src/core.jl")

@pyimport fgbuster

sky = fgbuster.observation_helpers.get_sky(nside = 64, tag = "c1d0s0")
instrument = fgbuster.observation_helpers.get_instrument("LiteBIRD")
obs = fgbuster.observation_helpers.get_observation(instrument, sky, noise=false)

comps = [cmb, sync, dust]
freqs = instrument.frequency.values
Nmat = hcat(instrument.depth_i.values, instrument.depth_p.values, instrument.depth_p.values)

@test KRJ_to_KCMB(10) == 1.0025855995398942
@test cmb(10) == 1
@test sync(10, 1) == 0.4961455328647282

A = mixing_matrix(comps, freqs)([1, 1, 1])
@test A == [
    1.0 2.0626334259332406 9.815453359661301;
    1.0 2.6382900231722526 9.114580182117537;
    1.0 3.2557642984584727 8.046440704408909;
    1.0 3.7866971136091427 7.0902990640710835;
    1.0 4.505968483464336 5.899485916816272;
    1.0 5.382467859245799 4.696059918561268;
    1.0 6.366491575126633 3.659955224215565;
    1.0 8.383971577110621 2.293010694115081;
    1.0 11.228267394204112 1.3159387837590166;
    1.0 16.005883887233743 0.6381082911100033;
    1.0 23.747118110854704 0.27643713362980327;
    1.0 41.20436169372608 0.08476489708722748;
    1.0 77.73165561122241 0.02198001194636403;
    1.0 177.80436777058557 0.003924654059052397;
    1.0 469.79084931558725 0.0005464803027606633
]

LᵀA = 𝔣LᵀA(Nmat, A)
res = 𝔣s(LᵀA, obs)

# listing full matrix is too long, just test aggregated result here
@test sum(res) == -2.4736592014096767e8
res = compsep(comps, freqs, Nmat, obs, x₀ = [-3, 1.54, 20.0])

@test isapprox(res["params"], [-3, 1.54, 20.0], rtol=0.01)

# performance testing

# @btime 𝔣s($LᵀA, $obs);
# 4.243 ms (14 allocations: 6.75 MiB)

# @btime compsep($comps, $freqs, $Nmat, $obs, x₀=[-3,1.54,20.]);
# 369.915 ms (10274 allocations: 25.51 MiB)
