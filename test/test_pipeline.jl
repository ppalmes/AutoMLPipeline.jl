module TestPipeline

using Test
using AutoMLPipeline
using AutoMLPipeline.Pipelines
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Utils
 

function test_pipeline()
  data = getiris()
  X=data[:,1:5]
  X[!,5]= X[!,5] .|> string
  ohe = OneHotEncoder()
  linear1 = LinearPipeline(Dict(:name=>"lp",:machines => [ohe]))
  linear2 = LinearPipeline(Dict(:name=>"lp",:machines => [ohe]))
  combo1 = ComboPipeline(Dict(:machines=>[ohe,ohe]))
  combo2 = ComboPipeline(Dict(:machines=>[linear1,linear2]))
  linear1 = LinearPipeline([ohe])
  linear2 = LinearPipeline([ohe])
  combo1 = ComboPipeline([ohe,ohe])
  combo2 = ComboPipeline([linear1,linear2])
  fit!(combo1,X)
  res1=transform!(combo1,X)
  res2=fit_transform!(combo1,X)
  @test (res1 .== res2) |> Matrix |> sum == 2100
  fit!(combo2,X)
  res3=transform!(combo2,X)
  res4=fit_transform!(combo2,X)
  @test (res3 .== res3) |> Matrix |> sum == 2100
  #global ohe2 = OneHotEncoder()
  #pcombo1 = @pipelinesetup ohe2 * ohe2
  #pres1 = fit_transform!(pcombo1,X)
  #@test (pres1 .== res1) |> Matrix |> sum == 2100
end
@testset "Pipelines" begin
    test_pipeline()
end


end
