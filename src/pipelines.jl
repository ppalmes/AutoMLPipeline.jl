module Pipelines

export fit!, transform!
export LinearPipeline

using AutoMLPipeline.AbsTypes: fit!, transform!
using AutoMLPipeline.AbsTypes: Machine, Transformer, Learner, Workflow, Computer
using AutoMLPipeline.Utils: nested_dict_merge

using Random

mutable struct LinearPipeline <: Workflow
  name::String
  model::Vector{Computer}
  args::Dict

  function LinearPipeline(args::Dict = Dict())
    default_args = Dict(
			:name => "linearpipeline",
			# machines as list to chain in sequence.
			:machines => Vector{Computer}(),
			# Transformer args as list applied to same index transformer.
			:machine_args => Dict()
		       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Vector{Computer}(),cargs)
  end
end


mutable struct ComboPipeline <: Workflow
  name::String
  model::Vector{LinearPipeline}
  args::Dict

  function ComboPipeline(args::Dict = Dict())
    default_args = Dict(
			:name => "combopipeline",
			# machines as list to chain in sequence.
			:machines => Vector{LinearPipeline}(),
			# Transformer args as list applied to same index transformer.
			:machine_args => Dict()
		       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Vector{LinearPipeline}(),cargs)
  end
end


function pipeline_test()
  linear = LinearPipeline(Dict(:name=>"lp"))
  combo = ComboPipeline()
  println(linear)
  println(combo)
end

end
