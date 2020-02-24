module Pipelines

using DataFrames
using Random

using AutoMLPipeline.AbsTypes: Machine, Transformer, Learner, Workflow, Computer, fit_transform!
using AutoMLPipeline.BaseFilters
using AutoMLPipeline.Utils

import AutoMLPipeline.AbsTypes: fit!, transform!
export fit!, transform!
export LinearPipeline, ComboPipeline

mutable struct LinearPipeline <: Workflow
  name::String
  model::Dict
  args::Dict

  function LinearPipeline(args::Dict = Dict())
    default_args = Dict(
			:name => "linearpipeline",
			# machines as list to chain in sequence.
			:machines => Vector{Machine}(),
			# Transformer args as list applied to same index transformer.
			:machine_args => Dict()
		       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function LinearPipeline(machs::Vector{T}) where {T<:Machine}
  LinearPipeline(Dict(:machines => machs))
end

function LinearPipeline(machs...)
  combo=nothing
  if eltype(machs) <: Machine
    v=[x for x in machs] # convert tuples to vector
    combo = LinearPipeline(v)
  else
    error("argument setup error")
  end
  return combo
end

function fit!(pipe::LinearPipeline, features::DataFrame, labels::Vector=[])
  instances=deepcopy(features)
  machines = pipe.args[:machines]
  machine_args = pipe.args[:machine_args]

  current_instances = instances
  new_machines = Machine[]

  # fit-transform all except last element
  # last element calls fit only
  trlength = length(machines)
  for t_index in 1:(trlength - 1)
    machine = createmachine(machines[t_index], machine_args)
    push!(new_machines, machine)
    fit!(machine, current_instances, labels)
    current_instances = transform!(machine, current_instances)
  end
  machine = createmachine(machines[trlength], machine_args)
  push!(new_machines, machine)
  fit!(machine, current_instances, labels)

  pipe.model = Dict(
      :machines => new_machines,
      :machine_args => machine_args
  )
end

function transform!(pipe::LinearPipeline, instances::DataFrame)
  machines = pipe.model[:machines]

  current_instances = deepcopy(instances)
  for t_index in 1:length(machines)
    machine = machines[t_index]
    current_instances = transform!(machine, current_instances)
  end

  return current_instances
end


mutable struct ComboPipeline <: Workflow
  name::String
  model::Dict
  args::Dict

  function ComboPipeline(args::Dict = Dict())
    default_args = Dict(
			:name => "combopipeline",
			# machines as list to chain in sequence.
			:machines => Vector{Machine}(),
			# Transformer args as list applied to same index transformer.
			:machine_args => Dict()
		       )
    cargs = nested_dict_merge(default_args, args)
    cargs[:name] = cargs[:name]*"_"*randstring(3)
    new(cargs[:name],Dict(),cargs)
  end
end

function ComboPipeline(machs::Vector{T}) where {T<:Machine}
  ComboPipeline(Dict(:machines => machs))
end

function ComboPipeline(machs...)
  combo=nothing
  if eltype(machs) <: Machine
    v=[x for x in machs] # convert tuples to vector
    combo = ComboPipeline(v)
  else
    error("argument setup error")
  end
  return combo
end


function fit!(pipe::ComboPipeline, features::DataFrame, labels::Vector=[])
  instances=deepcopy(features)
  machines = pipe.args[:machines]
  machine_args = pipe.args[:machine_args]

  new_machines = Machine[]
  new_instances = DataFrame()
  trlength = length(machines)
  for t_index in eachindex(machines)
    machine = createmachine(machines[t_index], machine_args)
    push!(new_machines, machine)
    fit!(machine, instances, labels)
  end

  pipe.model = Dict(
      :machines => new_machines,
      :machine_args => machine_args
  )
end

function transform!(pipe::ComboPipeline, features::DataFrame)
  machines = pipe.model[:machines]
  instances = deepcopy(features)
  new_instances = DataFrame()
  for t_index in eachindex(machines)
    machine = machines[t_index]
    current_instances = transform!(machine, instances)
    new_instances = hcat(new_instances,current_instances,makeunique=true)
  end

  return new_instances
end

function processexpr(args)
  for ndx in eachindex(args)
    if typeof(args[ndx]) == Expr
      processexpr(args[ndx].args)
    elseif args[ndx] == :+
      args[ndx] = :LinearPipeline
    elseif args[ndx] == :*
      args[ndx] = :ComboPipeline
    #else
    #  esc(:(args[ndx] = eval(args[ndx]))) # refer to local variable
    #  #args[ndx] = eval(:($(args[ndx])))
    #  #args[ndx] = eval(args[ndx])
    #  #println(args[ndx])
    end
  end
  return args
end

macro pipelinesetup(expr)
  res = processexpr(expr.args)
  expr.args = res
  #println(expr.args)
  expr
end

function pipeline_test()
  data = getiris()
  X=data[:,1:5]
  X[!,5]= X[!,5] .|> string
  ohe = OneHotEncoder()
  global ohe2 = OneHotEncoder()
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
  @assert (res1 .== res2) |> Matrix |> sum == 2100
  fit!(combo2,X)
  res3=transform!(combo2,X)
  res4=fit_transform!(combo2,X)
  @assert (res3 .== res3) |> Matrix |> sum == 2100
  pcombo1 = @pipelinesetup(ohe2 * ohe2)
  pres1 = fit_transform!(pcombo1,X)
  @assert (pres1 .== res1) |> Matrix |> sum == 2100
end

end
