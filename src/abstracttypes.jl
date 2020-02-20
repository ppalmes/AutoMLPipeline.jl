module AbsTypes

using DataFrames

export fit!, transform!
export Machine, Learner, Transformer, Workflow

abstract type Machine end
abstract type Computer <: Machine end # does computation: learner and transformer
abstract type Workflow <: Machine end # different pipeline types: Linear vs Combine
abstract type Learner <: Computer end
abstract type Transformer <: Computer end

# multiple disparch for fit
function fit!(mc::Machine, input::DataFrame, output::Vector)
	error(typeof(mc),"not implemented")
end

function transform!(mc::Machine, input::DataFrame)
	error(typeof(mc),"not implemented")
end

end

