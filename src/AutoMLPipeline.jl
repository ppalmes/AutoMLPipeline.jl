module AutoMLPipeline

greet() = print("Hello World!")

include("abstracttypes.jl")
using .AbsTypes

include("utils.jl")
using .Utils

include("pipelines.jl")
using .Pipelines

end # module
