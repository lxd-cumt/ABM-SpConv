#include "arm_compute/graph/GraphManager.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/Logger.h"
#include "arm_compute/graph/PassManager.h"
#include "arm_compute/graph/TypePrinter.h"
#include "arm_compute/graph/Utils.h"
#include "arm_compute/graph/detail/CrossLayerMemoryManagerHelpers.h"
#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "arm_compute/graph/algorithms/TopologicalSort.h"

#include <ctime>
#include <cstdlib>

namespace arm_compute
{
namespace graph
{
GraphManager::GraphManager()
    : _workloads()
{
}

void GraphManager::finalize_graph(Graph &graph, GraphContext &ctx, PassManager &pm, Target target)
{
    if(_workloads.find(graph.id()) != std::end(_workloads))
    {
        ARM_COMPUTE_ERROR("Graph is already registered!");
    }

    Target forced_target = target;
    if(!is_target_supported(target))
    {
        forced_target = get_default_target();
        ARM_COMPUTE_LOG_GRAPH_INFO("Switching target from " << target << " to " << forced_target << std::endl);
    }
    force_target_to_graph(graph, forced_target);

    setup_requested_backend_context(ctx, forced_target);

    detail::configure_all_tensors(graph);

    pm.run_all(graph);

    std::vector<NodeID> topological_sorted_nodes = dfs(graph);

    detail::validate_all_nodes(graph);

    auto workload = detail::configure_all_nodes(graph, ctx, topological_sorted_nodes);
    ARM_COMPUTE_ERROR_ON_MSG(workload.tasks.empty(), "Could not configure all nodes!");

    detail::allocate_const_tensors(graph);
    detail::call_all_const_node_accessors(graph);

    detail::prepare_all_tasks(workload);

    if(ctx.config().use_transition_memory_manager)
    {
        detail::configure_transition_manager(graph, ctx, workload);
    }
    else
    {
        detail::allocate_all_tensors(graph);
    }

    ctx.finalize();

    _workloads.insert(std::make_pair(graph.id(), std::move(workload)));
    ARM_COMPUTE_LOG_GRAPH_VERBOSE("Created workload for graph with ID : " << graph.id() << std::endl);
}

void GraphManager::execute_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");
    /*double t=0;
    unsigned int i=0;*/
    while(true)
    {
        // auto begin=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_input_node_accessors(it->second))
        {
            return;
        }
        // auto end=std::chrono::high_resolution_clock::now();
        // double time1=std::chrono::duration_cast<std::chrono::duration<double>>(end-begin).count();
        // std::cout<<"before call all tasks ="<<"           "<<time1*1000<<"ms"<<std::endl;

        // auto tbegin=std::chrono::high_resolution_clock::now();
        detail::call_all_tasks(it->second);
        // auto tend=std::chrono::high_resolution_clock::now();
        // double time=std::chrono::duration_cast<std::chrono::duration<double>>(tend-tbegin).count();
        // std::cout<<"-------------------------------------------------tasks ="<<"           "<<time*1000<<"ms"<<std::endl;
        
        // auto ttbegin=std::chrono::high_resolution_clock::now();
        if(!detail::call_all_output_node_accessors(it->second))
        {
            return;
        }
        // auto ttend=std::chrono::high_resolution_clock::now();
        // double time2=std::chrono::duration_cast<std::chrono::duration<double>>(ttend-ttbegin).count();
        // std::cout<<"after call all tasks ="<<"           "<<time2*1000<<"ms"<<std::endl;
        // std::cout<<std::endl;


    }
    
}

void GraphManager::invalidate_graph(Graph &graph)
{
    auto it = _workloads.find(graph.id());
    ARM_COMPUTE_ERROR_ON_MSG(it == std::end(_workloads), "Graph is not registered!");

    _workloads.erase(it);
}
} 
} 
