#include "arm_compute/graph/detail/ExecutionHelpers.h"

#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/GraphContext.h"
#include "arm_compute/graph/GraphManager.h"
#include "arm_compute/graph/Tensor.h"
#include "arm_compute/graph/backends/BackendRegistry.h"
#include <fstream>

namespace arm_compute
{
namespace graph
{
namespace detail
{
void validate_all_nodes(Graph &g)
{
    auto &nodes = g.nodes();

    for(auto &node : nodes)
    {
        if(node != nullptr)
        {
            Target                    assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            Status                    status          = backend.validate_node(*node);
            ARM_COMPUTE_ERROR_ON_MSG(!bool(status), status.error_description().c_str());
        }
    }
}

void configure_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && tensor->handle() == nullptr)
        {
            Target                         target  = tensor->desc().target;
            backends::IDeviceBackend      &backend = backends::BackendRegistry::get().get_backend(target);
            std::unique_ptr<ITensorHandle> handle  = backend.create_tensor(*tensor);
            ARM_COMPUTE_ERROR_ON_MSG(!handle, "Couldn't create backend handle!");
            tensor->set_handle(std::move(handle));
        }
    }
}

void allocate_all_input_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_inputs(); ++i)
    {
        Tensor *tensor = node.input(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_all_output_tensors(INode &node)
{
    for(unsigned int i = 0; i < node.num_outputs(); ++i)
    {
        Tensor *tensor = node.output(i);
        if(tensor != nullptr && !tensor->bound_edges().empty())
        {
            ARM_COMPUTE_ERROR_ON_MSG(!tensor->handle(), "Tensor handle is not configured!");
            tensor->handle()->allocate();
        }
    }
}

void allocate_const_tensors(Graph &g)
{
    for(auto &node : g.nodes())
    {
        if(node != nullptr)
        {
            switch(node->type())
            {
                case NodeType::Const:
                case NodeType::Input:
                    allocate_all_output_tensors(*node);
                    break;
                case NodeType::Output:
                    allocate_all_input_tensors(*node);
                default:
                    break;
            }
        }
    }
}

void allocate_all_tensors(Graph &g)
{
    auto &tensors = g.tensors();

    for(auto &tensor : tensors)
    {
        if(tensor && !tensor->bound_edges().empty() && tensor->handle() != nullptr && tensor->handle()->tensor().info()->is_resizable() && tensor->handle()->tensor().is_used())
        {
            tensor->handle()->allocate();
        }
    }
}

ExecutionWorkload configure_all_nodes(Graph &g, GraphContext &ctx, const std::vector<NodeID> &node_order)
{
    ExecutionWorkload workload;
    workload.graph = &g;
    workload.ctx   = &ctx;

    workload.tasks.reserve(node_order.size());

    for(auto &node_id : node_order)
    {
        auto node = g.node(node_id);
        if(node != nullptr)
        {
            Target                     assigned_target = node->assigned_target();
            backends::IDeviceBackend &backend         = backends::BackendRegistry::get().get_backend(assigned_target);
            std::unique_ptr<IFunction> func            = backend.configure_node(*node, ctx);
            if(func != nullptr)
            {
                workload.tasks.emplace_back(ExecutionTask(std::move(func), node));
            }
        }
    }

    for(auto &node : g.nodes())
    {
        if(node != nullptr && node->type() == NodeType::Input)
        {
            workload.inputs.push_back(node->output(0));
        }

        if(node != nullptr && node->type() == NodeType::Output)
        {
            workload.outputs.push_back(node->input(0));
            continue;
        }
    }

    return workload;
}

void release_unused_tensors(Graph &g)
{
    for(auto &tensor : g.tensors())
    {
        if(tensor != nullptr && tensor->handle() != nullptr)
        {
            tensor->handle()->release_if_unused();
        }
    }
}

void call_tensor_accessor(Tensor *tensor)
{
    ARM_COMPUTE_ERROR_ON(!tensor);
    tensor->call_accessor();
}

void call_all_const_node_accessors(Graph &g)
{
    auto &nodes = g.nodes();

    for(auto &node : nodes)
    {
        if(node != nullptr && node->type() == NodeType::Const)
        {
            call_tensor_accessor(node->output(0));
        }
    }
}

bool call_all_input_node_accessors(ExecutionWorkload &workload)
{
    bool is_valid = true;
    std::for_each(std::begin(workload.inputs), std::end(workload.inputs), [&](Tensor * input_tensor)
    {
        bool valid_input = (input_tensor != nullptr) && input_tensor->call_accessor();
        is_valid         = is_valid && valid_input;
    });
    return is_valid;
}

void prepare_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.graph == nullptr);
    for(auto &task : workload.tasks)
    {
        task.prepare();
        release_unused_tensors(*workload.graph);
    }
}

void call_all_tasks(ExecutionWorkload &workload)
{
    ARM_COMPUTE_ERROR_ON(workload.ctx == nullptr);

    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->acquire();
        }
    }
    std::ofstream out("./alexnet_graph.csv", std::ios::out | std::ios::app);

    std::string  name[5]={"conv", "fc", "norm", "pool", "relu"};
    double layer_time[6]={0.f};
    for(auto &task : workload.tasks)
    {
        std::string node_name = task.node->name();
        auto tbegin=std::chrono::high_resolution_clock::now();
        task();
        auto tend=std::chrono::high_resolution_clock::now();
        double time=std::chrono::duration_cast<std::chrono::duration<double>>(tend-tbegin).count();
        // std::cout<<node_name<<"           "<<time*1000<<"ms"<<std::endl;
        const char *show;
        unsigned int index=0;
        for(; index<5; index++)
        {
            show = strstr(node_name.c_str(), name[index].c_str());
            if(show!=NULL){
                layer_time[index]+=time;
                break;
            }
        }
        if(index==5){
            layer_time[5]+=time;
        }
    }
    // std::cout<<"conv layers: "<<layer_time[0]*1000<<"ms"<<std::endl;
    // std::cout<<"act  layers: "<<layer_time[4]*1000<<"ms"<<std::endl;
    // std::cout<<"pool layers: "<<layer_time[3]*1000<<"ms"<<std::endl;
    // std::cout<<"norm layers: "<<layer_time[2]*1000<<"ms"<<std::endl;
    // std::cout<<"fc   layers: "<<layer_time[1]*1000<<"ms"<<std::endl;
    // std::cout<<"other layers: "<<layer_time[5]*1000<<"ms"<<std::endl;
    out<<layer_time[0]*1000<<","<<layer_time[1]*1000<<","<<layer_time[2]*1000<<","<<layer_time[3]*1000<<","<<layer_time[4]*1000<<","<<layer_time[5]*1000<<std::endl;

    for(auto &mm_ctx : workload.ctx->memory_managers())
    {
        if(mm_ctx.second.cross_group != nullptr)
        {
            mm_ctx.second.cross_group->release();
        }
    }
}

bool call_all_output_node_accessors(ExecutionWorkload &workload)
{
    bool is_valid = true;
    std::for_each(std::begin(workload.outputs), std::end(workload.outputs), [&](Tensor * output_tensor)
    {
        bool valid_output = (output_tensor != nullptr) && output_tensor->call_accessor();
        is_valid          = is_valid && valid_output;
    });

    return is_valid;
}
}
}
} 
