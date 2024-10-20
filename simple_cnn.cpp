#include <iostream>
#include <vector>
#include <dnnl.hpp>

using namespace dnnl;

class SimpleCNN {
public:
    // convolution layer
    memory::desc conv_input_md{
        {1, 1, 32, 32},
        memory::data_type::f32,
        memory::format_tag::nchw
    };

    memory::desc conv_weights_md{
        {10, 1, 5, 5},
        memory::data_type::f32,
        memory::format_tag::oihw
    };

    memory::desc conv_bias_md{
        {10},
        memory::data_type::f32,
        memory::format_tag::x
    };

    memory::desc conv_output_md{
        {1, 10, 28, 28},
        memory::data_type::f32,
        memory::format_tag::nchw
    };

    memory conv_input_mem, conv_weights_mem, conv_bias_mem, conv_output_mem;
    convolution_forward conv;

    // relu layer
    eltwise_forward relu;

    // max pool layer
    memory::desc pool_output_md{
        {1, 10, 13, 13},
        memory::data_type::f32,
        memory::format_tag::nchw
    };

    memory pool_output_mem;
    pooling_forward pool;

    // full connection layer
    memory::desc fc_input_md{
        {1, 1690},
        memory::data_type::f32,
        memory::format_tag::nc
    };

    memory::desc fc_weights_md{
        {10, 1690},
        memory::data_type::f32,
        memory::format_tag::oi
    };

    memory::desc fc_bias_md{
        {10},
        memory::data_type::f32,
        memory::format_tag::x
    };

    memory::desc fc_output_md{
        {1, 10},
        memory::data_type::f32,
        memory::format_tag::nc
    };

    memory fc_weights_mem, fc_bias_mem, fc_output_mem;
    inner_product_forward fc;

    explicit SimpleCNN(const engine &eng)
        : conv_input_mem(conv_input_md, eng),
          conv_weights_mem(conv_weights_md, eng),
          conv_bias_mem(conv_bias_md, eng),
          conv_output_mem(conv_output_md, eng),
          conv({
              eng,
              prop_kind::forward_inference,
              algorithm::convolution_direct,
              conv_input_md,
              conv_weights_md,
              conv_bias_md,
              conv_output_md,
              {1, 1},
              {0, 0,},
              {0, 0},
              {0, 0}
          }),

          relu({
              eng,
              prop_kind::forward_inference,
              algorithm::eltwise_relu,
              conv_output_md,
              conv_output_md
          }),

          pool_output_mem(pool_output_md, eng),
          pool({
              eng,
              prop_kind::forward_inference,
              algorithm::pooling_max,
              conv_output_md,
              pool_output_md,
              {2, 2},
              {3, 3},
              {0, 0},
              {0, 0},
              {0, 0}
          }),

          fc_weights_mem(fc_weights_md, eng),
          fc_bias_mem(fc_bias_md, eng),
          fc_output_mem(fc_output_md, eng),
          fc({
              eng,
              prop_kind::forward_inference,
              fc_input_md,
              fc_weights_md,
              fc_bias_md,
              fc_output_md
          }) {
    }

    void inference(const engine &eng, stream &str, float *input, float *output) {
        // copy input data
        std::memcpy(
            conv_input_mem.get_data_handle(),
            input,
            conv_input_md.get_size());

        // execute primitives
        conv.execute(
            str, {
                {DNNL_ARG_SRC, conv_input_mem},
                {DNNL_ARG_WEIGHTS, conv_weights_mem},
                {DNNL_ARG_BIAS, conv_bias_mem},
                {DNNL_ARG_DST, conv_output_mem}
            });

        relu.execute(
            str, {
                {DNNL_ARG_SRC, conv_output_mem},
                {DNNL_ARG_DST, conv_output_mem}
            });

        pool.execute(
            str, {
                {DNNL_ARG_SRC, conv_output_mem},
                {DNNL_ARG_DST, pool_output_mem}
            });

        // reuse pool_output_mem as fc_input_mem
        auto fc_input_mem = memory(fc_input_md, eng, pool_output_mem.get_data_handle());
        fc.execute(
            str, {
                {DNNL_ARG_SRC, fc_input_mem},
                {DNNL_ARG_WEIGHTS, fc_weights_mem},
                {DNNL_ARG_BIAS, fc_bias_mem},
                {DNNL_ARG_DST, fc_output_mem}
            });

        // wait execute
        str.wait();

        // copy output data
        auto fc_output_data = static_cast<float *>(fc_output_mem.get_data_handle());
        std::memcpy(output, fc_output_data, fc_output_md.get_size());
    }
};

int main() {
    std::vector<float> input(1 * 1 * 32 * 32);
    std::vector<float> output(10);

    engine eng(engine::kind::cpu, 0);
    stream str(eng);

    SimpleCNN model(eng);
    model.inference(eng, str, input.data(), output.data());

    for (int i = 0; i < output.size(); ++i) {
        std::cout << "Class " << i << ": " << output[i] << std::endl;
    }
}
