#include <iostream>
#include <vector>
#include <dnnl.hpp>
#include <npy.hpp>

using namespace dnnl;

class SimpleCNN {
public:
    // user input & output
    memory::desc user_input_md, user_output_md;
    memory user_input_mem, user_output_mem;

    // convolution layer
    memory::desc conv_input_md, conv_weights_md, conv_bias_md, conv_output_md;
    memory conv_input_mem, conv_weights_mem, conv_bias_mem, conv_output_mem;
    convolution_forward conv;

    // convolution scale and zero points
    memory conv_input_scale_mem, conv_weights_scale_mem, conv_output_scale_mem;
    memory conv_input_zero_point_mem, conv_output_zero_point_mem;

    // max pool layer
    memory::desc pool_output_md;
    memory pool_output_mem;
    pooling_forward pool;

    // full connection layer
    memory::desc fc_input_md, fc_weights_md, fc_bias_md, fc_output_md;
    memory fc_weights_mem, fc_bias_mem, fc_output_mem;
    inner_product_forward fc;

    // full connection scale and zero points
    memory fc_weights_scale_mem, fc_output_scale_mem;
    memory fc_output_zero_point_mem;

    explicit SimpleCNN(const engine &eng):
        // user input & output
        user_input_md(
            {1, 1, 32, 32},
            memory::data_type::f32,
            memory::format_tag::nchw
        ),
        user_output_md(
            {1, 10},
            memory::data_type::f32,
            memory::format_tag::nc
        ),
        user_input_mem(user_input_md, eng),
        user_output_mem(user_output_md, eng),

        // max pool layer
        conv_input_md(
            {1, 1, 32, 32},
            memory::data_type::u8,
            memory::format_tag::nchw
        ),
        conv_weights_md(
            {10, 1, 5, 5},
            memory::data_type::s8,
            memory::format_tag::oihw
        ),
        conv_bias_md(
            {10},
            memory::data_type::s32,
            memory::format_tag::x
        ),
        conv_output_md(
            {1, 10, 28, 28},
            memory::data_type::u8,
            memory::format_tag::nchw
        ),
        conv_input_mem(conv_input_md, eng),
        conv_weights_mem(conv_weights_md, eng),
        conv_bias_mem(conv_bias_md, eng),
        conv_output_mem(conv_output_md, eng),

        // convolution scale and zero points
        conv_input_scale_mem({{1}, memory::data_type::f32, memory::format_tag::x}, eng),
        conv_weights_scale_mem({{1}, memory::data_type::f32, memory::format_tag::x}, eng),
        conv_output_scale_mem({{1}, memory::data_type::f32, memory::format_tag::x}, eng),
        conv_input_zero_point_mem({{1}, memory::data_type::u8, memory::format_tag::x}, eng),
        conv_output_zero_point_mem({{1}, memory::data_type::u8, memory::format_tag::x}, eng),

        // max pool layer
        pool_output_md(
            {1, 10, 13, 13},
            memory::data_type::u8,
            memory::format_tag::nchw
        ),
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

        // full connection layer
        fc_input_md(
            {1, 1690},
            memory::data_type::u8,
            memory::format_tag::nc
        ),
        fc_weights_md(
            {10, 1690},
            memory::data_type::s8,
            memory::format_tag::oi
        ),
        fc_bias_md(
            {10},
            memory::data_type::s32,
            memory::format_tag::x
        ),
        fc_output_md(
            {1, 10},
            memory::data_type::u8,
            memory::format_tag::nc
        ),
        fc_weights_mem(fc_weights_md, eng),
        fc_bias_mem(fc_bias_md, eng),
        fc_output_mem(fc_output_md, eng),

        // full connection scale and zero points
        fc_weights_scale_mem({{1}, memory::data_type::f32, memory::format_tag::x}, eng),
        fc_output_scale_mem({{1}, memory::data_type::f32, memory::format_tag::x}, eng),
        fc_output_zero_point_mem({{1}, memory::data_type::u8, memory::format_tag::x}, eng) {
        // conv with quant and relu post-op
        primitive_attr conv_attr;

        // conv relu post-op
        post_ops ops;
        ops.append_eltwise(algorithm::eltwise_relu, 0.f, 0.f);
        conv_attr.set_post_ops(ops);

        // conv quant
        conv_attr.set_scales_mask(DNNL_ARG_SRC, 0);
        conv_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
        conv_attr.set_scales_mask(DNNL_ARG_DST, 0);
        conv_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
        conv_attr.set_zero_points_mask(DNNL_ARG_DST, 0);

        conv = convolution_forward{
            {
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
                {0, 0},
                conv_attr
            }
        };

        // full connection with quant
        primitive_attr fc_attr;
        fc_attr.set_scales_mask(DNNL_ARG_SRC, 0);
        fc_attr.set_scales_mask(DNNL_ARG_WEIGHTS, 0);
        fc_attr.set_scales_mask(DNNL_ARG_DST, 0);
        fc_attr.set_zero_points_mask(DNNL_ARG_SRC, 0);
        fc_attr.set_zero_points_mask(DNNL_ARG_DST, 0);

        fc = inner_product_forward{
            {
                eng,
                prop_kind::forward_inference,
                fc_input_md,
                fc_weights_md,
                fc_bias_md,
                fc_output_md
            }
        };
    }

    void load_weights(const std::string &weights_dir) {
        // Conv layer
        load_weight<float>(weights_dir + "/input_scale.npy", conv_input_scale_mem);
        load_weight<uint8_t>(weights_dir + "/input_zero_point.npy", conv_input_zero_point_mem);
        load_weight<int8_t>(weights_dir + "/conv.weight_quantized.npy", conv_weights_mem);
        load_weight<float>(weights_dir + "/conv.weight_scale.npy", conv_weights_scale_mem);
        load_weight<int32_t>(weights_dir + "/conv.bias_quantized.npy", conv_bias_mem);
        load_weight<float>(weights_dir + "/_conv_Conv_output_0_scale.npy", conv_output_scale_mem);
        load_weight<uint8_t>(weights_dir + "/_conv_Conv_output_0_zero_point.npy", conv_output_zero_point_mem);

        // full connection layer
        load_weight<int8_t>(weights_dir + "/fc.weight_quantized.npy", fc_weights_mem);
        load_weight<float>(weights_dir + "/fc.weight_scale.npy", fc_weights_scale_mem);
        load_weight<int32_t>(weights_dir + "/fc.bias_quantized.npy", fc_bias_mem);
        load_weight<float>(weights_dir + "/output_scale.npy", fc_output_scale_mem);
        load_weight<uint8_t>(weights_dir + "/output_zero_point.npy", fc_output_zero_point_mem);
    }

    template<typename T>
    static void load_weight(const std::string &weights_file_path, memory &mem) {
        const auto weights = npy::read_npy<T>(weights_file_path).data;
        std::memcpy(
            mem.get_data_handle(),
            weights.data(),
            mem.get_desc().get_size()
        );
    }

    void inference(const engine &eng, stream &str, float *input, float *output) {
        // copy input data and quant
        std::memcpy(
            user_input_mem.get_data_handle(),
            input,
            user_input_md.get_size());

        primitive_attr input_attr;
        input_attr.set_scales_mask(DNNL_ARG_DST, 0);
        input_attr.set_zero_points_mask(DNNL_ARG_DST, 0);
        auto src_reorder_pd = reorder::primitive_desc(
            eng, user_input_mem.get_desc(),
            eng, conv_input_mem.get_desc(), input_attr);
        auto src_reorder = reorder(src_reorder_pd);
        src_reorder.execute(
            str, {
                {DNNL_ARG_FROM, user_input_mem},
                {DNNL_ARG_TO, conv_input_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, conv_input_scale_mem},
                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, conv_input_zero_point_mem}
            });

        str.wait();

        // execute primitives
        conv.execute(
            str, {
                {DNNL_ARG_SRC, conv_input_mem},
                {DNNL_ARG_WEIGHTS, conv_weights_mem},
                {DNNL_ARG_BIAS, conv_bias_mem},
                {DNNL_ARG_DST, conv_output_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, conv_input_scale_mem},
                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, conv_input_zero_point_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, conv_weights_scale_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, conv_output_scale_mem},
                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, conv_output_zero_point_mem}
            });

        pool.execute(
            str, {
                {DNNL_ARG_SRC, conv_output_mem},
                {DNNL_ARG_DST, pool_output_mem}
            });

        str.wait();

        // reuse pool_output_mem as fc_input_mem
        auto fc_input_mem = memory(fc_input_md, eng, pool_output_mem.get_data_handle());

        fc.execute(
            str, {
                {DNNL_ARG_SRC, fc_input_mem},
                {DNNL_ARG_WEIGHTS, fc_weights_mem},
                {DNNL_ARG_BIAS, fc_bias_mem},
                {DNNL_ARG_DST, fc_output_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, conv_output_scale_mem},
                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, conv_output_zero_point_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, fc_weights_scale_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, fc_output_scale_mem},
                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, fc_output_zero_point_mem}
            });

        // De-quant output
        primitive_attr dst_attr;
        dst_attr.set_scales_mask(DNNL_ARG_SRC, 0);
        auto dst_reorder_pd = reorder::primitive_desc(
            eng, fc_output_mem.get_desc(),
            eng, user_output_mem.get_desc(), dst_attr);
        auto dst_reorder = reorder(dst_reorder_pd);
        dst_reorder.execute(
            str, {
                {DNNL_ARG_FROM, fc_output_mem},
                {DNNL_ARG_TO, user_output_mem},
                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, fc_output_scale_mem},
                {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, fc_output_zero_point_mem}
            });

        // wait execute
        str.wait();

        // copy output data
        auto user_output_data = static_cast<float *>(user_output_mem.get_data_handle());
        std::memcpy(output, user_output_data, user_output_md.get_size());
    }
};

int main() {
    engine eng(engine::kind::cpu, 0);
    stream str(eng);

    SimpleCNN model(eng);
    model.load_weights("simple_cnn/weights/int8_static_quant");

    // predict on MNIST test dataset
    auto data = npy::read_npy<float>("simple_cnn/mnist_data.npy").data;
    auto label = npy::read_npy<int64_t>("simple_cnn/mnist_label.npy").data;

    int size_per_sample = data.size() / label.size();
    std::vector<float> tmp(10);
    int acc_count = 0;
    for (int i = 0; i < label.size(); ++i) {
        float *data_i = data.data() + i * size_per_sample;
        model.inference(eng, str, data_i, tmp.data());
        auto max_it = std::ranges::max_element(tmp);
        int predict_label = std::distance(tmp.begin(), max_it);
        int actual_label = label[i];
        if (predict_label == actual_label) acc_count++;
    }

    std::cout << "acc: " << static_cast<float>(acc_count) / label.size() << std::endl;
}
