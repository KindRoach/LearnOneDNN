#include <iostream>
#include <dnnl.hpp>

int main() {
    using namespace dnnl;

    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    using tag = memory::format_tag;
    using dt = memory::data_type;

    // AlexNet: conv3
    // {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
    // strides: {1, 1}
    memory::dims conv_src_tz = {1, 256, 13, 13};
    memory::dims conv_weights_tz = {384, 256, 3, 3};
    memory::dims conv_bias_tz = {384};
    memory::dims conv_dst_tz = {1, 384, 13, 13};
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {1, 1};

    // Set scaling mask
    const int src_mask = 0;
    const int weight_mask = 0;
    const int dst_mask = 0;

    // Input buffers
    auto user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);

    // Create convolution memory descriptors
    auto conv_src_md = memory::desc({conv_src_tz}, dt::u8, tag::nchw);
    auto conv_bias_md = memory::desc({conv_bias_tz}, dt::s32, tag::x);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::s8, tag::iohw);
    auto conv_dst_md = memory::desc({conv_dst_tz}, dt::u8, tag::nchw);

    // Configure scaling and zero points
    primitive_attr conv_attr;
    conv_attr.set_scales_mask(DNNL_ARG_SRC, src_mask);
    conv_attr.set_scales_mask(DNNL_ARG_WEIGHTS, weight_mask);
    conv_attr.set_scales_mask(DNNL_ARG_DST, dst_mask);
    conv_attr.set_zero_points_mask(DNNL_ARG_SRC, src_mask);
    conv_attr.set_zero_points_mask(DNNL_ARG_DST, dst_mask);

    // Configure post-ops
    const float ops_alpha = 0.f; // relu negative slope
    const float ops_beta = 0.f;
    post_ops ops;
    ops.append_eltwise(algorithm::eltwise_relu, ops_alpha, ops_beta);
    conv_attr.set_post_ops(ops);

    // check if int8 convolution is supported
    try {
        convolution_forward::primitive_desc(
            eng,
            prop_kind::forward,
            algorithm::convolution_auto,
            conv_src_md, conv_weights_md,
            conv_bias_md, conv_dst_md,
            conv_strides, conv_padding,
            conv_padding, conv_attr);
    } catch (error &e) {
        if (e.status == dnnl_unimplemented)
            throw std::exception{
                "No int8 convolution implementation is available for this platform.\n"
                "Please refer to the developer guide for details."
            };

        // on any other error just re-throw
        throw;
    }

    // Create convolution primitive descriptor
    auto conv_prim_desc = convolution_forward::primitive_desc(
        eng,
        prop_kind::forward, algorithm::convolution_direct,
        conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
        conv_padding, conv_padding, conv_attr);

    // Create convolution memory
    auto conv_src_memory = memory(conv_src_md, eng);
    auto conv_bias_memory = memory(conv_bias_md, eng);
    auto conv_weights_memory = memory(conv_weights_md, eng);
    auto conv_dst_memory = memory(conv_dst_md, eng);

    // Scale and zero_point memory
    auto src_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto src_scale_memory = memory(src_scale_md, eng);
    auto src_zero_point_md = memory::desc({1}, dt::f32, tag::x);
    auto src_zero_point_memory = memory(src_zero_point_md, eng);

    auto bias_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto bias_scale_memory = memory(bias_scale_md, eng);
    auto bias_zero_point_md = memory::desc({1}, dt::f32, tag::x);
    auto bias_zero_point_memory = memory(bias_zero_point_md, eng);

    auto wei_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto wei_scale_memory = memory(wei_scale_md, eng);

    auto dst_scale_md = memory::desc({1}, dt::f32, tag::x);
    auto dst_scale_memory = memory(dst_scale_md, eng);
    auto dst_zero_point_md = memory::desc({1}, dt::f32, tag::x);
    auto dst_zero_point_memory = memory(dst_zero_point_md, eng);

    // Quantize input data
    primitive_attr src_attr;
    src_attr.set_scales_mask(DNNL_ARG_DST, src_mask);
    auto src_reorder_pd = reorder::primitive_desc(
        eng, user_src_memory.get_desc(),
        eng, conv_src_memory.get_desc(), src_attr);
    auto src_reorder = reorder(src_reorder_pd);
    src_reorder.execute(
        s, {
            {DNNL_ARG_FROM, user_src_memory},
            {DNNL_ARG_TO, conv_src_memory},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, src_scale_memory},
            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, src_zero_point_memory}
        });

    auto conv = convolution_forward(conv_prim_desc);
    conv.execute(
        s, {
            {DNNL_ARG_SRC, conv_src_memory},
            {DNNL_ARG_WEIGHTS, conv_weights_memory},
            {DNNL_ARG_BIAS, conv_bias_memory},
            {DNNL_ARG_DST, conv_dst_memory},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scale_memory},
            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, src_zero_point_memory},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scale_memory},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, dst_scale_memory},
            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, dst_zero_point_memory}
        });

    // De-quantize the result
    auto user_dst_memory = memory({{conv_dst_tz}, dt::f32, tag::nchw}, eng);
    primitive_attr dst_attr;
    dst_attr.set_scales_mask(DNNL_ARG_SRC, dst_mask);
    auto dst_reorder_pd = reorder::primitive_desc(
        eng, conv_dst_memory.get_desc(),
        eng, user_dst_memory.get_desc(), dst_attr);
    auto dst_reorder = reorder(dst_reorder_pd);
    dst_reorder.execute(
        s, {
            {DNNL_ARG_FROM, conv_dst_memory},
            {DNNL_ARG_TO, user_dst_memory},
            {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, dst_scale_memory},
            {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, dst_zero_point_memory}
        });

    s.wait();
}
