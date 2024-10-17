#include <iostream>
#include <dnnl.hpp>

int main() {
    using namespace dnnl;

    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Create memory descriptors for primitive
    // memory::format_tag::any means let primitive choose best format
    const int N = 1, H = 14, W = 14, IC = 128, OC = 256, KH = 3, KW = 3;

    auto conv_src_md = memory::desc(
        {N, IC, H, W},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_weights_md = memory::desc(
        {OC, IC, KH, KW},
        memory::data_type::f32,
        memory::format_tag::any
    );

    auto conv_dst_md = memory::desc(
        {N, OC, H, W},
        memory::data_type::f32,
        memory::format_tag::any
    );

    const auto &pool_dst_md = conv_dst_md;

    // Create convolution and pooling primitive descriptors
    auto conv_pd = convolution_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        algorithm::convolution_auto,
        conv_src_md,
        conv_weights_md,
        conv_dst_md,
        {1, 1}, // strides
        {1, 1}, {1, 1} // left and right padding
    );

    auto pool_pd = pooling_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        algorithm::pooling_max,
        conv_pd.dst_desc(),
        pool_dst_md,
        {1, 1}, {KH, KW}, // strides and kernel
        {0, 0}, // dilation
        {1, 1}, {1, 1} // left and right padding
    );

    // Suppose we have input memory as following
    auto src_mem = memory(
        {
            {N, IC, H, W},
            memory::data_type::f32,
            memory::format_tag::nchw
        },
        eng);

    auto weights_mem = memory(
        {
            {OC, IC, KH, KW},
            memory::data_type::f32,
            memory::format_tag::oihw
        },
        eng);

    auto dst_mem = memory(
        {
            {N, OC, H, W},
            memory::data_type::f32,
            memory::format_tag::nchw
        },
        eng);

    // Determine if source and weight needs to be reordered]
    bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();
    bool need_reorder_weights = conv_pd.weights_desc() != weights_mem.get_desc();
    bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.get_desc();

    // Allocate intermediate memory if necessary
    auto conv_src_mem = need_reorder_src ? memory(conv_pd.src_desc(), eng) : src_mem;
    auto conv_weights_mem = need_reorder_weights ? memory(conv_pd.weights_desc(), eng) : weights_mem;
    auto conv_dst_mem = memory(conv_pd.dst_desc(), eng);
    auto pool_dst_mem = need_reorder_dst ? memory(pool_pd.dst_desc(), eng) : dst_mem;

    // Perform reorders for source and weight if necessary
    if (need_reorder_src) {
        auto reorder_src = reorder(src_mem, conv_src_mem);
        reorder_src.execute(
            s, {
                {DNNL_ARG_FROM, src_mem},
                {DNNL_ARG_TO, conv_src_mem}
            });
        s.wait();
    }

    if (need_reorder_weights) {
        auto reorder_weights = reorder(weights_mem, conv_weights_mem);
        reorder_weights.execute(
            s, {
                {DNNL_ARG_FROM, weights_mem},
                {DNNL_ARG_TO, conv_weights_mem}
            });
        s.wait();
    }

    // Create and execute convolution and pooling primitives
    auto conv = convolution_forward(conv_pd);
    conv.execute(
        s, {
            {DNNL_ARG_SRC, conv_src_mem},
            {DNNL_ARG_WEIGHTS, conv_weights_mem},
            {DNNL_ARG_DST, conv_dst_mem},
        });

    auto pool = pooling_forward(pool_pd);
    pool.execute(
        s, {
            {DNNL_ARG_SRC, conv_dst_mem},
            {DNNL_ARG_DST, pool_dst_mem},
        });
    s.wait();

    // Reorder destination memory if necessary
    if (need_reorder_dst) {
        auto reorder_dst = reorder(pool_dst_mem, dst_mem);
        reorder_dst.execute(
            s, {
                {DNNL_ARG_FROM, pool_dst_mem},
                {DNNL_ARG_TO, dst_mem}
            });
        s.wait();
    }

    // Now output could be read from dst_mem
}
