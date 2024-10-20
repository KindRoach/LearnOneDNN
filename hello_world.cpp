#include <iostream>
#include <dnnl.hpp>

int main() {
    using namespace dnnl;

    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    const int N = 4, H = 224, W = 224, C = 3;

    // Compute physical strides for each dimension
    const int stride_N = H * W * C;
    const int stride_H = W * C;
    const int stride_W = C;
    const int stride_C = 1;

    // An auxiliary function that maps logical index to the physical offset
    auto offset = [=](int n, int h, int w, int c) {
        return n * stride_N + h * stride_H + w * stride_W + c * stride_C;
    };

    // Create input memory object
    auto src_md = memory::desc(
        {N, C, H, W}, // logical dims, the order is defined by a primitive
        memory::data_type::f32, // tensor's data type
        memory::format_tag::nhwc // memory format, NHWC in this case
    );
    auto src_mem = memory(src_md, eng);

    // Initialize the image with some values
    auto src_ptr = static_cast<float *>(src_mem.get_data_handle());
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(n, h, w, c);
                    src_ptr[off] = -std::cos(static_cast<float>(off) / 10.f);
                }

    // Create output memory object
    auto dst_md = memory::desc(
        {N, C, H, W}, // logical dims, the order is defined by a primitive
        memory::data_type::f32, // tensor's data type
        memory::format_tag::nhwc // memory format, NHWC in this case
    );
    auto dst_mem = memory(dst_md, eng);

    // Create ReLU primitive
    auto relu_pd = eltwise_forward::primitive_desc(
        eng, // an engine the primitive will be created for
        prop_kind::forward_inference, algorithm::eltwise_relu,
        src_md, // source memory descriptor for an operation to work on
        src_md, // destination memory descriptor for an operation to work on
        0.f, // alpha parameter means negative slope in case of ReLU
        0.f // beta parameter is ignored in case of ReLU
    );
    auto relu = eltwise_forward(relu_pd);

    // The execute primitive via stream
    relu.execute(
        s,
        {
            // A map with all inputs and outputs
            {DNNL_ARG_SRC, src_mem}, // Source tag and memory obj
            {DNNL_ARG_DST, dst_mem}, // Destination tag and memory obj
        });

    // Wait the stream to complete the execution
    s.wait();

    // Verify the output
    auto dst_ptr = static_cast<float *>(dst_mem.get_data_handle());
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h)
            for (int w = 0; w < W; ++w)
                for (int c = 0; c < C; ++c) {
                    int off = offset(n, h, w, c);
                    float expected = src_ptr[off] < 0 ? 0.f : src_ptr[off];
                    float actual = dst_ptr[off];
                    if (actual != expected) {
                        std::cout << "At index(" << n << ", " << c << ", " << h
                                << ", " << w << ") expect " << expected
                                << " but got " << actual << std::endl;
                        throw std::logic_error("Accuracy check failed.");
                    }
                }

    std::cout << "Passed!" << std::endl;
}
