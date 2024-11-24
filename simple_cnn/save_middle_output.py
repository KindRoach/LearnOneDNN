import numpy as np
import onnx
import onnxruntime as ort

model = onnx.load('simple_cnn_static_quant.onnx')
inter_layers = [
    "input_quantized",
    "/conv/Conv_output_0_quantized",
    "/pool/MaxPool_output_0_quantized",
    "/Reshape_output_0_quantized",
]  # output tensor names

value_info_protos = []
shape_info = onnx.shape_inference.infer_shapes(model)
for idx, node in enumerate(shape_info.graph.value_info):
    if node.name in inter_layers:
        value_info_protos.append(node)
assert len(value_info_protos) == len(inter_layers)

model.graph.output.extend(value_info_protos)  # in inference stage, these tensor will be added to output dict.
onnx.checker.check_model(model)
onnx.save(model, './simple_cnn_debug.onnx')

# 加载 ONNX 模型
session = ort.InferenceSession('./simple_cnn_debug.onnx')
input_name = session.get_inputs()[0].name
calibrate_data = np.load("mnist_data.npy")
input_data = calibrate_data[0][np.newaxis, ...]  # 示例输入数据

# 执行推理
output = session.run(None, {input_name: input_data})

pass