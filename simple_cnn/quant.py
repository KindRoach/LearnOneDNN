import numpy as np
import subprocess

from onnxruntime import quantization
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType

from train import save_weight_from_onnx, infer_onnx_shape


def dynamic_quant():
    onnx_model = "simple_cnn.onnx"
    quantized_onnx_model = "simple_cnn_dynamic_quant.onnx"
    quantization.quantize_dynamic(
        onnx_model, quantized_onnx_model,
        weight_type=QuantType.QInt8
    )
    save_weight_from_onnx(quantized_onnx_model, "int8_dynamic_quant")
    infer_onnx_shape(quantized_onnx_model)


def static_quant():
    class DataReader(CalibrationDataReader):
        def __init__(self, data):
            self.data = data
            self.data_iter = iter(self.data)

        def get_next(self):
            return next(self.data_iter, None)

        def rewind(self):
            self.data_iter = iter(self.data)

    calibrate_data = np.load("mnist_data.npy")
    calibrate_data = [{"input": x[np.newaxis, :]} for x in calibrate_data]
    data_reader = DataReader(calibrate_data)

    onnx_model = "simple_cnn.onnx"
    quantized_onnx_model = "simple_cnn_static_quant.onnx"
    quantization.quantize_static(
        onnx_model, quantized_onnx_model, data_reader,
        QuantFormat.QOperator,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )
    save_weight_from_onnx(quantized_onnx_model, "int8_static_quant")

    # fix conv.bias: oneDNN conv bias == Onnx QLinearConv bias * x_scale * w_scale
    weights_dir = "weights/int8_static_quant"
    conv_bias = np.load(f"{weights_dir}/conv.bias_quantized.npy")
    conv_x_scale = np.load(f"{weights_dir}/input_scale.npy")
    conv_w_scale = np.load(f"{weights_dir}/conv.weight_scale.npy")
    conv_bias = (conv_bias * conv_x_scale * conv_w_scale).astype(conv_bias.dtype)
    np.save(f"{weights_dir}/conv.bias_quantized.npy", conv_bias)

    infer_onnx_shape(quantized_onnx_model)


if __name__ == '__main__':
    onnx_model = "simple_cnn.onnx"
    command = [
        "python", "-m", "onnxruntime.quantization.preprocess",
        "--input", onnx_model,
        "--output", onnx_model
    ]

    subprocess.run(command, check=True, shell=True)

    dynamic_quant()
    static_quant()
