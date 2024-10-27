import nncf
import numpy as np
import openvino as ov

ov_model = ov.convert_model('simple_cnn.onnx')
ov.save_model(ov_model, "simple_cnn.xml", compress_to_fp16=False)

calibrate_data = np.load("mnist_data.npy")
calibration_dataset = nncf.Dataset(calibrate_data, lambda x: x[np.newaxis, ...])
quantized_model = nncf.quantize(ov_model, calibration_dataset)
ov.save_model(quantized_model, "simple_cnn_quant.xml")
