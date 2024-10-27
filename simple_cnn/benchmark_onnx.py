import argparse
import time

import numpy as np
import onnxruntime as ort
import tqdm
from tqdm import tqdm


def main(model_path):
    session = ort.InferenceSession(model_path)

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_data = np.random.random(size=input_shape).astype(np.float32)

    test_duration = 10
    num_iterations = 0
    start_time = time.time()

    with tqdm(total=test_duration, desc="Inference Progress") as pbar:
        while time.time() - start_time < test_duration:
            session.run(None, {input_name: input_data})
            num_iterations += 1
            pbar.update(1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = num_iterations / elapsed_time

    print(f'Test Duration: {test_duration} seconds')
    print(f'Total Inferences: {num_iterations}')
    print(f'Average FPS: {fps:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ONNX model inference speed.')
    parser.add_argument('model_path', type=str, help='Path to the ONNX model file.')
    args = parser.parse_args()

    main(args.model_path)
