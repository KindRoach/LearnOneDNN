from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from onnx import shape_inference, numpy_helper
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc = nn.Linear(1690, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)}'
          f'({accuracy:.2f}%)\n')


def infer_onnx_shape(model_path):
    onnx_model = onnx.load(model_path)
    inferred_model = shape_inference.infer_shapes(onnx_model)
    onnx.save(inferred_model, model_path)


def save_onnx_model(model, model_name):
    torch.onnx.export(
        model, (torch.randn(1, 1, 32, 32),),
        f"{model_name}.onnx",
        input_names=['input'],
        output_names=['output'],
    )
    infer_onnx_shape(f"{model_name}.onnx")
    print("Onnx saved.")


def save_weight_from_onnx(onnx_model_path, dtype):
    # save weight and model
    weights_dir = f"weights/{dtype}"
    Path(weights_dir).mkdir(exist_ok=True, parents=True)
    model = onnx.load(onnx_model_path)

    # Extract weights (initializers in ONNX model)
    for initializer in model.graph.initializer:
        i_name = initializer.name.replace("/", "_")
        nparray = numpy_helper.to_array(initializer)
        np.save(f"{weights_dir}/{i_name}.npy", nparray)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # save test dataset
    mnist_transformed_data = []
    mnist_label = []
    for data, label in test_loader:
        mnist_transformed_data.append(data.numpy())
        mnist_label.append(label.numpy())

    mnist_transformed_data = np.vstack(mnist_transformed_data)
    mnist_label = np.hstack(mnist_label)
    np.save('mnist_data.npy', mnist_transformed_data)
    np.save('mnist_label.npy', mnist_label)

    # train model
    for epoch in range(10):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    model = model.eval().cpu()
    save_onnx_model(model, "simple_cnn")
    save_weight_from_onnx("simple_cnn.onnx", "fp32")


if __name__ == '__main__':
    main()
