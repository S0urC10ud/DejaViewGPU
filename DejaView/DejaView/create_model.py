import torch
import torchvision.models as models
model = models.mobilenet_v2(pretrained=True)
model.eval()
dynamic_axes = {
    'input': {0: 'batch_size', 2: 'height', 3: 'width'},
    'output': {0: 'batch_size'}
}
dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor
torch.onnx.export(model, dummy_input, "Static\\mobilenetv2_dynamic.onnx",
                input_names=['input'], output_names=['output'],
                dynamic_axes=dynamic_axes, opset_version=11)