import torch
import torch.nn as nn
from torchviz import make_dot # NOTE: requires installation of graphviz tool

# Define your CustomClassifier
class CustomClassifier(nn.Module):
    def __init__(self, basic_input_size, num_audio_processing_features, num_classes, activation_function):
        super(CustomClassifier, self).__init__()
        self.basic_input_size = basic_input_size
        self.num_audio_processing_features = num_audio_processing_features
        self.num_classes = num_classes

        print(f'Custom classifier input size: {basic_input_size + num_audio_processing_features}')

        self.fc1 = nn.Linear(basic_input_size + num_audio_processing_features, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 256, bias=True)
        self.fc4 = nn.Linear(256, num_classes, bias=True)
        self.activation_function = activation_function

        self.linear = nn.Linear(basic_input_size, num_classes, bias=True)

    def forward(self, x):
        if self.num_audio_processing_features == 0:  # text only
            output = self.linear(x)
        else:
            x = self.fc1(x)
            x = self.activation_function(x)
            x = self.fc2(x)
            x = self.activation_function(x)
            x = self.fc3(x)
            x = self.activation_function(x)
            output = self.fc4(x)

        return output

# Example usage:
# Create an instance of the CustomClassifier
basic_input_size = 768
num_audio_processing_features = 1025
num_classes = 29
activation_function = nn.ReLU()

model_with_audio = CustomClassifier(basic_input_size, num_audio_processing_features, num_classes, activation_function)
model_text_only = CustomClassifier(basic_input_size, 0, num_classes, activation_function)

# Generate visualizations
# For the model with audio processing features
x = torch.randn(1, basic_input_size + num_audio_processing_features)
y = model_with_audio(x)
dot = make_dot(y, params=dict(model_with_audio.named_parameters()))
dot.render("model_with_audio", format="png")

# For the text-only model
x = torch.randn(1, basic_input_size)
y = model_text_only(x)
dot = make_dot(y, params=dict(model_text_only.named_parameters()))
dot.render("model_text_only", format="png")
