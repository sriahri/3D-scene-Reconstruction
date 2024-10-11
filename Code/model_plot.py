import torch
from torchviz import make_dot
from resnet_model import ResnetModel  # Replace 'your_module' with the actual module name

# Create an instance of the model
model = ResnetModel(num_in_layers=3)  # Assuming num_in_layers is 3 for RGB input

# Generate a dummy input tensor
dummy_input = torch.randn(1, 3, 256, 512)  # Replace with appropriate input shape

# Perform a forward pass to compute the model output
output = model(dummy_input)

# Generate the model plot
dot = make_dot(output, params=dict(model.named_parameters()))

# Save the model plot to a file (optional)
dot.render("model_plot", format="png")

# Display the model plot (optional)
dot.view()