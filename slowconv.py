import torch
import torch.nn as nn

device = torch.device("cuda")

# Define an extremely large input tensor (exceeding 2**31 elements for a single sample), use 3d convolutions
# Total elements = 1 * 256 * 7 * 1090 * 1106 = 2,160,327,680 > 2**31 (2,147,483,648)
num_channels = 256

# Define a convolution layer
conv_layer = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1).to(device).half()

input_tensor = torch.randn(1, num_channels, 7, 1088, 1088, device=device).half().to(memory_format=torch.channels_last_3d)

# Takes 0.7 seconds

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
output_tensor = conv_layer(input_tensor)
torch.cuda.synchronize()
end_event.record()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"short: {elapsed_time_ms}")
# Input exceeding 2 ** 31
input_tensor = torch.randn(1, num_channels, 7, 1090, 1106, device=device).half().to(memory_format=torch.channels_last_3d)
# Takes 22 seconds

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
output_tensor = conv_layer(input_tensor)
torch.cuda.synchronize()
end_event.record()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"long: {elapsed_time_ms})
