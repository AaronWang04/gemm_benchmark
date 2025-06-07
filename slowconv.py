import torch
import torch.nn as nn

device = torch.device("cuda")
num_channels = 256

conv_layer = nn.Conv3d(num_channels, num_channels, kernel_size=3, stride=1, padding=1).to(device).half()

input_tensor = torch.randn(1, num_channels, 7, 1088, 1088, device=device).half()
# input_tensor = torch.randn(1, num_channels, 7, 1088, 1088, device=device).half().to(memory_format=torch.channels_last_3d)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
output_tensor = conv_layer(input_tensor)
torch.cuda.synchronize()
end_event.record()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"32 bit: {elapsed_time_ms}")


input_tensor = torch.randn(1, num_channels, 7, 1090, 1106, device=device).half()
# input_tensor = torch.randn(1, num_channels, 7, 1090, 1106, device=device).half().to(memory_format=torch.channels_last_3d)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
output_tensor = conv_layer(input_tensor)
torch.cuda.synchronize()
end_event.record()
elapsed_time_ms = start_event.elapsed_time(end_event)
print(f"64 bit: {elapsed_time_ms}")
