import torch
from nr_transformer_model import NrTransformerModel

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

model = NrTransformerModel(input_dim=6)
model.load_state_dict(torch.load('/home/maximilianrosca/Masterarbeit/Simulation-files/nr_transformer_model.pth', weights_only=True))
model = model.to(device)
model.eval()

inputs = torch.tensor([[1.00006500663827, 1.00006500663827, -0.669604008250139, -1.00006500663827, -0.879627639948147, -0.8522485608675], [-0.99992499753734, -0.99992499753734, 0.077811062539645, 0.99992499753734, -0.839088006101818, -0.93972216062866]], dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    outputs = model(inputs)
    outputs = outputs.detach().cpu().numpy()
print("Outputs:", outputs)
