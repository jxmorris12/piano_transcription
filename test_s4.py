from pytorch.models import Regress_onset_offset_frame_velocity_S4
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "%"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        total_params += parameter.numel()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param_num = parameter.numel()
        table.add_row([name, param_num, f'{param_num/total_params*100}'])
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
model = Regress_onset_offset_frame_velocity_S4(100, 88)
print(model)
print()
count_parameters(model)