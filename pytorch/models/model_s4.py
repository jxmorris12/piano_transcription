import torch
import torch.nn as nn

from .s4 import S4

class S4Model(nn.Module):
    def __init__(
        self, 
        d_input, 
        d_output=10, 
        d_model=256, 
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(H=d_model, l_max=1024, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)
        
        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

class Regress_onset_offset_frame_velocity_S4(nn.Module):
    def __init__(self, frames_per_second, classes_num):
        super().__init__()

        self.time_steps = 1001 # output predictions for 1001 timesteps (just copying the RCNN model)
        self.mid_rep_size = 16#32
        rep_size = self.time_steps*self.mid_rep_size
        self.s4 = S4Model(
            1, # for waveform (number of "channels") 
            d_output=rep_size, 
            d_model=128, #256
            n_layers=4, 
            dropout=0.2,
        )

        self.dropout = nn.Dropout(0.5)
        # TODO is dropout the right choice here?
        self.frame_model = nn.Linear(self.mid_rep_size, classes_num)
        self.reg_onset_model = nn.Linear(self.mid_rep_size, classes_num)
        self.reg_offset_model = nn.Linear(self.mid_rep_size, classes_num)
        self.velocity_model = nn.Linear(self.mid_rep_size, classes_num)
 
    def forward(self, input):
        """
        Args:
          input: (batch_size, data_length)

        Outputs:
          output_dict: dict, {
            'reg_onset_output': (batch_size, time_steps, classes_num),
            'reg_offset_output': (batch_size, time_steps, classes_num),
            'frame_output': (batch_size, time_steps, classes_num),
            'velocity_output': (batch_size, time_steps, classes_num)
          }
        """
        batch_size, data_length = input.shape
        input = torch.unsqueeze(input, dim=-1) # add fake channel dimension for s4
        x = self.s4(input)   # (batch_size, 1, time_steps * rep_size)
        x = self.dropout(x)
        x = x.reshape((batch_size, self.time_steps, self.mid_rep_size))
        
        frame_output = self.frame_model(x)  # (batch_size, time_steps, classes_num)
        reg_onset_output = self.reg_onset_model(x)  # (batch_size, time_steps, classes_num)
        reg_offset_output = self.reg_offset_model(x)    # (batch_size, time_steps, classes_num)
        velocity_output = self.velocity_model(x)    # (batch_size, time_steps, classes_num)

        output_dict = {
            'reg_onset_output': reg_onset_output, 
            'reg_offset_output': reg_offset_output, 
            'frame_output': frame_output, 
            'velocity_output': velocity_output
        }

        # print('output_dict:', {k: v.shape for k, v in output_dict.items()}, f'(input.shape = {input.shape})')
        # breakpoint()
        return output_dict