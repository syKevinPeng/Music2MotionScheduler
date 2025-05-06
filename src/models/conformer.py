import torch.nn as nn


class ConvolutionModule(nn.Module):
    def __init__(self,
                 input_dim,
                 num_channels,
                 cnn_kernel_size,
                 dropout,
                 bias,
                 use_group_norm=False):
        super().__init__()
        if (cnn_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        
        self.layer_norm = nn.LayerNorm(input_dim)
        self.sequential = nn.Sequential(
            nn.Conv1d(input_dim, 2 * num_channels, 1, stride=1, padding=0, bias=bias),
            nn.GLU(dim=1),
            nn.Conv1d(num_channels, num_channels, cnn_kernel_size, stride=1, padding=(cnn_kernel_size - 1) // 2, groups=num_channels, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=num_channels) if use_group_norm else nn.BatchNorm1d(num_channels),
            nn.SiLU(),
            nn.Conv1d(num_channels, input_dim, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Dropout(dropout),
        )
    
    def forward(self, input):
        """
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        x = x.transpose(1, 2)
        return x

class ConformerLayer(nn.Module):
    """
    Conformer mostly used from https://github.com/MTG/violin-transcription
    """
    def __init__(self,
                 d_model, 
                 cnn_kernel_size,
                 num_head,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dim_feedforward = dim_feedforward

        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, dim_feedforward, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=True),
            nn.Dropout(dropout),
        )
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward, bias=True),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias=True),
            nn.Dropout(dropout),
        )

        self.conv_module = ConvolutionModule(
            input_dim=d_model,
            num_channels=d_model,
            cnn_kernel_size=cnn_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=False
        )
        
    
    def _apply_convolution(self, input):
        """
        Args:
            input (torch.Tensor): with shape `(T, B, D)`.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input
        
    def forward(self,
                src,
                src_mask=None,
                src_key_padding_mask=None,
                pos=None):
        """
        Args:
            src with shape `(T, B, D)`
        Returns:
            src with shape `(T, B, D)`
        """
        residual = src
        x = self.ffn1(src)
        x = x * 0.5 + residual

        residual = x

        x = self.layer_norm(x)
        x = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]        

        x = self.dropout(x)
        x = x + residual

        x = self._apply_convolution(x)

        residual = x 

        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.layer_norm2(x)

        return x
