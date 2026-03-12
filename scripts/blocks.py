import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CLS_PositionalEncoding(nn.Module):
    def __init__(
        self,
        base_dim, # 128
        hidden_dim, # 256
        output_dim, # 512
    ):
        super().__init__()

        # In this example, we assume that the number of embedding dimension is always even.
        # (If not, please pad the result.)
        assert(base_dim % 2 == 0)
        self.timestep_dim = base_dim

        self.hidden1 = nn.Linear(
            base_dim,
            hidden_dim)
        self.hidden2 = nn.Linear(
            hidden_dim,
            output_dim)

    def forward(self, picked_up_timesteps):
        """
        Generate timestep embedding vectors
    
        Parameters
        ----------
        picked_up_timesteps : torch.tensor((batch_size), dtype=int)
            Randomly picked up timesteps
    
        Returns
        ----------
        out : torch.tensor((batch_size, output_dim), dtype=float)
            Generated timestep embeddings (vectors) for each timesteps.
        """

        # Generate 1 / 10000^{2i / d_e}
        # shape : (timestep_dim / 2, )
        interval = 1.0 / (10000**(torch.arange(0, self.timestep_dim, 2.0).to(device) / self.timestep_dim))
        # Generate t / 10000^{2i / d_e}
        # shape : (batch_size, timestep_dim / 2)
        position = picked_up_timesteps.type(torch.get_default_dtype())
        radian = position[:, None] * interval[None, :]
        # Get sin(t / 10000^{2i / d_e}) and unsqueeze
        # shape : (batch_size, timestep_dim / 2, 1)
        sin = torch.sin(radian).unsqueeze(dim=-1)
        # Get cos(t / 10000^{2i / d_e}) and unsqueeze
        # shape : (batch_size, timestep_dim / 2, 1)
        cos = torch.cos(radian).unsqueeze(dim=-1)
        # Get sinusoidal positional encoding
        # shape : (batch_size, timestep_dim)
        pe_tmp = torch.concat((sin, cos), dim=-1)   # shape : (num_timestep, timestep_dim / 2, 2)
        d = pe_tmp.size()[1]
        pe = pe_tmp.view(-1, d * 2)                 # shape : (num_timestep, timestep_dim)
        # Apply feedforward
        # shape : (batch_size, timestep_dim * 4)
        out = self.hidden1(pe)
        out = F.silu(out)
        out = self.hidden2(out)

        return out
    



def mha_operation(q, k, v, num_heads):
    # get size and check
    channel = q.size()[-1]
    assert channel % num_heads == 0
    dim_heads = channel // num_heads

    q_len = q.size()[1]
    k_len = k.size()[1]
    v_len = v.size()[1]
    assert k_len == v_len

    # divide into multiple heads :
    #   --> (batch_size, length, num_heads, channel/num_heads)
    q_h = q.view(-1, q_len, num_heads, dim_heads)
    k_h = k.view(-1, k_len, num_heads, dim_heads)
    v_h = v.view(-1, v_len, num_heads, dim_heads)

    # compute Q K^T
    #   --> (batch_size, q_len, k_len, num_heads)
    score = torch.einsum("bihc,bjhc->bijh", q_h, k_h)

    # scale the result by 1/sqrt(channel)
    #   --> (batch_size, q_len, k_len, num_heads)
    score = score / channel**0.5

    # apply softtmax
    #   --> (batch_size, q_len, k_len, num_heads)
    score = F.softmax(score, dim=2)

    # apply dot product with values
    #   --> (batch_size, q_len, num_heads, channel/num_heads)
    out = torch.einsum("bijh,bjhc->bihc", score, v_h)

    # concatenate all heads (without heads)
    #   --> (batch_size, q_len, channel)
    out = out.reshape(-1, q_len, channel)

    return out




class CLS_ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_norm_groups, # 32
        timestep_embedding_dim, # 512
        down=False,
    ):
        super().__init__()

        self.down = down

        # for normalization
        self.norm1 = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=in_channel,
        )
        self.norm2 = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=out_channel,
        )

        # for applying conv
        self.conv1 = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.conv2 = nn.Conv2d(
            out_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        # to make first conv layer much contribute initially
        # (see https://arxiv.org/pdf/1901.09321)
        for p in self.conv2.parameters():
            p.detach().zero_()

        # for adding timestep
        self.linear_pos = nn.Linear(timestep_embedding_dim, out_channel)

        # for residual block
        if in_channel != out_channel:
            self.linear_src = nn.Linear(in_channel, out_channel)
        else:
            self.linear_src = None

    def forward(self, x, t_emb):
        """
        Parameters
        ----------
        x : torch.tensor((batch_size, in_channel, width, height), dtype=float)
            input x
        t_emb : torch.tensor((batch_size, base_channel_dim * 4), dtype=float)
            timestep embeddings
        """

        # apply group norm
        out = self.norm1(x)
        out = F.silu(out)

        # transform in each cases
        # (x_trans is used in last residual layer.)
        if self.down:
            out = F.avg_pool2d(out, (2, 2))
            x_trans = F.avg_pool2d(x, (2, 2))
        else:
            x_trans = x

        # apply conv
        out = self.conv1(out)

        # timestep projection
        pos = F.silu(t_emb)
        pos = self.linear_pos(pos)
        pos = pos[:, :, None, None]
        out = out + pos

        # apply dropout + conv
        out = self.norm2(out)
        out = F.silu(out)
        ##### out = F.dropout(out, p=0.1, training=self.training)
        out = self.conv2(out)

        # apply residual
        if self.linear_src is not None:
            x_trans = x_trans.permute(0, 2, 3, 1) # (N,C,H,W) --> (N,H,W,C)
            x_trans = self.linear_src(x_trans)
            x_trans = x_trans.permute(0, 3, 1, 2) # (N,H,W,C) --> (N,C,H,W)
        out = out + x_trans

        return out
    


class CLS_AttentionBlock(nn.Module):
    def __init__(
        self,
        channel,
        num_heads,
        num_norm_groups, # 32
    ):
        super().__init__()

        self.num_heads = num_heads

        self.norm = nn.GroupNorm(
            num_groups=num_norm_groups,
            num_channels=channel,
        )

        self.q_layer = nn.Linear(channel, channel)
        self.k_layer = nn.Linear(channel, channel)
        self.v_layer = nn.Linear(channel, channel)

        self.output_linear = nn.Linear(channel, channel, bias=False)

    def forward(self, x):
        channel = x.size(dim=1)
        height = x.size(dim=2)
        width = x.size(dim=3)

        out = self.norm(x)

        # reshape : (N,C,H,W) --> (N,H*W,C)
        out = out.permute(0, 2, 3, 1)
        out = out.view(-1, height*width, channel)

        # generate query/key/value
        q = self.q_layer(out)
        k = self.k_layer(out)
        v = self.v_layer(out)

        # apply multi-head attention
        out = mha_operation(q, k, v, num_heads=self.num_heads)

        # apply final linear
        out = self.output_linear(out)

        # reshape : (N,H*W,C) --> (N,C,H,W)
        out = out.view(-1, height, width, channel)
        out = out.permute(0, 3, 1, 2)

        # apply residual
        out = out + x

        return out


class CLS_ResnetAndAttention(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        num_heads,
        num_norm_groups, # 32
        timestep_embedding_dim, # 512
    ):
        super().__init__()

        self.resnet = CLS_ResnetBlock(
            in_channel,
            out_channel,
            num_norm_groups,
            timestep_embedding_dim,
        )
        self.attention = CLS_AttentionBlock(
            out_channel,
            num_heads,
            num_norm_groups,
        )

    def forward(self, x, t_emb):
        """
        Parameters
        ----------
        x : torch.tensor((batch_size, in_channel, width, height), dtype=float)
            input x
        t_emb : torch.tensor((batch_size, base_channel_dim * 4), dtype=float)
            timestep embeddings
        """
        out = self.resnet(x, t_emb)
        out = self.attention(out)
        return out

# Used in final pooling, as mentioned above.
# (See https://github.com/openai/CLIP/blob/main/clip/model.py)
class CLS_AttentionPool2d(nn.Module):
    def __init__(
        self,
        in_resolution, # tuple (height, width)
        channel,
        num_heads,
    ):
        super().__init__()

        self.height = in_resolution[0]
        self.width = in_resolution[1]
        self.num_heads = num_heads

        self.pos_embedding = nn.Parameter(
            torch.randn(self.height*self.width + 1, channel) / channel ** 0.5
        )
        self.q_layer = nn.Linear(channel, channel)
        self.k_layer = nn.Linear(channel, channel)
        self.v_layer = nn.Linear(channel, channel)

    def forward(self, x):
        channel = x.size(dim=1)
        height = x.size(dim=2)
        width = x.size(dim=3)

        # reshape
        #   --> (batch_size, height*width, channel)
        out = x.permute(0, 2, 3, 1)
        out = out.view(-1, height*width, channel)

        # apply QKV projection
        mean = torch.mean(out, dim=1, keepdim=True)  # (batch_size, 1, channel)
        out = torch.cat([mean, out], dim=1)          # (batch_size, height*width+1, channel)
        out = out + self.pos_embedding[None,:,:]     # (batch_size, height*width+1, channel)
        q = self.q_layer(out[:,:1,:])                # (batch_size, 1, channel)
        k = self.k_layer(out)                        # (batch_size, height*width+1, channel)
        v = self.v_layer(out)                        # (batch_size, height*width+1, channel)

        # apply multi-head attention
        #   --> (batch_size, 1, channel)
        out = mha_operation(q, k, v, num_heads=self.num_heads)

        #   --> (batch_size, channel)
        return out.squeeze(dim=1)

