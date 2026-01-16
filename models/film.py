import torch
import torch.nn as nn

class SafeFiLM(nn.Module):
    def __init__(
        self,
        ctx_dim: int,
        h_dim: int,
        hidden: int = 128,
        gamma_scale: float = 0.1,
        beta_scale: float = 0.1,
        detach_ctx: bool = True,
    ):
        super().__init__()
        self.detach_ctx = detach_ctx
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale

        self.ctx_net = nn.Sequential(
            nn.Linear(ctx_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * h_dim),
        )

        # Identity initialization
        nn.init.zeros_(self.ctx_net[-1].weight)
        nn.init.zeros_(self.ctx_net[-1].bias)

    def forward(self, h, ctx):
        """
        h:   (N, h_dim)
        ctx: (N, ctx_dim)
        """
        if ctx is None:
            return h

        if self.detach_ctx:
            ctx = ctx.detach()

        gb = self.ctx_net(ctx)
        g_raw, b_raw = torch.chunk(gb, 2, dim=-1)

        gamma = 1.0 + self.gamma_scale * torch.tanh(g_raw)
        beta  = self.beta_scale  * torch.tanh(b_raw)
        self.gamma = gamma.detach()
        self.beta = beta.detach()
        return gamma * h + beta
