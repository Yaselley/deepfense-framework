import torch
from deepfense.utils.registry import register_optimizer


@register_optimizer("adam")
def AdamOptimizer(params, config):
    lr = config.get("lr", 1e-6)
    weight_decay = config.get("weight_decay", 1e-04)
    betas = config.get("betas", (0.9, 0.999))
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=betas)


@register_optimizer("adamw")
def AdamWOptimizer(params, config):
    lr = config.get("lr", 1e-6)
    weight_decay = config.get("weight_decay", 1e-04)
    betas = config.get("betas", (0.9, 0.999))
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


@register_optimizer("sgd")
def SGDOptimizer(params, config):
    lr = config.get("lr", 1e-6)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 1e-04)
    return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        base_optim_class = getattr(torch.optim, base_optimizer.__class__.__name__)
        self.base_optimizer = base_optim_class(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
