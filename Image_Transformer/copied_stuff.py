#Reverse Engineering Stable Diffusion from https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py


print('starting to run code')

import sys
sys.path.append('/home/ciaran-mcdj/.local/share/pipx/venvs')
import torch





a = torch.randn(2)
print(a)

#
@torch.no_grad()
def p_sample_loop(self, shape, return_intermediates=False):
    device = self.betas.device
    b = shape[0]
    img = torch.randn(shape, device=device)
    intermediates = [img]
    for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
        img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                            clip_denoised=self.clip_denoised)
        if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
            intermediates.append(img)
    if return_intermediates:
        return img, intermediates
    return img


##
@torch.no_grad()
def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
    noise = noise_like(x.shape, device, repeat_noise)
    # no noise when t == 0
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

###
def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

###
def p_mean_variance(self, x, t, clip_denoised: bool):
    model_out = self.model(x, t) #TODO - THISSSSSS is thing we need to make
    if self.parameterization == "eps":
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
    elif self.parameterization == "x0":
        x_recon = model_out
    if clip_denoised:
        x_recon.clamp_(-1., 1.)

    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
    return model_mean, posterior_variance, posterior_log_variance

####
def predict_start_from_noise(self, x_t, t, noise):
    return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

####
def q_posterior(self, x_start, x_t, t):
    posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

#####
def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
