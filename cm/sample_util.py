import torch as th
import numpy as np
from .random_util import get_generator
from .nn import append_dims, append_zero
import cm.dist_util as dist_util
import blobfile as bf
from torchvision.utils import make_grid, save_image

def get_t(ind):
    rho = 7.
    sigma_max = 80.
    sigma_min = 0.002
    start_scales = 18
    t = sigma_max ** (1 / rho) + ind / (start_scales - 1) * (
            sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    )
    t = t ** rho
    return t

def karras_sample(
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
    x_T=None,
    ctm=False,
    teacher=False,
    clip_output=True,
    train=False,
    ind_1=0,
    ind_2=0,
    gamma=0.5,
    reverse=False,
    dwt=False,
):
    if generator is None:
        generator = get_generator("dummy", dwt=dwt)

    if sampler in ["progdist", 'euler', 'exact', 'cm_multistep', 'gamma_multistep']:
        sigmas = get_sigmas_karras(steps + 1, sigma_min, sigma_max, rho, device=device)
    else:
        if sampler in ['heun_reverse']:
            sig = th.range(0,steps).to(device) * 60. / steps + 20.
            sigmas = th.sort(th.cat((get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device), sig)), descending=True).values
        else:
            sigmas = get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    if x_T == None:
        x_T = generator.randn(*shape, device=device) * sigma_max

    sample_fn = {
        "heun": sample_heun,
        "heun_reverse": sample_heun_reverse,
        "dpm": sample_dpm,
        "ancestral": sample_euler_ancestral,
        "onestep": sample_onestep,
        "exact": sample_exact,
        "gamma": sample_gamma,
        "gamma_multistep": sample_gamma_multistep,
        "progdist": sample_progdist,
        "euler": sample_euler,
        "multistep": stochastic_iterative_sampler,
        "cm_multistep": sample_multistep,
    }[sampler]

    if sampler in ["heun", "dpm"]:
        sampler_args = dict(
            s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise
        )
    elif sampler in ["multistep", "exact", "cm_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps
        )
    elif sampler in ["gamma"]:
        sampler_args = dict(ind_1=ind_1, ind_2=ind_2)
    elif sampler in ["gamma_multistep"]:
        sampler_args = dict(
            ts=ts, t_min=sigma_min, t_max=sigma_max, rho=rho, steps=steps, gamma=gamma,
        )
    else:
        sampler_args = {}
    if sampler in ['heun']:
        sampler_args['teacher'] = False if train else teacher
        sampler_args['ctm'] = ctm
    if sampler in ['heun', 'euler', 'heun_reverse']:
        sampler_args['diffusion'] = diffusion
        sampler_args['reverse'] = reverse
    #print("clip_denoised, clip_output: ", clip_denoised, clip_output)
    def denoiser(x_t, t, s=th.ones(x_T.shape[0], device=device)):
        if sampler in ['exact', 'cm_multistep', 'onestep', 'gamma', 'gamma_multistep']:
            denoised = diffusion.get_T_to_0(model, x_t, t, **model_kwargs)
        else:
            denoised = diffusion.get_denoiser(model, x_t, t, **model_kwargs)
        if clip_denoised:
            #print("clip denoised!!!")
            denoised = denoised.clamp(-1, 1)
        return denoised

    x_0 = sample_fn(
        denoiser,
        x_T,
        sigmas,
        generator,
        progress=progress,
        callback=callback,
        **sampler_args,
    )
    print("x_0 scale: ", (x_0 ** 2).mean())
    if clip_output:
        #print("clip output")
        return x_0.clamp(-1, 1)
    return x_0


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = th.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def to_d(x, sigma, denoised, diffusion=None):
    """Converts a denoiser output to a Karras ODE derivative."""
    if diffusion == None:
        return (x - denoised) / append_dims(sigma, x.ndim)
    dlogalpha, dlogbeta = diffusion.get_diff_alpha_beta(sigma)
    alpha, _ = diffusion.get_alpha_beta(sigma)
    d = append_dims(dlogbeta, x.ndim) * x + append_dims(dlogalpha - dlogbeta, x.ndim) * append_dims(alpha, x.ndim) * denoised
    return d



def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (
        sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2
    ) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


@th.no_grad()
def sample_euler_ancestral(model, x, sigmas, generator, progress=False, callback=None):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigmas[i],
                    "denoised": denoised,
                }
            )
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + generator.randn_like(x) * sigma_up
    return x


@th.no_grad()
def sample_midpoint_ancestral(model, x, ts, generator, progress=False, callback=None):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        from tqdm.auto import tqdm

        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
        if callback is not None:
            callback({"x": x, "tn": tn, "dn": dn, "dn_2": dn_2})
    return x



@th.no_grad()
def sample_multistep(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)


    for i in indices[:-1]:
        sigma = sigmas[i]
        print(i, sigma, sigmas[i+1])
        #print(0.002 * s_in)
        denoised = denoiser(x, sigma * s_in, s=0.002 * s_in)
        if i < len(indices) - 2:
            print(th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2).item())
            x = denoised + th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2) * th.randn_like(denoised)
        else:
            x = denoised

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x


@th.no_grad()
def sample_multistep(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)


    for i in indices[:-1]:
        sigma = sigmas[i]
        print(i, sigma, sigmas[i+1])
        #print(0.002 * s_in)
        denoised = denoiser(x, sigma * s_in, s=0.002 * s_in)
        if i < len(indices) - 2:
            print(th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2).item())
            x = denoised + th.sqrt(sigmas[i+1] ** 2 - 0.002 ** 2) * th.randn_like(denoised)
        else:
            x = denoised

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x


@th.no_grad()
def sample_exact(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)


    for i in indices[:-1]:
        sigma = sigmas[i]
        print(sigma, sigmas[i+1])
        if sigmas[i+1] != 0:
            denoised = denoiser(x, sigma * s_in, s=sigmas[i + 1] * s_in)
            x = denoised
        else:
            denoised = denoiser(x, sigma * s_in, s=sigma * s_in)
            d = to_d(x, sigma, denoised)
            dt = sigmas[i + 1] - sigma
            x = x + d * dt
        #else:
        #    denoised = denoiser(x, sigma * s_in)

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        #x = denoised
    return x


@th.no_grad()
def sample_gamma_multistep(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    gamma=0.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    if ts != [] and ts != None:
        sigmas = []
        t_max_rho = t_max ** (1 / rho)
        t_min_rho = t_min ** (1 / rho)
        s_in = x.new_ones([x.shape[0]])

        for i in range(len(ts)):
            sigmas.append((t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho)
        sigmas = th.tensor(sigmas)
        sigmas = append_zero(sigmas).to(x.device)
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    assert gamma != 0.0 and gamma != 1.0
    for i in indices[:-1]:
        sigma = sigmas[i]
        print(sigma, sigmas[i+1], gamma)
        s = (np.sqrt(1. - gamma ** 2) * (sigmas[i + 1] - 0.002) + 0.002)
        denoised = denoiser(x, sigma * s_in, s=s * s_in)
        if i < len(indices) - 2:
            std = th.sqrt(sigmas[i + 1] ** 2 - s ** 2)
            x = denoised + std * th.randn_like(denoised)
        else:
            x = denoised

    return x


@th.no_grad()
def sample_gamma(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    ts=[],
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    ind_1=0,
    ind_2=0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    assert ind_1 >= ind_2
    #print(f"{get_t(0)} -> {get_t(ind_1)} -> {get_t(ind_2)} -> {get_t(17)}")
    ones = th.ones(x.shape[0], device=x.device)
    denoised = denoiser(x, get_t(0) * ones, s=get_t(ind_1) * ones)
    denoised = denoised + th.randn_like(denoised) * np.sqrt(get_t(ind_2) ** 2 - get_t(ind_1) ** 2)
    x = denoiser(denoised, get_t(ind_2) * ones, s=get_t(17) * ones)

    return x


@th.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    teacher=False,
    ctm=False,
    diffusion=None,
    reverse=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if reverse:
        indices = indices[::-1]
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        print("sigmas: ", sigmas[i], sigmas[i+1], sigmas[i-1], ctm, teacher, (x ** 2).mean())
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if ctm:
            denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        else:
            #if teacher:
            denoised = denoiser(x, sigma_hat * s_in, s=None)
            #else:
            #    denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        #print("denoised: ", denoised[0][0][0][:3])
        d = to_d(x, sigma_hat, denoised, diffusion=diffusion)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        if reverse:
            dt = sigmas[i - 1] - sigma_hat
        else:
            dt = sigmas[i + 1] - sigma_hat
        if reverse:
            if sigmas[i - 1] == 0:
                print("final: ", (x ** 2).mean())
                return x
            else:
                # Heun's method
                x_2 = x + d * dt
                if ctm:
                    denoised_2 = denoiser(x_2, sigmas[i - 1] * s_in, s=sigmas[i - 1] * s_in)
                else:
                    #if teacher:
                    denoised_2 = denoiser(x_2, sigmas[i - 1] * s_in, s=None)
                    #else:
                    #    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
                d_2 = to_d(x_2, sigmas[i - 1], denoised_2, diffusion=diffusion)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        else:
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
                #print("last")
            else:
                #print("no last")
                # Heun's method
                x_2 = x + d * dt
                if ctm:
                    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
                else:
                    #if teacher:
                    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=None)
                    #else:
                    #    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
                d_2 = to_d(x_2, sigmas[i + 1], denoised_2, diffusion=diffusion)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    print("final: ", (x ** 2).mean())
    return x


@th.no_grad()
def sample_heun_reverse(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    teacher=False,
    ctm=False,
    diffusion=None,
    reverse=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if reverse:
        indices = indices[::-1]
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        print("sigmas: ", sigmas[i], sigmas[i+1], sigmas[i-1], ctm, teacher, (x ** 2).mean())
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        if ctm:
            denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        else:
            #if teacher:
            denoised = denoiser(x, sigma_hat * s_in, s=None)
            #else:
            #    denoised = denoiser(x, sigma_hat * s_in, s=sigma_hat * s_in)
        #print("denoised: ", denoised[0][0][0][:3])
        d = to_d(x, sigma_hat, denoised, diffusion=diffusion)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        if reverse:
            dt = sigmas[i - 1] - sigma_hat
        else:
            dt = sigmas[i + 1] - sigma_hat
        if reverse:
            if sigmas[i - 1] == 0:
                print("final: ", (x ** 2).mean())
                return x
            else:
                # Heun's method
                x_2 = x + d * dt
                if ctm:
                    denoised_2 = denoiser(x_2, sigmas[i - 1] * s_in, s=sigmas[i - 1] * s_in)
                else:
                    #if teacher:
                    denoised_2 = denoiser(x_2, sigmas[i - 1] * s_in, s=None)
                    #else:
                    #    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
                d_2 = to_d(x_2, sigmas[i - 1], denoised_2, diffusion=diffusion)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        else:
            if sigmas[i + 1] == 0:
                # Euler method
                x = x + d * dt
                #print("last")
            else:
                #print("no last")
                # Heun's method
                x_2 = x + d * dt
                if ctm:
                    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
                else:
                    #if teacher:
                    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=None)
                    #else:
                    #    denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, s=sigmas[i + 1] * s_in)
                d_2 = to_d(x_2, sigmas[i + 1], denoised_2, diffusion=diffusion)
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
    print("final: ", (x ** 2).mean())
    return x


@th.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    diffusion=None,
    reverse=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if reverse:
        indices = indices[::-1]
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        print("sigmas: ", sigmas[i], (x ** 2).mean())
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised, diffusion=diffusion)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "denoised": denoised,
                }
            )
        if reverse:
            dt = sigmas[i - 1] - sigma
            if sigmas[i - 1] == 0.0:
                return x
        else:
            dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


@th.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    generator,
    progress=False,
    callback=None,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = generator.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                }
            )
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@th.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in, None)


@th.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    generator,
    ts,
    progress=False,
    callback=None,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in, None)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@th.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    generator=None,
    progress=False,
    callback=None,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        from tqdm.auto import tqdm

        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": i,
                    "sigma": sigma,
                    "denoised": denoised,
                }
            )
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x



def application_sample(
    images,
    diffusion,
    model,
    shape,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
    generator=None,
    ts=None,
    ctm=False,
    teacher=False,
    clip_output=True,
    train=False,
    ind_1=0,
    ind_2=0,
    gamma=0.5,
):
    if generator is None:
        generator = get_generator("dummy")

    sample_fn = {
        "colorization": iterative_colorization,
        "inpainting": iterative_inpainting,
        "inpainting_flip": iterative_inpainting,
        "superres": iterative_superres,
    }[sampler]

    def denoiser(x_t, t, s=th.ones(shape[0], device=device)):
        denoised, G_theta = diffusion.get_denoised_and_G(model, x_t, t, s, ctm, teacher, **model_kwargs)
        denoised = G_theta
        return denoised

    if 'inpainting' in sampler:
        sampler_args = {'flip': False if sampler == 'inpainting' else True}
    else:
        sampler_args = {}

    x_out, x_in = sample_fn(
        denoiser,
        images,
        images,
        ts=ts,
        generator=generator,
        gamma=gamma,
        **sampler_args
    )
    if clip_output:
        #print("clip output")
        return x_out.clamp(-1, 1), x_in.clamp(-1, 1)
    return x_out, x_in

@th.no_grad()
def iterative_colorization(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    gamma = 0.0,
):
    def obtain_orthogonal_matrix():
        vector = np.asarray([0.2989, 0.5870, 0.1140])
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(3)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)
    mask = th.zeros(*x.shape[1:], device=dist_util.dev())
    mask[0, ...] = 1.0

    def replacement(x0, x1):
        x0 = th.einsum("bchw,cd->bdhw", x0, Q)
        x1 = th.einsum("bchw,cd->bdhw", x1, Q)

        x_mix = x0 * mask + x1 * (1.0 - mask)
        x_mix = th.einsum("bdhw,cd->bchw", x_mix, Q)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    monochrome_images = replacement(images, th.zeros_like(images))

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in, s=None)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(monochrome_images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)
    colored_images = x
    return colored_images, monochrome_images


@th.no_grad()
def iterative_inpainting(
    distiller,
    reference_images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    gamma=0.0,
    flip=False,
):
    from PIL import Image, ImageDraw, ImageFont

    image_size = x.shape[-1]

    # create a blank image with a white background
    img = Image.new("RGB", (image_size, image_size), color="white")

    # get a drawing context for the image
    draw = ImageDraw.Draw(img)

    # load a font
    #font = ImageFont.truetype("Arial.ttf", 250)
    font = ImageFont.truetype("Arial.ttf", 500)

    # draw the letter "C" in black
    #draw.text((50, 0), "S", font=font, fill=(0, 0, 0))
    draw.rectangle((50,50, 200,200), fill=(0, 0, 0))

    # convert the image to a numpy array
    img_np = np.array(img)
    img_np = img_np.transpose(2, 0, 1)
    img_th = th.from_numpy(img_np).to(dist_util.dev())

    mask = th.zeros(*x.shape, device=dist_util.dev())
    #mask = mask.reshape(-1, 7, 3, image_size, image_size)
    mask = mask.reshape(-1, 1, 3, image_size, image_size)

    mask[::2, :, img_th > 0.5] = 1.0
    mask[1::2, :, img_th < 0.5] = 1.0
    mask = mask.reshape(-1, 3, image_size, image_size)

    def replacement(x0, x1, flip=False):
        if flip:
            x_mix = x0 * (1 - mask) + x1 * mask
        else:
            x_mix = x0 * mask + x1 * (1 - mask)
        return x_mix

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])
    masked_images = replacement(reference_images, -th.ones_like(reference_images), flip=flip)
    t = (t_max_rho + ts[0] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
    x = masked_images + t * generator.randn_like(x)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)

        if gamma == 1.0:
            try:
                x0 = distiller(x, t * s_in, s=0.002)
            except:
                x0 = distiller(x, t * s_in, s=None)
        else:
            s = (np.sqrt(1. - gamma ** 2) * (next_t - 0.002) + 0.002)
            x0 = distiller(x, t * s_in, s=s * s_in)
            print(t, next_t, s)

        if gamma == 1.0:
            x0 = replacement(masked_images, x0, flip=flip) # mask again
            x = x0 + generator.randn_like(x) * np.sqrt(next_t ** 2 - t_min ** 2)
        else:
            masked_images = replacement(reference_images + s * generator.randn_like(masked_images), -th.ones_like(reference_images), flip=flip)
            x0 = replacement(masked_images, x0, flip=flip) # mask again
            x = x0 + generator.randn_like(x) * next_t * gamma
    inpainted_images = x
    return inpainted_images, masked_images


@th.no_grad()
def iterative_superres(
    distiller,
    images,
    x,
    ts,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
    generator=None,
    gamma=0.0,
):
    patch_size = 8

    def obtain_orthogonal_matrix():
        vector = np.asarray([1] * patch_size**2)
        vector = vector / np.linalg.norm(vector)
        matrix = np.eye(patch_size**2)
        matrix[:, 0] = vector
        matrix = np.linalg.qr(matrix)[0]
        if np.sum(matrix[:, 0]) < 0:
            matrix = -matrix
        return matrix

    Q = th.from_numpy(obtain_orthogonal_matrix()).to(dist_util.dev()).to(th.float32)

    image_size = x.shape[-1]

    def replacement(x0, x1):
        x0_flatten = (
            x0.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x1_flatten = (
            x1.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x0 = th.einsum("bcnd,de->bcne", x0_flatten, Q)
        x1 = th.einsum("bcnd,de->bcne", x1_flatten, Q)
        x_mix = x0.new_zeros(x0.shape)
        x_mix[..., 0] = x0[..., 0]
        x_mix[..., 1:] = x1[..., 1:]
        x_mix = th.einsum("bcne,de->bcnd", x_mix, Q)
        x_mix = (
            x_mix.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )
        return x_mix

    def average_image_patches(x):
        x_flatten = (
            x.reshape(-1, 3, image_size, image_size)
            .reshape(
                -1,
                3,
                image_size // patch_size,
                patch_size,
                image_size // patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size**2 // patch_size**2, patch_size**2)
        )
        x_flatten[..., :] = x_flatten.mean(dim=-1, keepdim=True)
        return (
            x_flatten.reshape(
                -1,
                3,
                image_size // patch_size,
                image_size // patch_size,
                patch_size,
                patch_size,
            )
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(-1, 3, image_size, image_size)
        )

    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    low_res_images = average_image_patches(images)

    for i in range(len(ts) - 1):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in, s=None)
        x0 = th.clamp(x0, -1.0, 1.0)
        x0 = replacement(low_res_images, x0)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + generator.randn_like(x) * np.sqrt(next_t**2 - t_min**2)
    high_res_images = x
    return high_res_images, low_res_images
