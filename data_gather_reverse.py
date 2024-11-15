"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time
import copy
import numpy as np
import torch as th
import torch.distributed as dist
from cm.image_datasets import load_data
from PIL import Image

from cm import dist_util, logger
from cm.script_util import (
    train_defaults,
    model_and_diffusion_defaults,
    cm_train_defaults,
    ctm_train_defaults,
    ctm_eval_defaults,
    ctm_loss_defaults,
    ctm_data_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from cm.random_util import get_generator
from cm.sample_util import karras_sample
import blobfile as bf
from torchvision.utils import make_grid, save_image
#import classifier_lib

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def main():
    args = create_argparser().parse_args()

    if args.use_MPI:
        dist_util.setup_dist(args.device_id)
    else:
        dist_util.setup_dist_without_MPI(args.device_id)

    logger.configure(args, dir=args.out_dir)

    logger.log("creating model and diffusion...")


    type_ = 'ode'
    model, diffusion = create_model_and_diffusion(args, type_=type_)

    if 'pkl' in args.model_path:
        import pickle
        with open(args.model_path, 'rb') as f:
            pretrained_ode = pickle.load(f)['ema']
        for dst_name, dst in model.named_parameters():
            for src_name, src in pretrained_ode.named_parameters():
                if dst_name == src_name:
                    dst.data.copy_(src.data)
                    break
        del pretrained_ode
    else:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location=dist_util.dev()), strict=True
        )

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size() * batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size
    data = load_data(
        args=args,
        data_name=args.data_name,
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        train_classes=args.train_classes,
        num_workers=args.num_workers,
        type=args.type,
        deterministic=args.deterministic,
        path_out=True,
        class_start=args.class_start,
        class_end=args.class_end,
        num_data_count=True,
        random_flip=True,
        random_crop=False,
        flip_ratio=args.flip_ratio,
    )
    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    elif args.sampler in ["exact", "gamma", "cm_multistep", "gamma_multistep"]:
        try:
            ts = tuple(int(x) for x in args.ts.split(","))
        except:
            ts = []
    else:
        ts = None

    #for ind_1 in range(1,18):
    #    for ind_2 in range(ind_1+1):
    #        print("ind_1, ind_2: ", ind_1, ind_2)
    if args.stochastic_seed:
        args.eval_seed = np.random.randint(1000000)
    #generator = get_generator(args.generator, args.num_samples, args.seed)
    generator = get_generator(args.generator, args.eval_num_samples, args.eval_seed)

    step = args.model_path.split('.')[-2][-6:]
    try:
        ema = float(args.model_path.split('_')[-2])
        assert ema in [0.999, 0.9999, 0.9999432189950708]
    except:
        ema = 'model'
    if args.sampler in ['multistep', 'exact', 'cm_multistep']:
        out_dir = os.path.join(args.out_dir, f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}')
    elif args.sampler in ["gamma"]:
        out_dir = os.path.join(args.out_dir, f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}_ind1_{args.ind_1}_ind2_{args.ind_2}')
    elif args.sampler in ["gamma_multistep"]:
        out_dir = os.path.join(args.out_dir,
                               f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{"".join([str(i) for i in ts])}_gamma_{args.gamma}')

    else:
        out_dir = os.path.join(args.out_dir,
                                f'{args.training_mode}_{args.sampler}_sampler_{args.sampling_steps}_steps_{step}_itrs_{ema}_ema_{args.rho}_rho')
    os.makedirs(out_dir, exist_ok=True)
    itr = 0
    eval_num_samples = 0
    num_sample = args.init_num_sample
    while itr < args.eval_num_samples:
        if args.reverse:
            assert args.stochastic_seed == False
            x_T, cond = next(data)
            x_T = x_T.to(dist_util.dev())
            t = args.sigma_min * th.zeros(x_T.shape[0], device=x_T.device)
            dims = x_T.ndim
            alpha_t, beta_t = diffusion.get_alpha_beta(t)
            x_T = append_dims(alpha_t, dims) * x_T + append_dims(beta_t, dims) * th.randn_like(x_T)
            print("data range: ", x_T.min(), x_T.max())
            args.clip_output = False
            args.clip_denoised = False
        else:
            x_T, cond = next(data)
            x_T = generator.randn(
                *(args.batch_size, args.in_channels, args.image_size, args.image_size),
                device=dist_util.dev()) * args.sigma_max
            #a = np.load('/data/NinthArticleExperimentalResults/ImageNet64/samples/reverse_True_v8/edm_heun_sampler_40_steps_64_ema_itrs_model_ema_7_rho/n02123045/sample.npz')
            #x_T = th.tensor(a['arr_0']).to(dist_util.dev()).permute(0,3,1,2)
            print("x_T scale: ", (x_T ** 2).mean())
        #classes = generator.randint(0, 1000, (args.batch_size,))
        #if args.large_log:
        print("x_T: ", x_T[0][0][0][0])
        current = time.time()
        model_kwargs = {}
        if args.class_cond:
            if args.train_classes >= 0:
                classes = th.ones(size=(args.batch_size,), device=dist_util.dev(), dtype=int) * int(args.train_classes)
            elif args.train_classes == -2:
                classes = [0, 1, 9, 11, 29, 31, 33, 55, 76, 89, 90, 130, 207, 250, 279, 281, 291, 323, 386, 387,
                           388, 417, 562, 614, 759, 789, 800, 812, 848, 933, 973, 980]
                assert args.batch_size % len(classes) == 0
                #print("!!!!!!!!!!!!!!: ", [x for x in classes for _ in range(args.batch_size // len(classes))])
                #model_kwargs["y"] = th.from_numpy(np.array([[[x] * (args.batch_size // len(classes)) for x in classes]]).reshape(-1)).to(dist_util.dev())
                classes = th.tensor([x for x in classes for _ in range(args.batch_size // len(classes))], device=dist_util.dev())
            else:
                classes = th.randint(
                    low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
                )
            model_kwargs["y"] = classes
            if args.large_log:
                print("classes: ", model_kwargs)
            #if args.reverse:
            model_kwargs["y"] = cond['y'].to(dist_util.dev())
            #else:
            #    model_kwargs["y"] = th.tensor(a['arr_1']).to(dist_util.dev())
            print("classes: ", model_kwargs["y"])
        with th.no_grad():
            x = karras_sample(
                diffusion=diffusion,
                model=model,
                shape=(args.batch_size, args.in_channels, args.image_size, args.image_size),
                steps=args.sampling_steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=False if args.data_name in ['church'] else True if args.training_mode=='edm' else args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=None,
                ts=ts,
                teacher = True if args.training_mode == 'edm' else False,
                clip_output=args.clip_output,
                ctm=True if args.training_mode.lower() == 'ctm' else False,
                x_T=x_T if args.stochastic_seed == False else None,
                ind_1=args.ind_1,
                ind_2=args.ind_2,
                gamma=args.gamma,
                reverse=args.reverse,
            )
        sample = x
        if not args.reverse:
            a = copy.deepcopy(x_T)
            x_T = ((x + 1) * 127.5).clamp(0, 255).to(th.uint8).cpu().detach()
            sample = a.cpu().detach()#.permute(0,2,3,1).numpy()
            #sample = sample.permute(0, 2, 3, 1)
            #sample = sample.contiguous()
        if args.reverse:
            x_T = ((x_T + 1) * 127.5).clamp(0, 255).to(th.uint8)

        if dist.get_rank() == 0:
            sample = sample.cpu().detach()
            x_T = x_T.cpu().detach()

            if args.large_log:
                print(f"{(itr-1) * args.batch_size} sampling complete...")
            print(x_T.shape, sample.shape)
            sample = sample.permute(0,2,3,1).numpy()
            print(sample.shape)
            print("sample scale: ", (sample ** 2).mean())
            x_T = x_T.permute(0,2,3,1).numpy()
            for i in range(sample.shape[0]):
                #if args.reverse:
                dir_ = cond['path'][i].split('/')[-2]
                filename = cond['path'][i].split('/')[-1].split('.')[0]
                #else:
                #    dir_ = 'random'
                #    filename = num_sample
                os.makedirs(os.path.join(out_dir, f"{dir_}"), exist_ok=True)
                if args.class_cond:
                    if args.reverse:
                        np.savez(os.path.join(out_dir, f"{dir_}/{filename}.npz"), sample[i])
                    else:
                        np.savez(os.path.join(out_dir, f"{dir_}/{filename}.npz"), sample[i], x_T[i])
                    #print(sample[i].permute(1,2,0).shape)
                    if args.save_format == 'png':
                        im = Image.fromarray(sample[i], "RGB")
                        im.save(os.path.join(out_dir, f"{dir_}/{filename}.png"))
                        im = Image.fromarray(x_T[i], "RGB")
                        im.save(os.path.join(out_dir, f"{dir_}/{filename}_img.png"))
                        #np.savez(os.path.join(out_dir, f"{dir_}/{filename}_img.npz"), x_T[i].numpy())
                #np.savez(os.path.join(out_dir, f"{dir_}/sample.npz"), sample,
                #                           classes.cpu().detach().numpy())
                num_sample += 1
                itr += 1
            #import sys
            #sys.exit()
            # if args.reverse:
            #     print("sample before normalization: ", (sample ** 2).mean())
            #     sample = sample / (sample ** 2).mean([1,2,3], keepdim=True).sqrt() * args.sigma_max
            #     concatenated_sample = th.cat((x_T, sample))
            #     print("sample after normalization: ", (sample ** 2).mean())
            # else:
            #     concatenated_sample = th.cat((sample, x_T))
            # concatenated_sample = concatenated_sample.reshape(2, -1, 3, args.image_size, args.image_size)
            # concatenated_sample = concatenated_sample.permute(1, 0, 2, 3, 4)
            # assert concatenated_sample.shape[0] == x_T.shape[0]
            # for i, xz in enumerate(concatenated_sample):
            #     assert list(xz.shape) == [2,3,args.image_size,args.image_size]
            #     if args.class_cond:
            #         np.savez(os.path.join(out_dir, f"sample_{num_sample}.npz"), xz.numpy(),
            #                  classes[i].cpu().detach().numpy())
            #     else:
            #         np.savez(os.path.join(out_dir, f"sample_{num_sample}.npz"), xz.numpy())
            #     num_sample += 1
            # if args.save_format == 'png':# or itr == 1:
            #     r = np.random.randint(1000000)
            #     print("x range: ", x.min(), x.max())
            #     nrow = int(np.sqrt(sample.shape[0]))
            #     image_grid = make_grid(x_T / 255. if args.reverse else ( x + 1.) / 2., nrow, padding=2)
            #     if args.class_cond:
            #         with bf.BlobFile(os.path.join(out_dir, f"class_{args.train_classes}_sample_{r}.png"), "wb") as fout:
            #             save_image(image_grid, fout)
            #     else:
            #         with bf.BlobFile(os.path.join(out_dir, f"sample_{r}.png"), "wb") as fout:
            #             save_image(image_grid, fout)

        eval_num_samples += sample.shape[0]
        if args.large_log:
            print(f"sample {eval_num_samples} time {time.time() - current} sec")


    # dist.barrier()
    logger.log("sampling complete")

def create_argparser():

    defaults = dict(
        generator="determ",
        eval_batch=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        sampling_steps=40,
        model_path="",
        eval_seed=42,
        save_format='png',
        stochastic_seed=False,
        #data_name='cifar10',
        data_name='imagenet64',
        #schedule_sampler="lognormal",
        ind_1=0,
        ind_2=0,
        gamma=0.5,
        reverse=False,
        init_num_sample=0,
        class_start=-1,
        class_end=-1,
    )
    defaults.update(train_defaults(defaults['data_name']))
    defaults.update(model_and_diffusion_defaults(defaults['data_name']))
    defaults.update(cm_train_defaults(defaults['data_name']))
    defaults.update(ctm_train_defaults(defaults['data_name']))
    defaults.update(ctm_eval_defaults(defaults['data_name']))
    defaults.update(ctm_loss_defaults(defaults['data_name']))
    defaults.update(ctm_data_defaults(defaults['data_name']))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
