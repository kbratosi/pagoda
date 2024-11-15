"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.script_util import (
    train_defaults,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    ctm_train_defaults,
    ctm_eval_defaults,
    ctm_loss_defaults,
    ctm_data_defaults,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
from cm.train_util import CMTrainLoop
import torch.distributed as dist
import copy
import torch
import cm.enc_dec_lib as enc_dec_lib
import pickle
import numpy as np


def main():
    args = create_argparser().parse_args()
    if args.use_MPI:
        dist_util.setup_dist(args.device_id)
        #dist_util.setup_dist_guided_diffusion()
    else:
        dist_util.setup_dist_without_MPI(args.device_id, args.port)

    logger.configure(args, dir=args.out_dir)

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
        num_data=-1,
        z_no_flip_dir=args.z_no_flip_dir,
        z_flip_dir=args.z_flip_dir,
        training_mode=args.training_mode,
        proportion=args.data_proportion,
    )
    data_for_GAN = load_data(
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
        num_data=-1,
        z_no_flip_dir=args.z_no_flip_dir,
        z_flip_dir=args.z_flip_dir,
        training_mode=args.training_mode,
        proportion=1.,
    )

    logger.log("creating model and diffusion...")

    # Load Feature Extractor
    feature_extractor = enc_dec_lib.load_feature_extractor(args, eval=True)
    # Load Discriminator
    decoder_discriminator, discriminator_feature_extractor = enc_dec_lib.load_discriminator_and_d_feature_extractor(args.image_size,
                                    args.discriminator_use_fp16, args.discriminator_class_cond,
                                load_feature=args.decoder_discriminator_training, load_discriminator=args.decoder_discriminator_training)
    recon_discriminator, _ = enc_dec_lib.load_discriminator_and_d_feature_extractor(args.image_size,
                                    args.discriminator_use_fp16, args.discriminator_class_cond, load_feature=False, load_discriminator=args.recon_discriminator)

    # Load Model
    logger.log(f"loading the teacher model from {args.teacher_model_path}")
    if args.decoder_override or args.load_ode:
        ode, diffusion_ = create_model_and_diffusion(args, type_='ode')
        if args.teacher_model_path.split('.')[-1] == 'pkl':
            if args.decoder_style == 'unet':
                with open(args.teacher_model_path, 'rb') as f:
                    ode = pickle.load(f)['ema']
            elif args.decoder_style == 'stylegan':
                with open(args.teacher_model_path, 'rb') as f:
                    ode = pickle.load(f)['G_ema']
            else:
                raise NotImplementedError
            # for dst_name, dst in ode.named_parameters():
            #     for src_name, src in pretrained_ode.named_parameters():
            #         if dst_name == src_name:
            #             dst.data.copy_(src.data)
            #             break
            # del pretrained_ode
        else:
            if args.map_location == 'cuda':
                #state_dict = torch.load(args.teacher_model_path, map_location=dist_util.dev())  # "cpu")
                state_dict = torch.load(args.teacher_model_path, map_location="cpu")
            else:
                state_dict = dist_util.load_state_dict(
                    args.teacher_model_path, map_location='cpu',  # dist_util.dev()
                )
            ode.load_state_dict(state_dict, strict=True)

            #ode.to(dist_util.dev())
            ode.train()
            #ode.eval()
            if args.use_fp16:
                ode.convert_to_fp16()
            print("ode load end")
    else:
        ode = None

    if args.decoder_training:
        decoder, diffusion = create_model_and_diffusion(args, feature_extractor, discriminator_feature_extractor, type_='decoder')
        # if dist.get_rank() == 0:
        #     for name, params in decoder.named_parameters():
        #        print(name)
        #     print(decoder)

        decoder.to(dist_util.dev())
        decoder.train()
        if args.use_fp16:
            decoder.convert_to_fp16()

        if dist.get_rank() == 0:
            for dst_name, dst in decoder.named_parameters():
                print("decoder: ", dst_name, dst.shape)

        if args.decoder_override:# and (args.decoder_model_channels == 128 and args.decoder_channel_mult == '2,2,2' and args.decoder_num_blocks == 4):
            if dist.get_rank() == 0:
                for src_name, src in ode.named_parameters():
                    print("ode: ", src_name, src.shape)
            if args.decoder_style == 'unet':
                if args.superres:
                    for dst_name, dst in decoder.named_parameters():
                        for src_name, src in ode.named_parameters():
                            if dst_name == src_name and dst.data.shape == src.data.shape:
                                if dist.get_rank() == 0:
                                    print("copied layer: ", dst_name)
                                dst.data.copy_(src.data)
                                break
                    if args.new_arch == 'only_dec':
                        for dst_name, dst in decoder.named_parameters():
                            for src_name, src in ode.named_parameters():
                                if dst_name == src_name and dst.data.shape != src.data.shape:
                                    if dist.get_rank() == 0:
                                        print("new layer: ", dst_name)
                                        print(dst.data.shape, src.data.shape)
                                        #print(dst.data[:10])
                                    if src.data.ndim == 1:
                                        dst.data[:src.data.shape[0]] = src.data
                                    elif src.data.ndim == 2:
                                        dst.data[:src.data.shape[0], :src.data.shape[1]] = src.data
                                    elif src.data.ndim == 3:
                                        dst.data[:src.data.shape[0], :src.data.shape[1], :src.data.shape[2]] = src.data
                                    elif src.data.ndim == 4:
                                        dst.data[:src.data.shape[0], :src.data.shape[1],
                                                :src.data.shape[2], :src.data.shape[3]] = src.data
                                    break
                        #import sys
                        #sys.exit()
                    #import sys
                    #sys.exit()
                    if args.progressive:
                        for src_name, src in ode.named_parameters():
                            if src_name.split('.')[0] == 'output_blocks':
                                src_layer = int(src_name.split('.')[1])
                        for dst_name, dst in decoder.named_parameters():
                            if dst_name.split('.')[0] == 'output_blocks':
                                dst_layer = int(dst_name.split('.')[1])
                                if dst_layer < args.activate_from:
                                    dst.requires_grad = False
                            elif dst_name.split('.')[0] == 'out':
                                pass
                            else:
                                dst.requires_grad = False
                        if dist.get_rank() == 0:
                            for dst_name, dst in decoder.named_parameters():
                                if dst.requires_grad:
                                    print("requires_grad True: ", dst_name)
                        #import sys
                        #sys.exit()

                if args.lowerres:
                    for dst_name, dst in decoder.named_parameters():
                        for src_name, src in ode.named_parameters():
                            if 'input_blocks' in dst_name:
                                dst_layer = int(dst_name.split('.')[1])
                                diff = int((args.num_res_blocks + 1) * np.log2(args.image_size // args.input_size))
                                diff = 4
                                dst_name_ = dst_name
                                if dst_layer > 0:
                                    dst_name_ = dst_name.split('.')
                                    dst_name_[1] = str(dst_layer + diff)
                                    #print("before: ", dst_name, dst.data.shape, src_name, src.data.shape)
                                    dst_name_ = '.'.join(dst_name_)
                                    #print("after: " ,dst_name_)
                                if dst_name_ == src_name and dst.data.shape == src.data.shape:
                                    if dist.get_rank() == 0:
                                        print(f"copied {src_name} source layer to {dst_name} target layer")
                                    dst.data.copy_(src.data)
                                    break
                if not args.superres and not args.lowerres:
                    decoder = copy.deepcopy(ode)
                    decoder.to(dist_util.dev())
                    decoder.train()
                    if args.use_fp16:
                        decoder.convert_to_fp16()
            elif args.decoder_style == 'stylegan':
                for dst_name, dst in decoder.named_parameters():
                    for src_name, src in ode.named_parameters():
                        if dst_name.split('.')[0] == 'synthesis':
                            if dst_name == src_name and dst.data.shape == src.data.shape:
                                dst.data.copy_(src.data)
                                if dist.get_rank() == 0:
                                    print("copied layer: ", dst_name)
                        if dst_name.split('.')[0] == 'mapping' and dst_name.split('.')[1][:2] == 'fc':
                            if src_name.split('.')[0] == 'mapping' and src_name.split('.')[1][:2] == 'fc':
                                dst_idx = int(dst_name.split('.')[1][2:])
                                src_idx = int(src_name.split('.')[1][2:])
                                if dst_idx == src_idx + args.num_init_layers and dst.data.shape == src.data.shape:
                                    dst.data.copy_(src.data)
                                    if dist.get_rank() == 0:
                                        print("copied layer: ", dst_name)
                        if dst_name.split('.')[0] == 'mapping' and 'embed' in dst_name.split('.')[1]:
                            if dst_name == src_name and dst.data.shape == src.data.shape:
                                dst.data.copy_(src.data)
                                if dist.get_rank() == 0:
                                    print("copied layer: ", dst_name)
            elif args.decoder_style == 'ldm':
                decoder = copy.deepcopy(ode)
                decoder.to(dist_util.dev())
                decoder.train()
                if args.use_fp16:
                    decoder.convert_to_fp16()
            else:
                raise NotImplementedError

        def count_parameters(model, requires_grad=True):
            return sum(p.numel() for p in model.parameters() if p.requires_grad == requires_grad)
        if args.decoder_override:
            logger.log("Teacher number of parameters: ", count_parameters(ode, False))
            logger.log("Teacher number of parameters: ", count_parameters(ode, True))
        if args.training_mode == 'pgd_encoder':
            state_dict = torch.load(args.decoder_model_path, map_location="cpu")
            ode.load_state_dict(state_dict, strict=True)
            ode.to(dist_util.dev())
            ode.eval()
            if args.use_fp16:
                ode.convert_to_fp16()
        else:
            if args.load_encoder:
                del ode
                ode, diffusion_ = create_model_and_diffusion(args, type_='encoder')

                if args.map_location == 'cuda':
                    # state_dict = torch.load(args.teacher_model_path, map_location=dist_util.dev())  # "cpu")
                    state_dict = torch.load(args.encoder_model_path, map_location="cpu")
                else:
                    state_dict = dist_util.load_state_dict(
                        args.encoder_model_path, map_location='cpu',  # dist_util.dev()
                    )
                ode.load_state_dict(state_dict, strict=True)

                ode.to(dist_util.dev())
                ode.eval()
                # ode.eval()
                if args.use_fp16:
                    ode.convert_to_fp16()
                if dist.get_rank() == 0:
                    for src_name, src in ode.named_parameters():
                        print("encoder: ", src_name, src.shape)
                print("ode load end")
            else:
                del ode
                ode = None
        logger.log("Decoder number of parameters: ", count_parameters(decoder))
        # import sys
        # sys.exit()
    else:
        decoder = None

    logger.log("training...")
    CMTrainLoop(
        decoder=decoder,
        ode=ode,
        recon_discriminator=recon_discriminator,
        decoder_discriminator=decoder_discriminator,
        diffusion=diffusion,
        data=data,
        data_for_GAN=data_for_GAN,
        batch_size=batch_size,
        args=args,
    ).run_loop()

def create_argparser():
    #defaults = dict(data_name='cifar10')
    defaults = dict(data_name='imagenet64')
    #defaults = dict(data_name='afhq')
    defaults.update(train_defaults(defaults['data_name']))
    defaults.update(model_and_diffusion_defaults(defaults['data_name']))
    defaults.update(cm_train_defaults(defaults['data_name']))
    defaults.update(ctm_train_defaults(defaults['data_name']))
    defaults.update(ctm_eval_defaults(defaults['data_name']))
    defaults.update(ctm_loss_defaults(defaults['data_name']))
    defaults.update(ctm_data_defaults(defaults['data_name']))
    defaults.update()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
