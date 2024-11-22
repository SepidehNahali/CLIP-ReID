from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1 import do_train_stage1
from processor.processor_clipreid_stage2 import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str)
    
##################################################################################################################### CHANGED!!
    parser.add_argument("--run_stage1", action="store_true", help="Run Stage 1 Training")
    parser.add_argument("--run_stage2", action="store_true", help="Run Stage 2 Training")
    parser.add_argument("--stage1_checkpoint", default="stage1_checkpoint.pth", type=str,help="Path to save/load Stage 1 checkpoint")
        
##################################################################################################################### CHANGED!!

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
###############################################################################################################changed!
    #output_dir = cfg.OUTPUT_DIR
    output_dir = "/kaggle/working/output/"
###############################################################################################################changed!

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)

##############################################################################################################Changed!
    # Run Stage 1
    if args.run_stage1:
        logger.info("Starting Stage 1 Training...")
        optimizer_1stage = make_optimizer_1stage(cfg, model)
        scheduler_1stage = create_scheduler(
            optimizer_1stage,
            num_epochs=cfg.SOLVER.STAGE1.MAX_EPOCHS,
            lr_min=cfg.SOLVER.STAGE1.LR_MIN,
            warmup_lr_init=cfg.SOLVER.STAGE1.WARMUP_LR_INIT,
            warmup_t=cfg.SOLVER.STAGE1.WARMUP_EPOCHS,
            noise_range=None
        )
        do_train_stage1(cfg, model, train_loader_stage1, optimizer_1stage, scheduler_1stage, args.local_rank)

        # Save Stage 1 Checkpoint
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, args.stage1_checkpoint)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Stage 1 Checkpoint saved at {checkpoint_path}")

    # Run Stage 2
    if args.run_stage2:
        logger.info("Starting Stage 2 Training...")
        # Load Stage 1 Checkpoint
        checkpoint_path = os.path.join(cfg.OUTPUT_DIR, args.stage1_checkpoint)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
            logger.info(f"Loaded Stage 1 Checkpoint from {checkpoint_path}")
        else:
            logger.error(f"Stage 1 checkpoint not found at {checkpoint_path}. Exiting...")
            exit(1)

    # optimizer_1stage = make_optimizer_1stage(cfg, model)
    # scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN, \
    #                     warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)

    # do_train_stage1(
    #     cfg,
    #     model,
    #     train_loader_stage1,
    #     optimizer_1stage,
    #     scheduler_1stage,
    #     args.local_rank
    # )

    ######################################################################################################Changed!

    
        loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
        optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg, model, center_criterion)
        scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg.SOLVER.STAGE2.STEPS, cfg.SOLVER.STAGE2.GAMMA, cfg.SOLVER.STAGE2.WARMUP_FACTOR,
                                      cfg.SOLVER.STAGE2.WARMUP_ITERS, cfg.SOLVER.STAGE2.WARMUP_METHOD)

        do_train_stage2(
            cfg,
            model,
            center_criterion,
            train_loader_stage2,
            val_loader,
            optimizer_2stage,
            optimizer_center_2stage,
            scheduler_2stage,
            loss_func,
            num_query, args.local_rank
        )
