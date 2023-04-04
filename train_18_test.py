import argparse
import os
import pathlib
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
import yaml
from monai.data import decollate_batch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import csv

from config import get_config
from dataset.brats import get_datasets, get_datasets_train_rf_withvalid
from loss import EDiceLoss
from loss.dice import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch
from vtunet.vision_transformer import VTUNet as ViT_seg
from vtunet.Unet import Unet_missing

from torch.cuda.amp import autocast as autocast

from dataset.transforms import *
from vtunet.mask_utils import MaskEmbeeding1
from thop import profile
import thop

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description='VTUNET BRATS 2021 Training')
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--mdp', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--patch_shape', default=128, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
                    
parser.add_argument('--part', '--data_part', default=1, type=float, help='low shot, data proportion')
parser.add_argument('--all_data', default=False, type=bool, help='if train, when part != 1, do semi-supervise when finetune')
parser.add_argument('--semi_proportion', default=1, type=float, help='the proportion of the data used without label')
parser.add_argument('--use_weight', action='store_true', help='use zipped dataset instead of folder dataset')

parser.add_argument('--wd', '--weight-decay', default=0.00001, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--model_type', type=str, default='vtnet', choices=['vtnet', 'cnnnet'])
parser.add_argument('--feature_level', default=2, type=int, metavar='N', help='mini-batch size (default: 1)')

parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--exp_name', default='baseline3_real', type=str, help='exp name')

parser.add_argument('--val', default=20, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/vt_unet_costum.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=True, type=bool, help='resume from checkpoint')
parser.add_argument('--deep_supervised', default=False, type=bool, help='resume from checkpoint')

parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--weight_kl', default=1, type=float, help="train sub")

class LR_Scheduler(object):
    def __init__(self, base_lr, num_epochs, mode='poly'):
        self.mode = mode
        self.lr = base_lr
        self.num_epochs = num_epochs

    def __call__(self, optimizer, epoch):
        if self.mode == 'poly':
            now_lr = round(self.lr * np.power(1 - np.float32(epoch)/np.float32(self.num_epochs), 0.9), 8) 
        self._adjust_learning_rate(optimizer, now_lr)
        return now_lr

    def _adjust_learning_rate(self, optimizer, lr):
        optimizer.param_groups[0]['lr'] = lr

def KL_loss(pred1, pred2):
    
    
    sigmoid1 = torch.sigmoid(pred1)
    sigmoid2 = torch.sigmoid(pred2)
    
    kl_loss = F.kl_div(torch.log(sigmoid1), sigmoid2.detach(), reduction='none')
    
    entropy1 = -1 * sigmoid1.detach() * torch.log(sigmoid1.detach())
    entropy2 = -1 * sigmoid2.detach() * torch.log(sigmoid2.detach())
    
    #print(entropy1.max(), entropy1.min())
    
    weight = entropy1 - entropy2
    weight[weight < 0] = 0
    #print(weight.min(), weight.max())
    #print(kl_loss.mean())
    
    kl_loss = kl_loss * weight
    
    return kl_loss
  
def mse_loss(fine_pred, missing_pred):  
    loss1 = F.mse_loss(fine_pred[:, 0], missing_pred[:, 0], reduction='mean') 
    loss2 = F.mse_loss(fine_pred[:, 1], missing_pred[:, 1], reduction='mean') 
    loss3 = F.mse_loss(fine_pred[:, 2], missing_pred[:, 2], reduction='mean') 
    
    return loss1 + loss2 + loss3
    
def main(args):

    args.train_transforms = 'Compose([RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0),])'
    args.train_transforms = 'Compose([RandomIntensityChange((0.1,0.1)), RandomFlip(0),])'
    args.train_transforms = eval(args.train_transforms)
    
    
    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")

    args.save_folder_1 = pathlib.Path(f"/data5/lh/brats/runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"/data5/lh/brats/runs/{args.exp_name}/model_1")

    print(args)
    
    # Create model
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = get_config(args)
    if args.model_type == "vtnet":
        pass
    else:
        model_1 = Unet_missing(input_shape = [128,128,128], init_channels = 16, out_channels=3, mdp=3, pre_train = False, deep_supervised = args.deep_supervised, patch_shape = args.patch_shape)
        
        #args.checkpoint = "/data5/lh/brats/runs/smunet_pre2/model_1model_best_599.pth.tar"
        args.checkpoint = "/data5/lh/brats/runs/ours_pre_rfsplit/model_1model_best_599.pth.tar"
        args.checkpoint = "/apdcephfs/share_1290796/lh/brats/runs/our_pre_18_re_mdp3/model_1model_best_599.pth.tar"
        #args.checkpoint = "/data5/lh/brats/runs/ours_pre_rfsplit_noinversion/model_1model_best_599.pth.tar"
        #args.checkpoint = "/data5/lh/brats/runs/ours_pre_rfsplit_withmean/model_1model_best_599.pth.tar"
        #args.checkpoint = "/data5/lh/brats/runs/normnet_pre_mask9375/model_1model_best_599.pth.tar"
        #args.checkpoint = "/data5/lh/brats/runs/ours_pre_rfsplit_withoutreg/model_1model_best_599.pth.tar"
        #args.checkpoint = "/data5/lh/brats/runs/ours_pre_rfsplit_75/model_1model_best_599.pth.tar"
        #args.checkpoint = "/data5/lh/brats/runs/pre_for_20/model_1model_best_599.pth.tar"
        
        #args.checkpoint = "/data5/lh/brats/runs/modelg_init32/model_1model_best_599.pth.tar"
        
        ck = torch.load(args.checkpoint, map_location=torch.device('cpu'))
        del ck['state_dict']['module.unet.up1conv.weight']
        del ck['state_dict']['module.unet.up1conv.bias']
        #del ck['state_dict']['module.unet.decoder.seg_layer.weight']
        #del ck['state_dict']['module.unet.decoder.seg_layer.bias']
        
        model_1 = nn.DataParallel(model_1)
        
        model_1.load_state_dict(ck['state_dict'], strict=False)
        model_1 = model_1.module
        
        model_1 = nn.DataParallel(model_1)
        model_1 = model_1.cuda()
        #model_1.module.limage = model_1.module.limage.cpu()
        model_1.module.raw_input = model_1.module.raw_input.cpu()
    
    print(f"total number of trainable parameters {count_parameters(model_1)}")

    #model_1.load_state_dict(torch.load("/data5/lh/brats/runs/newbackbone2/model_1model_best_149.pth.tar")['state_dict'])

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    criterion = EDiceLoss().cuda()
    criterian_val = EDiceLoss_Val().cuda()
    CE_L = torch.nn.BCELoss(reduction = "none").cuda()
    metric = criterian_val.metric
    print(metric)
    params = model_1.parameters()
    
    limage = []
    ori_para = []
    
    for pname, p in model_1.named_parameters():
        if pname.endswith("limage"):
            limage.append(p)
            #ori_para.append(p)
        else:
            ori_para.append(p)

    optimizer = torch.optim.Adam(ori_para, lr=args.lr, weight_decay=args.weight_decay)
    #lr_schedule = LR_Scheduler(args.lr, args.epochs)
    
    if args.val != 0:
        full_train_dataset, l_val_dataset,_,_labeled_name_list = get_datasets_train_rf_withvalid(args.seed, fold_number=args.fold, part = args.part, all_data = args.all_data, patch_shape = args.patch_shape)
        print(_labeled_name_list)
    else:
        full_train_dataset, l_val_dataset, _labeled_name_list = get_datasets_train_rf_all(args.seed, fold_number=args.fold, part = args.part)
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=1, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    #bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)

    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))
    #print("Bench Test dataset number of batch:", len(bench_loader))

    # Actual Train loop
    best_1 = 0.0
    patients_perf = []

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    best_loss = 10000000
    
    print("start training now!")
        
    limage = model_1.module.limage.detach().cpu().numpy()
    raw_input = model_1.module.raw_input.cpu().numpy()
    #start_epoch = 150
    
    for epoch in range(args.epochs):
        # if epoch < start_epoch:
            # if scheduler is not None:
                # scheduler.step()
            # continue
        try:
        
            # step_lr = lr_schedule(optimizer, epoch)
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')
            kl_loss = AverageMeter('Loss', ':.4e')
            
            model_1.train()
            
            mode = "train" if model_1.training else "val"
            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_],
                prefix=f"{mode} Epoch: [{epoch}]")

            end = time.perf_counter()
            metrics = []

            for i, batch in enumerate(zip(train_loader)):
            
                # measure data loading time
                data_time.update(time.perf_counter() - end)
                if not batch[0]["patient_id"][0] in _labeled_name_list and (np.random.random() > args.semi_proportion): # or epoch < 50):
                    continue
                    
                inputs_S1, labels_S1 = batch[0]["image"].numpy(), batch[0]["label"].numpy()
                patch_locations = batch[0]["crop_indexes"]
                
                
                inputs_S1 = inputs_S1.repeat(args.batch_size, axis = 0) # double in batch channel
                print(inputs_S1.shape)
                
                for ci in range(args.batch_size):
                    mask = MaskEmbeeding1(1, raw_input = raw_input, mdp = args.mdp, mask = False, patch_shape = args.patch_shape)
                    #print(mask.shape)
                    #print(model_1.module.limage.shape)
                    inputs_S1[ci] = inputs_S1[ci] #* mask #+ limage[:,:,patch_locations[0][0]: patch_locations[0][1],patch_locations[1][0]: patch_locations[1][1],patch_locations[2][0]: patch_locations[2][1]] * (1-mask)# not 
                
                #print(inputs_S1.shape, labels_S1.shape)
                inputs_S1, labels_S1 = args.train_transforms([inputs_S1.transpose(0,2,3,4,1), labels_S1.transpose(0,2,3,4,1)])
                
                inputs_S1, labels_S1 = [torch.from_numpy(np.ascontiguousarray(x.transpose(0,4,1,2,3))).float() for x in [inputs_S1, labels_S1]]
                #print(inputs_S1.shape, labels_S1.shape)
                
                
                labels_S1 = labels_S1.repeat(args.batch_size,1,1,1,1)
                patch_locations = [[s.repeat(args.batch_size) for s in l] for l in patch_locations]
                
                

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

                optimizer.zero_grad()
                
                
                
                segs_S1, mask_sum, style, content = model_1(inputs_S1, patch_locations)#, fmdp = fmdp)
                
               
                #print(mask_sum)
                if not args.deep_supervised:
                    style.append(segs_S1)
                else:
                    style.append(segs_S1[-1])
                loss_ = 0
                #print(inputs_S1.shape, segs_S1.shape)
                #weight = [0.25,0.5, 1]
                 
                if not args.deep_supervised:
                    loss_ += criterion(segs_S1, labels_S1)

                    
                else:
                    for idxss, l in enumerate(segs_S1):
                        loss_ += criterion(l, labels_S1) #* weight[idxss]

                
                if args.batch_size == 2:
                    if args.use_weight and epoch > 50:
                        print("what???????????????")
                        pass
                    elif not args.use_weight:
                        print(style[args.feature_level][0].shape)
                        loss_kl = F.mse_loss(style[args.feature_level][0], style[args.feature_level][1], reduction = "none").mean()
                    else:
                        loss_kl = torch.tensor(0)
                elif args.batch_size == 3:
                    loss_kl = F.mse_loss(style[args.feature_level][0], style[args.feature_level][1])
                    loss_kl += F.mse_loss(style[args.feature_level][1], style[args.feature_level][2])
                    loss_kl += F.mse_loss(style[args.feature_level][0], style[args.feature_level][2])
                    loss_kl /= 6
                    
                # compute gradient and do SGD step
                #print(batch[0]["patient_id"])
                if batch[0]["patient_id"][0] in _labeled_name_list:
                    loss = loss_ + loss_kl * args.weight_kl
                    #print("aha")
                else:
                    loss = loss_kl * args.weight_kl
                    
                loss.backward()  #为了梯度放大
                optimizer.step()
                
                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                    kl_loss.update(loss_kl.item())
                else:
                    print("NaN in model loss!!")
                
                
                print(kl_loss.avg)
                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)
                #optimizer.step()

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)

               

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)
            if scheduler is not None:
                scheduler.step()
                    
            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            #torch.cuda.empty_cache()
            if epoch > args.epochs - 50:
                args.val = 1
            # Validate at the end of epoch every val step
            if epoch > args.epochs - 50:
                args.val = 1
            if args.val != 0:
                if (epoch + 1) % args.val == 0 and epoch > args.epochs-100:
                    total = 0
                    validation_loss_1, validation_dice = step(val_loader, model_1, criterian_val, metric, epoch, t_writer_1,
                                                                  save_folder=args.save_folder_1,
                                                                  patients_perf=patients_perf, mask_modal = [1], patch_shape = args.patch_shape)
                    total += validation_dice
                    print(validation_dice)
                    validation_loss_1, validation_dice = step(val_loader, model_1, criterian_val, metric, epoch, t_writer_1,
                                                                  save_folder=args.save_folder_1,
                                                                  patients_perf=patients_perf, mask_modal = [1,3], patch_shape = args.patch_shape)
                    total += validation_dice
                    print(validation_dice)
                    validation_loss_1, validation_dice = step(val_loader, model_1, criterian_val, metric, epoch, t_writer_1,
                                                                  save_folder=args.save_folder_1,
                                                                  patients_perf=patients_perf, mask_modal = [0,1,3], patch_shape = args.patch_shape)
                    total += validation_dice
                    print(validation_dice)
                    validation_loss_1, validation_dice = step(val_loader, model_1, criterian_val, metric, epoch, t_writer_1,
                                                                  save_folder=args.save_folder_1,
                                                                  patients_perf=patients_perf, patch_shape = args.patch_shape)

                    total += validation_dice
                    print(validation_dice)
                    t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)
                    t_writer_1.add_scalar(f"SummaryDice", validation_dice, epoch)

                    if total > best_1:
                        print(f"Saving the model with DSC {validation_dice}")
                        best_1 = total
                        model_dict = model_1.state_dict()
                        save_checkpoint(
                            dict(
                                epoch=epoch,
                                state_dict=model_dict,
                                optimizer=optimizer.state_dict(),
                                #scheduler=scheduler.state_dict(),
                            ),
                            save_folder=args.save_folder_1, )
                    
                    ts = time.perf_counter()
                    print(f"Val epoch done in {ts - te} s")
                if (epoch + 1) % 50 == 0:
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            #scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder_1, )
            else:
                if (epoch + 1) % 20 == 0:
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            #scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder_1, )
                elif losses_.avg < best_loss:
                    best_loss  = losses_.avg
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=1000,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            #scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder_1, )
                        

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break


def step(data_loader, model, criterion: EDiceLoss_Val, metric, epoch, writer, save_folder=None, patients_perf=None, mask_modal = [], patch_shape = 128):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []
    model.module.mask_modal = mask_modal
    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        patient_id = val_data["patient_id"]

        model.eval()
        with torch.no_grad():
            val_inputs, val_labels = (
                val_data["image"].cuda(),
                val_data["label"].cuda(),
            )
            val_outputs = inference(val_inputs, model, patch_shape = patch_shape)
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]

            segs = val_outputs
            targets = val_labels
            loss_ = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()
    
    #print(metrics)
    metrics = list(zip(*metrics))
    metrics = [np.nanmean(torch.tensor(dice, device="cpu").numpy()) for dice in metrics]
    labels = ("ET", "TC", "WT")
    #metrics = {key: value for key, value in zip(labels, metrics)}
    #print(metrics)
    
    return losses.avg, np.nanmean(metrics)#, metrics


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
