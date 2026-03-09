import argparse
import json
import os
import shutil
import warnings

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from data.concept_map_dataset import ConceptMapDataset, VAL_PARTITION, TRAIN_PARTITION
from src.models.ViT import generate_relevance, get_image_with_relevance
from src.models.ViT import vit_base_patch16_224 as vit
from utils.logging import get_logger

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names.append("vit")

logger = get_logger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', metavar='DATA', help='path to dataset')
    parser.add_argument('--concept_maps_data', metavar='concept_maps_data', help='path to concept maps')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=3e-6, type=float, metavar='LR', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url for distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--save_interval', default=20, type=int, help='interval to save results.')
    parser.add_argument('--num_samples', default=3, type=int, help='number of samples per class for training')
    parser.add_argument('--multiprocessing-distributed', action='store_true')
    parser.add_argument('--lambda_align', default=0.8, type=float, help='influence of semantic alignment loss.')
    parser.add_argument('--lambda_acc', default=0.2, type=float, help='influence of accuracy loss.')
    parser.add_argument('--experiment_folder', default=None, type=str, help='path to folder to use for experiment.')
    parser.add_argument('--dilation', default=0, type=float, help='Use dilation on the concept maps.')
    parser.add_argument('--lambda_ctx', default=1.2, type=float, help='coefficient of loss for non-concept regions.')
    parser.add_argument('--lambda_concept', default=0.5, type=float, help='coefficient of loss for concept regions.')
    parser.add_argument('--num_classes', default=500, type=int, help='number of training classes.')
    parser.add_argument('--temperature', default=1, type=float, help='temperature for softmax.')
    parser.add_argument('--class_seed', default=None, type=int, help='seed to randomly shuffle classes.')

    return parser.parse_args()


def setup_experiment_folder(args):
    if args.experiment_folder is None:
        folder_name = (f'experiment/lr_{args.lr}_seg_{args.lambda_align}_acc_{args.lambda_acc}'
                       f'_bckg_{args.lambda_ctx}_fgd_{args.lambda_concept}')

        if args.temperature != 1: folder_name += f'_tempera_{args.temperature}'
        if args.batch_size != 8: folder_name += f'_bs_{args.batch_size}'
        if args.num_classes != 500: folder_name += f'_num_classes_{args.num_classes}'
        if args.num_samples != 3: folder_name += f'_num_samples_{args.num_samples}'
        if args.epochs != 150: folder_name += f'_num_epochs_{args.epochs}'
        if args.class_seed is not None: folder_name += f'_seed_{args.class_seed}'

        args.experiment_folder = folder_name

    if os.path.exists(args.experiment_folder):
        raise FileExistsError(f"Experiment path {args.experiment_folder} already exists!")

    os.makedirs(args.experiment_folder)
    os.makedirs(os.path.join(args.experiment_folder, 'train_samples'))
    os.makedirs(os.path.join(args.experiment_folder, 'val_samples'))

    with open(os.path.join(args.experiment_folder, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def main():
    args = parse_arguments()
    setup_experiment_folder(args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    best_loss = float('inf')

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ.get("RANK", 0))
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )

    print("=> creating model")
    model = vit(pretrained=True).cuda()
    model.train()
    print("done")

    model = setup_model_parallelism(model, ngpus_per_node, args)

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    if args.resume:
        best_loss = resume_from_checkpoint(model, optimizer, args)

    cudnn.benchmark = True

    train_loader, train_sampler = create_data_loader(
        args, partition=TRAIN_PARTITION, num_samples=args.num_samples, is_train=True
    )
    val_loader, _ = create_data_loader(
        args, partition=VAL_PARTITION, num_samples=1, is_train=False
    )

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)
        args.logger = logger

        train(train_loader, model, criterion, optimizer, epoch, args)
        current_loss = validate(val_loader, model, criterion, epoch, args)

        is_best = current_loss <= best_loss
        best_loss = min(current_loss, best_loss)

        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, folder=args.experiment_folder)


def setup_model_parallelism(model, ngpus_per_node, args):
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        return model

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        print("start")
        model = torch.nn.DataParallel(model).cuda()

    return model


def resume_from_checkpoint(model, optimizer, args):
    if not os.path.isfile(args.resume):
        print(f"=> no checkpoint found at '{args.resume}'")
        return float('inf')

    print(f"=> loading checkpoint '{args.resume}'")
    loc = f'cuda:{args.gpu}' if args.gpu is not None else None
    checkpoint = torch.load(args.resume, map_location=loc)

    args.start_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']

    if args.gpu is not None and isinstance(best_loss, torch.Tensor):
        best_loss = best_loss.to(args.gpu)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
    return best_loss


def create_data_loader(args, partition, num_samples, is_train):
    dataset = ConceptMapDataset(
        args.concept_maps_data,
        args.data,
        partition=partition,
        train_classes=args.num_classes,
        num_samples=num_samples,
        seed=args.class_seed
    )

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed and is_train else None
    batch_size = args.batch_size if is_train else 10
    shuffle = (sampler is None) if is_train else False

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler
    )
    return loader, sampler


def _compute_forward_and_losses(args, model, orig_model, images, masks, targets, criterion, bce_criterion, is_train):
    with torch.enable_grad():
        relevance = generate_relevance(model, images, index=targets)

    reverse_masks = masks.clone()
    reverse_masks[reverse_masks == 1] = -1
    reverse_masks[reverse_masks == 0] = 1
    reverse_masks[reverse_masks == -1] = 0

    contex_loss = bce_criterion(relevance * reverse_masks, torch.zeros_like(relevance))
    concept_loss = bce_criterion(relevance * masks, masks)

    alignment_loss = (args.lambda_ctx * contex_loss) + (args.lambda_concept * concept_loss)

    with torch.set_grad_enabled(is_train):
        output = model(images)

    with torch.no_grad():
        output_orig = orig_model(images)

    _, preds = output.topk(1, 1, True, True)
    preds = preds.flatten()

    if args.temperature != 1:
        output = output / args.temperature

    classification_loss = criterion(output, preds)
    total_loss = (args.lambda_align * alignment_loss) + (args.lambda_acc * classification_loss)

    return total_loss, alignment_loss, classification_loss, output, output_orig, relevance


def save_visualizations(args, orig_model, images, masks, relevance, targets, batch_idx, mode):
    folder = os.path.join(args.experiment_folder, f'{mode}_samples')

    with torch.enable_grad():
        orig_relevance = generate_relevance(orig_model, images, index=targets)

    for j in range(images.shape[0]):
        image_vis = get_image_with_relevance(images[j], torch.ones_like(images[j]))
        new_vis = get_image_with_relevance(images[j], relevance[j])
        old_vis = get_image_with_relevance(images[j], orig_relevance[j])
        gt_vis = get_image_with_relevance(images[j], masks[j])

        h_img = cv2.hconcat([image_vis, gt_vis, old_vis, new_vis])
        cv2.imwrite(os.path.join(folder, f'res_{batch_idx}_{j}.jpg'), h_img)


def _log_metrics(args, mode, epoch, step, num_batches, alignment_loss, classification_loss, loss,
                 acc1, acc5, acc1_orig, acc5_orig, progress):
    progress.display(step)
    global_step = epoch * num_batches + step

    args.logger.info(f'{mode}/alignment_loss', alignment_loss, global_step)
    args.logger.info(f'{mode}/classification_loss', classification_loss, global_step)
    args.logger.info(f'{mode}/tot_loss', loss, global_step)
    args.logger.info(f'{mode}/orig_top1', acc1_orig, global_step)
    args.logger.info(f'{mode}/top1', acc1, global_step)
    args.logger.info(f'{mode}/orig_top5', acc5_orig, global_step)
    args.logger.info(f'{mode}/top5', acc5, global_step)


def train(train_loader, model, criterion, optimizer, epoch, args):
    bce_criterion = nn.BCELoss(reduction='mean')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    orig_top1 = AverageMeter('Acc@1_orig', ':6.2f')
    orig_top5 = AverageMeter('Acc@5_orig', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [losses, top1, top5, orig_top1, orig_top5],
        prefix=f"Epoch: [{epoch}]"
    )

    orig_model = vit(pretrained=True).cuda()
    orig_model.eval()
    model.train()

    for i, (masks, images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            masks = masks.cuda(args.gpu, non_blocking=True)
            targets = targets.cuda(args.gpu, non_blocking=True)

        loss, seg_loss, cls_loss, output, output_orig, relevance = _compute_forward_and_losses(
            args, model, orig_model, images, masks, targets, criterion, bce_criterion, is_train=True
        )

        if i % args.save_interval == 0:
            save_visualizations(args, orig_model, images, masks, relevance, targets, i, mode='train')

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        acc1_orig, acc5_orig = accuracy(output_orig, targets, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        orig_top1.update(acc1_orig[0], images.size(0))
        orig_top5.update(acc5_orig[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            _log_metrics(args, 'train', epoch, i, len(train_loader), seg_loss, cls_loss, loss,
                         acc1, acc5, acc1_orig, acc5_orig, progress)


def validate(val_loader, model, criterion, epoch, args):
    bce_criterion = nn.BCELoss(reduction='mean')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    orig_top1 = AverageMeter('Acc@1_orig', ':6.2f')
    orig_top5 = AverageMeter('Acc@5_orig', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5, orig_top1, orig_top5],
        prefix=f"Epoch: [{epoch}]"
    )

    orig_model = vit(pretrained=True).cuda()
    orig_model.eval()
    model.eval()

    with torch.no_grad():
        for i, (masks, images, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                masks = masks.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)

            loss, seg_loss, cls_loss, output, output_orig, relevance = _compute_forward_and_losses(
                args, model, orig_model, images, masks, targets, criterion, bce_criterion, is_train=False
            )

            if i % args.save_interval == 0:
                save_visualizations(args, orig_model, images, masks, relevance, targets, i, mode='val')

            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            acc1_orig, acc5_orig = accuracy(output_orig, targets, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            orig_top1.update(acc1_orig[0], images.size(0))
            orig_top5.update(acc5_orig[0], images.size(0))

            if i % args.print_freq == 0:
                _log_metrics(args, 'val', epoch, i, len(val_loader), seg_loss, cls_loss, loss,
                             acc1, acc5, acc1_orig, acc5_orig, progress)

    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return losses.avg


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    filepath = os.path.join(folder, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(folder, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.85 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter:
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()