from datasets.data import get_PPI_training_set_opt, get_PPI_test_set_opt
from torch.utils.data import DataLoader
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
from networks.DPG import DPG
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
import numpy as np
import pandas as pd
from math import log10
from torch.nn import functional as F
import cv2
#设置使用GPU=0
os.environ['CUDA_VISIBLE_DEVICES'] ='0'

parser = argparse.ArgumentParser(description='PyTorch PPI network Training')
parser.add_argument("--batchSize", type=int, default=16, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=3000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default='checkpoint/pre-training/PPI_model_epoch_2000.pth', type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 1e-4")
parser.add_argument("--pretrained", default=None, type=str, help="path to pretrained model (default: none)")
parser.add_argument('--msfa_size', '-uf',  type=int, default=4, help="the size of square msfa")
parser.add_argument("--train_dir", default="train_data", type=str, help="path to train dataset")
parser.add_argument("--val_dir", default="test_data", type=str, help="path to validation dataset")


def Listtoarray(List):
    low_freq1 = np.array(List[0][0])
    high_freq1 = np.array(List[0][1])
    low_freq2 = np.array(List[1][0])
    high_freq2 = np.array(List[1][0])
    return low_freq1,high_freq1,low_freq2,high_freq2
def main():
    global opt, model
    opt = parser.parse_args()

    cuda = True
    opt.cuda = True
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    #设计随即数种子 这个每次随机生成的数值都是一样 网络的初始化参数都是一样的
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
   #设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法
    cudnn.benchmark = True
   
    print("===> Loading datasets")
    opt.norm_flag = False
    opt.augment_flag = False
    #读取数据
    train_set = get_PPI_training_set_opt(opt.train_dir, opt.msfa_size, opt.norm_flag, opt.augment_flag)
    test_set = get_PPI_test_set_opt(opt.val_dir, opt.msfa_size, opt.norm_flag)
    training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
    testing_data_loader = DataLoader(dataset=test_set, batch_size=24, shuffle=False)

    print("===> Building model")
    model = DPG()
    print("===> Setting GPU")
    if cuda:
        device_flag = torch.device('cuda')
        model = model.cuda()
    else:
        device_flag = torch.device('cpu')
        model = model.cpu()
    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            #加载预训练的的模型
            weights = torch.load(opt.pretrained, map_location=lambda storage, loc: storage)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    #model.parameters()储存的是模型的权
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    save_opt(opt)

    print("===> Training")
    print('# parameters:', sum(param.numel() for param in model.parameters()))  # 输出模型参数数量
    results = {'im_loss': [], 're_loss': [], 'all_loss': [], 'psnr': []}
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        running_results = train(training_data_loader, optimizer, model, epoch, opt.nEpochs)

        results['im_loss'].append(running_results['im_loss'] / running_results['batch_sizes'])
        results['re_loss'].append(running_results['re_loss'] / running_results['batch_sizes'])
        results['all_loss'].append(running_results['all_loss'] / running_results['batch_sizes'])
        test_results = test(testing_data_loader, optimizer, model, epoch, opt.nEpochs)
        results['psnr'].append(test_results['psnr'])
        if epoch % 100 == 0:
            save_checkpoint(model, epoch)
        if epoch != 0:
            save_statistics(opt, results, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.5 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, epoch, num_epochs):
    lr = adjust_learning_rate(optimizer, epoch - 1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    train_bar = tqdm(training_data_loader)
    running_results = {'batch_sizes': 0, 'im_loss': 0, 're_loss': 0, 'all_loss': 0}
    model.train()

    for batch in train_bar:
        input_raw, target_PPI = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        N, C, H, W = batch[0].size()
        running_results['batch_sizes'] = N

        if opt.cuda:
            input_raw = input_raw.cuda()
            target_PPI = target_PPI.cuda()

        estimated_PPI = model(input_raw)

        # wavelet loss
        filter1h = torch.Tensor([[[[1, -1]]]]).cuda()
        wavelet1h = F.conv2d(estimated_PPI - target_PPI, filter1h, padding='valid')
        score1h = torch.mean(torch.sum(wavelet1h ** 2, dim=tuple(range(1, estimated_PPI.dim()))))
        filter1v = torch.Tensor([[[[1], [-1]]]]).cuda()
        wavelet1v = F.conv2d(estimated_PPI - target_PPI, filter1v, padding='valid')
        score1v = torch.mean(torch.sum(wavelet1v ** 2, dim=tuple(range(1, estimated_PPI.dim()))))
        filter1d = torch.Tensor([[[[1 / 2, -1 / 2], [-1 / 2, 1 / 2]]]]).cuda()
        wavelet1d = F.conv2d(estimated_PPI - target_PPI, filter1d, padding='valid')
        score1d = torch.mean(torch.sum(wavelet1d ** 2, dim=tuple(range(1, estimated_PPI.dim()))))
        filter2h = torch.Tensor([[[[1 / 2, 1 / 2, -1 / 2, -1 / 2]]]]).cuda()
        wavelet2h = F.conv2d(estimated_PPI - target_PPI, filter2h, padding='valid')
        score2h = torch.mean(torch.sum(wavelet2h ** 2, dim=tuple(range(1, estimated_PPI.dim()))))
        filter2v = torch.Tensor([[[[1 / 2], [1 / 2], [-1 / 2], [-1 / 2]]]]).cuda()
        wavelet2v = F.conv2d(estimated_PPI - target_PPI, filter2v, padding='valid')
        score2v = torch.mean(torch.sum(wavelet2v ** 2, dim=tuple(range(1, estimated_PPI.dim()))))
        filter2d = torch.Tensor([[[[1 / 8, 1 / 8, -1 / 8, -1 / 8], [1 / 8, 1 / 8, -1 / 8, -1 / 8],
                                   [-1 / 8, -1 / 8, 1 / 8, 1 / 8], [-1 / 8, -1 / 8, 1 / 8, 1 / 8]]]]).cuda()
        wavelet2d = F.conv2d(estimated_PPI - target_PPI, filter2d, padding='valid')
        score2d = torch.mean(torch.sum(wavelet2d ** 2, dim=tuple(range(1, estimated_PPI.dim()))))
        wavelet_loss_PPI = (score1h + score1v + score1d + score2h + score2v + score2d)/6

        alpha = 1
        
        PPI_loss = torch.sum((estimated_PPI - target_PPI) ** 2, dim=tuple(range(1, estimated_PPI.dim())))
        PPI_loss = torch.mean(PPI_loss)
        loss_x4 = PPI_loss + wavelet_loss_PPI*alpha

        loss = loss_x4
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        running_results['im_loss'] = loss_x4.item()
        running_results['all_loss'] = loss.item()

        train_bar.set_description(desc='[%d/%d] Loss_im: %.4f Loss_re: %.1f Loss_all: %.1f' % (
            epoch, num_epochs, running_results['im_loss'] / running_results['batch_sizes'],
            running_results['re_loss'] / running_results['batch_sizes'],
            running_results['all_loss'] / running_results['batch_sizes']))
    return running_results


def test(testing_data_loader, optimizer, model, epoch, num_epochs):
    test_bar = tqdm(testing_data_loader)
    test_results = {'batch_sizes': 0, 'psnr': 0, 'mse': 0}
    model.eval()

    with torch.no_grad():
        for batch in test_bar:

            input_raw, target_PPI = Variable(batch[0]), Variable(batch[1], requires_grad=False)
            N, C, H, W = batch[0].size()
            test_results['batch_sizes'] = N

            if opt.cuda:
                input_raw = input_raw.cuda()
                target_PPI = target_PPI.cuda()

            estimated_PPI = model(input_raw)

            batch_mse = ((estimated_PPI - target_PPI) ** 2).data.mean()
            test_results['mse'] = batch_mse * N
            test_results['psnr'] = 10 * log10(1 / (test_results['mse'] / test_results['batch_sizes']))
            test_bar.set_description(desc='[%d/%d] psnr: %.4f ' % (
                epoch, num_epochs, test_results['psnr']))
    return test_results


def save_checkpoint(model, epoch):
    model_folder = "checkpoint/pre-training/"
    model_out_path = model_folder + "PPI_model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def save_opt(opt):
    statistics_folder = "checkpoint/pre-training/"
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    data_frame = pd.DataFrame(
        data=vars(opt), index=range(1, 2))
    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_opt.csv', index_label='Epoch')
    print("save--opt")

def save_statistics(opt, results, epoch):
    statistics_folder = "checkpoint/pre-training/"
    if not os.path.exists(statistics_folder):
        os.makedirs(statistics_folder)
    #
    data_frame = pd.DataFrame(
        data=results,index=range(opt.start_epoch, epoch + 1))
    data_frame.to_csv(statistics_folder + str(opt.start_epoch) + '_train_results.csv', index_label='Epoch')

if __name__ == "__main__":
    main()
