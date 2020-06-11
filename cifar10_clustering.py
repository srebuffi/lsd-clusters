import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import utils_datasets as datasets
from utils_misc import seed_torch, prepare_save_dir, create_logger, save_logger, write_txt
from utils_algo import PairEnum, BCE_softlabels, sigmoid_rampup, cluster_acc
import utils_net
import numpy as np
import os


def train(args, model, device, train_loader, optimizer, epoch):
    """ Train for 1 epoch."""
    model.train()
    w_cons = args.rampup_coefficient * sigmoid_rampup(epoch, args.rampup_length)
    bce = BCE_softlabels()
    sum_loss = 0
    for batch_idx, ((x, x_bar, x_bar2), target, idx_batch) in enumerate(train_loader):
        # RICAP
        I_x, I_y = x_bar.size()[2:]
        w = int(np.round(I_x * np.random.beta(0.3, 0.3)))
        h = int(np.round(I_y * np.random.beta(0.3, 0.3)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]
        cropped_images = {}
        cropped_images_bar = {}
        idx_ = {}
        W_ = {}
        for k in range(4):
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            x_k_bar = np.random.randint(0, I_x - w_[k] + 1)
            y_k_bar = np.random.randint(0, I_y - h_[k] + 1)
            if (w_[k] * h_[k]) > (I_x / 2 * I_y / 2):
                idx = torch.arange(x_bar.size(0))
                idx_[k] = 'identity'
            else:
                idx = torch.randperm(x_bar.size(0))
                idx_[k] = idx
            cropped_images[k] = x_bar[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            cropped_images_bar[k] = x_bar2[idx][:, :, x_k_bar:x_k_bar + w_[k], y_k_bar:y_k_bar + h_[k]]
            W_[k] = w_[k] * h_[k] / (I_x * I_y)

        x_bar = torch.cat((
            torch.cat((cropped_images[0], cropped_images[1]), 2),
            torch.cat((cropped_images[2], cropped_images[3]), 2)), 3
        )

        x_bar2 = torch.cat((
            torch.cat((cropped_images_bar[0], cropped_images_bar[1]), 2),
            torch.cat((cropped_images_bar[2], cropped_images_bar[3]), 2)), 3
        )

        x, x_bar, x_bar2, target = x.to(device), x_bar.to(device), x_bar2.to(device), target.to(device)
        optimizer.zero_grad()
        output, feat = model(x)
        output_bar, _ = model(x_bar)
        output_bar2, _ = model(x_bar2)
        prob, prob_bar, prob_bar2 = F.softmax(output, dim=1), F.softmax(output_bar, dim=1), F.softmax(output_bar2, dim=1)

        # Similarity labels
        feat_detach = feat.detach()
        if args.similarity_type == 'cosine':
            feat_norm = feat_detach / torch.norm(feat_detach, 2, 1, keepdim=True)
            cosine_feat = torch.mm(feat_norm, feat_norm.t()).view(-1)
            target_ulb = torch.ones_like(cosine_feat).float()
            target_ulb[cosine_feat < args.hyperparam] = 0
        elif args.similarity_type == 'SNE':
            feat_row, feat_col = PairEnum(feat_detach)
            tmp_distance_ori = -(((feat_row - feat_col) / args.temperature) ** 2.).sum(1)
            tmp_distance = tmp_distance_ori.view(x.size(0), x.size(0)) - 1000 * torch.eye(x.size(0)).cuda()
            prob_simil = (0.5 * torch.softmax(tmp_distance, 1) + 0.5 * torch.softmax(tmp_distance, 0) + torch.eye(x.size(0)).cuda()).view(-1)
            target_ulb = torch.zeros_like(prob_simil).float()
            target_ulb[prob_simil > args.hyperparam] = 1
        elif args.similarity_type == 'kNN':
            feat_row, feat_col = PairEnum(feat_detach)
            tmp_distance_ori = ((feat_row - feat_col) ** 2.).sum(1).view(x.size(0), x.size(0))
            target_ulb = torch.zeros_like(tmp_distance_ori).float()
            target_ulb[tmp_distance_ori < torch.kthvalue(tmp_distance_ori, int(args.hyperparam), 1, True)[0]] = 1
            target_ulb[tmp_distance_ori < torch.kthvalue(tmp_distance_ori, int(args.hyperparam), 0, True)[0]] = 1
            target_ulb = target_ulb.view(-1)

        # Adapting the target to RICAP
        target_ulb_tmp = target_ulb.clone()
        target_ulb = torch.zeros_like(target_ulb).float().to(device)
        for k in range(4):
            if type(idx_[k]) == str:
                target_ulb += W_[k] * target_ulb_tmp
            else:
                target_ulb += W_[k] * (target_ulb_tmp.view(feat.size(0), -1)[:, idx_[k]]).view(-1)

        # Clustering and consistency losses
        prob_bottleneck_row, _ = PairEnum(prob)
        _, prob_bottleneck_col = PairEnum(prob_bar)
        loss = bce(prob_bottleneck_row, prob_bottleneck_col, target_ulb)
        loss += w_cons * F.mse_loss(prob_bar, prob_bar2)

        optimizer.zero_grad()
        sum_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % (len(train_loader) // 4) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), sum_loss / (batch_idx + 1)
            ))

    args.logger['train_loss'].append(sum_loss / (batch_idx + 1))


def test(args, model, device, test_loader):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    with torch.no_grad():
        for batch_idx, (x, label, _) in enumerate(test_loader):
            x, label = x.to(device), label.to(device)
            output, _ = model(x)
            _, pred = output.max(1)
            targets = np.append(targets, label.cpu().numpy())
            preds = np.append(preds, pred.cpu().numpy())

    acc = cluster_acc(targets.astype(int), preds.astype(int))
    print('Test acc {:.4f}'.format(acc))
    args.logger['test_acc'].append(acc)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='LSD-C: Linearly Separable Deep Clusters')
    parser.add_argument('--milestones', nargs='+', type=int, default=[140, 180])
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--epochs', type=int, default=220)
    parser.add_argument('--rampup_length', type=int, default=100)
    parser.add_argument('--rampup_coefficient', type=float, default=5)
    parser.add_argument('--similarity_type', type=str, default='kNN')
    parser.add_argument('--hyperparam', type=float, default=20)
    parser.add_argument('--temperature', type=float, default=1.)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # Seed the run and create saving directory
    seed_torch(args.seed)
    args = prepare_save_dir(args, __file__)

    # Logger
    args = create_logger(args, metrics=['train_loss', 'test_acc'])
    save_logger(args)

    # Initialize the splits
    trainset = datasets.CIFAR10_ALL(root=os.getcwd(), train=True, download=True, transform=datasets.TransformThrice(datasets.dict_transform['cifar_train']))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, drop_last=True)

    testset = datasets.CIFAR10_ALL(root=os.getcwd(), train=True, download=True, transform=datasets.dict_transform['cifar_test'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=1)

    # First network intialization: pretrain the RotNet network
    model = utils_net.ResNet(utils_net.BasicBlock, [2, 2, 2, 2], 10)
    model = model.to(device)
    state_dict_rotnet = torch.load('RotNet_cifar10.pt')
    del state_dict_rotnet['linear.weight']
    del state_dict_rotnet['linear.bias']
    model.load_state_dict(state_dict_rotnet, strict=False)
    model = model.to(device)

    # Freeze the earlier filters
    for name, param in model.named_parameters():
        if 'linear' not in name and 'layer4' not in name:
            param.requires_grad = False

    # Set the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    for epoch in range(args.epochs):
        args.logger['epoch'] = epoch
        train(args, model, device, trainloader, optimizer, epoch)
        test(args, model, device, testloader)
        scheduler.step()
        write_txt(args, f"Test acc: {args.logger['test_acc'][-1]}")
        save_logger(args)


if __name__ == '__main__':
    main()
