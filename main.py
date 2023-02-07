import argparse
import os
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tqdm.auto import tqdm

import utils
from datasets import get_dataset
from loss import get_loss
from model import Model


def train(net, data_loader, loss_criterion, train_optimizer, batch_size, *, cuda=True, writer, m_agg_mode, step=0):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, data_loader
    for pos_1, pos_2, pos_m, target in train_bar:
        if cuda:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            pos_m = [pos.cuda(non_blocking=True) for pos in pos_m]
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        if m_agg_mode == utils.MAggMode.pos_grouping:
            out_m = []
            for pos_i in pos_m:
                feature_i, out_i = net(pos_i)
                out_m.append(out_i)
            loss = loss_criterion(out_1, out_2, out_m, target)
        elif m_agg_mode == utils.MAggMode.loss_combination:
            out_m = [out_1, out_2]
            for pos_i in pos_m:
                feature_i, out_i = net(pos_i)
                out_m.append(out_i)
            losses = []
            for i in range(len(out_m)):
                for j in range(i + 1, len(out_m)):
                    losses.append(loss_criterion(out_m[i], out_m[j], [], target))
            loss = torch.stack(losses).mean()
        else:
            raise Exception(f"Unknown m_agg_mode '{m_agg_mode}'")

        writer.add_scalar("loss/train", loss, step)
        step += 1

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        # train_bar.set_description("Train Epoch: [{}/{}] Loss: {:.4f}".format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num, step


# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, *, top_k, class_cnt, cuda=True, temperature):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, _, target in memory_data_loader:
            if cuda:
                data = data.cuda(non_blocking=True)
            feature, out = net(data)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        # Probably AttributeError on Cifar10
        feature_labels = torch.tensor(
            memory_data_loader.dataset.labels, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = test_data_loader
        for data, _, _, target in test_bar:
            if cuda:
                data, target = data.cuda(
                    non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=top_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(
                data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(
                data.size(0) * top_k, class_cnt, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(
                dim=-1, index=sim_labels.view(-1, 1).long(), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, class_cnt) * sim_weight.unsqueeze(dim=-1),
                                    dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            # test_bar.set_description(
            #     "KNN Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%".format(
            #         epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100
            #     )
            # )

    return total_top1 / total_num * 100, total_top5 / total_num * 100


def main(dataset: str, loss: str, root: str, batch_size: int, model_arch, *, cuda=True, writer, feature_dim=128,
         temperature=0.5, tau_plus=0.1, top_k=200, epochs=200, num_pos=1, drop_fn=False, noise_frac=0.0,
         m_agg_mode="m_agg_mode", run_uuid=None):
    wandb.config.update({
        "dataset": dataset,
        "loss": loss,
        "model": model_arch,
        "feature_dim": feature_dim,
        "temperature": temperature,
        "tau_plus": tau_plus,
        "k": top_k,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_pos": num_pos,
        "drop_fn": drop_fn,
        "noise_frac": noise_frac or 0.0,
        "m_agg_mode": m_agg_mode,
        "uuid": run_uuid,
    })

    m_agg_mode = utils.MAggMode[m_agg_mode]
    if m_agg_mode == utils.MAggMode.pos_grouping and loss != "DebiasedNeg" and num_pos > 1:
        raise Exception("Mean aggregation only available in DebiasedNeg loss")

    train_loader = DataLoader(
        get_dataset(dataset, root=root, split="train+unlabeled", num_pos=num_pos,
                    transform=utils.train_transform, noise_frac=noise_frac),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    memory_loader = DataLoader(
        get_dataset(
            dataset, root=root, split="train", num_pos=1, transform=utils.test_transform,
            noise_frac=noise_frac),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(
        get_dataset(dataset, root=root, split="test", num_pos=1, transform=utils.test_transform, noise_frac=noise_frac),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    loss_criterion = get_loss(loss, temperature, cuda, tau_plus, drop_fn)

    # model setup and optimizer config
    model = Model(feature_dim, model_arch)
    if cuda:
        model = model.cuda()
    model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_loader.dataset.classes)
    print("# Classes: {}".format(c))

    step = 0
    for epoch in range(1, epochs + 1):
        train_loss, step = train(model, train_loader, loss_criterion, optimizer, batch_size,
                                 cuda=cuda, writer=writer, m_agg_mode=m_agg_mode, step=step)
        if epoch % 5 == 0:
            test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, cuda=cuda, class_cnt=c, top_k=top_k,
                                          temperature=temperature)
            writer.add_scalar("loss/acc1", test_acc_1, epoch)
            writer.add_scalar("loss/acc5", test_acc_5, epoch)

            model_path = "{}/model_{}.pth".format(writer.log_dir, epoch)
            torch.save(model.state_dict(), model_path)
            wandb.log({
                "acc1": test_acc_1,
                "acc5": test_acc_5,
            })

            art = wandb.Artifact(f"model_{epoch}", type="model")
            art.add_file(model_path)
            wandb.log_artifact(art)
        writer.flush()
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--dataset', type=str,
                        help='Dataset name (STL10 or CIFAR10 or STL10Noise or CIFAR10Noise)')
    parser.add_argument(
        '--loss', type=str, help='Loss name (Contrastive or DebiasedNeg or DebiasedPos)')
    parser.add_argument('--root', type=str, help='Dataset source root')
    parser.add_argument('--root_out', type=str, help='Path to store logs')
    parser.add_argument('--wandb_project', type=str, help='Project name')
    parser.add_argument('--model_arch', type=str,
                        help='Model architecture (resnet18/34/50')
    parser.add_argument('--cuda', help='Use cuda', action=argparse.BooleanOptionalAction)

    parser.add_argument('--feature_dim', default=128,
                        type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5,
                        type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1,
                        type=float, help='Positive class priorx')
    parser.add_argument('--top_k', default=200, type=int,
                        help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--num_pos', default=1, type=int, help='Number of positive samples in loss while train')
    parser.add_argument('--drop_fn', help='Drop false negative samples', action=argparse.BooleanOptionalAction)
    parser.add_argument('--noise_frac', default=0.0, type=float, help='Fraction of noise augmentations')
    parser.add_argument('--m_agg_mode', default="mean", type=str, help='loss aggregation mode')

    args = parser.parse_args()

    if not os.path.exists(args.root_out):
        os.mkdir(args.root_out)
    wandb.init(project=args.wandb_project, dir=args.root_out)
    wandb.tensorboard.patch(root_logdir=args.root_out)
    writer = SummaryWriter(args.root_out)

    main(
        args.dataset, args.loss, args.root, args.batch_size, args.model_arch,
        cuda=args.cuda,
        writer=writer,
        feature_dim=args.feature_dim,
        temperature=args.temperature,
        tau_plus=args.tau_plus,
        top_k=args.top_k,
        epochs=args.epochs,
        num_pos=args.num_pos,
        noise_frac=args.noise_frac,
        m_agg_mode=args.m_agg_mode,
    )
