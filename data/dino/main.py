import os
import sys
# sys.path.append("../..")
# from hash_utils import tsne, test_kmeans

import torch
import argparse

import torchvision
from torchvision import models as torchvision_models
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import vision_transformer as vits
import utils
from sklearn.cluster import KMeans
import torch.nn as nn

sys.path.append("../..")
from models import resnet18
from models.layers import SubnetConv, SubnetLinear


def set_prune_rate_model(model, prune_rate):
    for _, v in model.named_modules():
        if hasattr(v, "set_prune_rate"):
            v.set_prune_rate(prune_rate)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])

parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
                    help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
    We typically set this to False for ViT-Small and to True with ViT-Base.""")
parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
parser.add_argument("--checkpoint_key", default="teacher", type=str,
                    help='Key to use in the checkpoint (example: "teacher")')
parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
    training (highest LR used during training). The learning rate is linearly scaled
    with the batch size, and specified here for a reference batch size of 256.
    We recommend tweaking the LR depending on the checkpoint evaluated.""")
parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size')
parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
    distributed training; see https://pytorch.org/docs/stable/distributed.html""")
parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
parser.add_argument('--data', default='./data/', type=str)
parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')

parser.add_argument('--max_iter', type=int, default=300)
parser.add_argument('--n_init', type=int, default=10)

args = parser.parse_args()

save_path = '/gdata2/cairs/temp/dino'

if args.arch == 'resnet50':
    path = '/gdata2/cairs/temp/dino_resnet50_pretrain.pth'
    model = torchvision_models.__dict__['resnet50']()
    utils.load_pretrained_weights(model, path, args.checkpoint_key, args.arch, args.patch_size)
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

elif args.arch == 'vit_small':
    path = '/gdata2/cairs/temp/dino/dino_deitsmall8_pretrain.pth'
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    N_dim = 384
    utils.load_pretrained_weights(model, path, args.checkpoint_key, args.arch, args.patch_size)
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

elif args.arch == "resnet18":
    path = "../../results/resnet18/resnet18_adv_pretrain/pretrain/latest_exp/checkpoint/model_best.pth.tar"
    model = resnet18(SubnetConv, nn.Linear, init_type="kaiming_normal", num_classes=0)
    set_prune_rate_model(model, 1.0)
    N_dim = 512
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])
    checkpoint = torch.load(path)["state_dict"]
    for name, param in model.state_dict().items():
        if checkpoint[name].shape != param.shape:
            checkpoint.pop(name)
    model.load_state_dict(checkpoint, strict=False)

else:
    raise NotImplementedError

model.cuda()
model.eval()

trainset = torchvision.datasets.CIFAR10(root=args.data + args.dataset, train=True, download=True,
                                        transform=val_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=5,
                                           pin_memory=True)
testset = torchvision.datasets.CIFAR10(root=args.data + args.dataset, train=False, download=True,
                                       transform=val_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=5,
                                          pin_memory=True)


def get_features(model, data_loader, mode):
    N = len(data_loader.dataset)
    features_all = torch.empty([N, N_dim])
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            features = model(images)
            print(i)

            start = i * args.batch_size
            end = min((i + 1) * args.batch_size, N)
            features_all[start:end, ...] = features
    torch.save(features_all, os.path.join(save_path, '%s_features_%s' % (mode, args.arch)))
    return features_all


# clean_train = get_features(model, train_loader, 'clean_train')
# print('finish saving')
clean_test = get_features(model, test_loader, 'clean_test')
# print('finish saving')

clean_train = torch.load(os.path.join(save_path, '%s_features_%s' % ('clean_train', args.arch)))
# clean_test = torch.load(os.path.join(save_path, '%s_features_%s'%('clean_test', args.arch)))

# args.adv_folder = '/gdata2/cairs/DeepMoE_robustness/hash_moe/base_model_step1/cifar10_std/adv_images'
# adv_images = torch.load(os.path.join(args.adv_folder, 'test_adv_images'), map_location='cpu')
# adv_targets = torch.load(os.path.join(args.adv_folder, 'test_adv_targets')).long()
# labels = torch.load(os.path.join(args.adv_folder, 'test_adv_labels')).long()
# y = torch.stack([adv_targets, labels], dim=1)
# adv_transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize(256, interpolation=3),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
# adv_images = torch.stack([adv_transform(adv_images[i]) for i in range(adv_images.shape[0])], dim=0)
# torch.save(adv_images, os.path.join(save_path, 'adv_images'))
# adv_dataset = TensorDataset(adv_images, y)
# adv_loader = torch.utils.data.DataLoader(dataset=adv_dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)
# adv_test = get_features(model, adv_loader, 'adv_test')
# print('finish saving')

# adv_test = torch.load(os.path.join(save_path, '%s_features_%s'%('adv_test', args.arch)))


N_use = 500
for cluster_num in [5]:
    embedding = clean_train
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, max_iter=args.max_iter, n_init=args.n_init)
    kmeans.fit(embedding)
    centroids = torch.tensor(kmeans.cluster_centers_)
    torch.save(centroids, os.path.join(save_path, '%s_centroids_num_%d' % (args.arch, cluster_num)))

    # tsne(torch.cat([clean_test[0:N_use], adv_test[0:N_use], centroids.float()], dim=0), labels[0:N_use], 
    #     save_path, N_use, name='test_%d_%s'%(cluster_num, args.arch))

    # mix_test_embedding = torch.cat([clean_test, adv_test], dim=0)
    # preds_test = torch.tensor(kmeans.predict(mix_test_embedding)).long().cuda()

    preds_train = torch.tensor(kmeans.predict(clean_train)).long().cuda()
    torch.save(preds_train, os.path.join(save_path, 'pred_train'))

    preds_test = torch.tensor(kmeans.predict(clean_test)).long().cuda()
    torch.save(preds_test, os.path.join(save_path, 'pred_test'))
