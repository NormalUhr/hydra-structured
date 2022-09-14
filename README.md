# Pretrain (Big Dense)

Adversarial pretrain: ResNet20s
```
python3 train.py --arch resnet20s --k 1.0 --exp-mode pretrain --exp-name resnet20s_adv_pretrain --dataset CIFAR10 --trainer adv --val-method adv --layer-type subnet | tee log/resnet20s_adv_pretrain.log
```

Adversarial Pretrain: ResNet18:
```
python3 train.py --arch resnet18 --k 1.0 --exp-mode pretrain --exp-name resnet18_adv_pretrain --dataset CIFAR10 --trainer adv --val-method adv --layer-type subnet | tee log/resnet18_adv_pretrain.log
```

# Prune (Small Dense)

`--k`: Remaining Ratio

`--source-net` Pretrained Model paths

```
python3 train.py --arch resnet18 --k ${k} --exp-mode prune --exp-name resnet18_ratio${k}_adv_pruning --source-net results/resnet18/resnet18_adv_pretrain/pretrain/latest_exp/checkpoint/model_best.pth.tar --dataset CIFAR10 --trainer adv --val-method adv --layer-type subnet --scaled-score-init | tee log/resnet18_ratio${k}_adv_pruning.log
```

# Prune with CIFAR10 Cluster
`--k`: Remaining Ratio

`--source-net` Pretrained Model paths

`--dataset-idx` 0~5

`--dataset-idx-method` dino or wrn

`--dataset` CIFAR10Idx

```
python3 train.py --arch resnet18 --k ${k} --exp-mode prune --exp-name resnet18_ratio${k}_idx${idx}_adv_pruning --dataset-idx-method wrn --source-net results/resnet18/resnet18_adv_pretrain/pretrain/latest_exp/checkpoint/model_best.pth.tar --dataset-idx ${idx} --dataset CIFAR10Idx --trainer adv --val-method adv --layer-type subnet --scaled-score-init | tee log/resnet18_ratio${k}_idx${idx}_adv_pruning.log
```

# Retrain (Small Dense)

```
python3 train.py --arch resnet18 --k ${k} --exp-mode finetune --exp-name resnet18_ratio${k}_adv_pruning --source-net results/resnet18/resnet18_ratio${k}_idx${idx}_adv_pruning/prune/latest_exp/checkpoint/model_best.pth.tar --dataset CIFAR10 --trainer adv --val-method adv --layer-type subnet --scaled-score-init | tee log/resnet18_ratio${k}_adv_pruning.log
```

# Retrain with CIFARIdx Cluster

`--k`: Remaining Ratio

`--source-net` Pretrained Model paths

`--dataset-idx` 0~5

`--dataset-idx-method` dino or wrn

`--dataset` CIFAR10Idx

```
python3 train.py --arch resnet18 --k ${k} --dataset-idx-method wrn --exp-mode finetune --exp-name resnet18_ratio${k}_idx${idx}_adv_pruning --source-net results/resnet18/resnet18_ratio${k}_idx${idx}/prune/latest_exp/checkpoint/model_best.pth.tar --dataset-idx ${idx} --dataset CIFAR10Idx --trainer adv --val-method adv --layer-type subnet --scaled-score-init | tee log/resnet18_ratio${k}_idx${idx}_adv_pruning_finetune.log
```

## Prune with Auto Router
```
python3 train.py --arch resnet18 --k ${k} --exp-mode prune --exp-name resnet18_ratio${k}_adv_auto_pruning --source-net results/resnet18/resnet18_adv_pretrain/pretrain/latest_exp/checkpoint/model_best.pth.tar --dataset CIFAR10 --trainer adv --val-method adv --layer-type subnet --scaled-score-init --use_trainable_router --router_checkpoint_path results/resnet18/resnet18_adv_pretrain/pretrain/latest_exp/checkpoint/model_best.pth.tar --router_arch resnet18
```
## Retrain with Auto Router
```
python3 train.py --arch resnet18 --k ${k} --exp-mode finetune --exp-name resnet18_ratio${k}_adv_auto_pruning --source-net results/resnet18/resnet18_ratio${k}_idx${idx}_adv_pruning/prune/latest_exp/checkpoint/model_best.pth.tar --dataset CIFAR10 --trainer adv --val-method adv --layer-type subnet --scaled-score-init  --use_trainable_router --router_arch resnet18
```
