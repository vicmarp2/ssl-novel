Instructions to reproduce reported results.

# CIFAR 10
## 4000 labels
best_model_cifar10_4000.pt - error rate 15.96%
```
& python "./main.py" --dataset cifar10 --num-labeled 4000 --num-validation 1000 --lr 0.03 --momentum 0.9 --wd 0.001 --train-batch 64 --total-iter 38400 --iter-per-epoch 128 --confidence-threshold 0.95 --similarity-threshold 0.9 --mu 7 --lambda-u 1 --lambda-pair-s 75 --lambda-pair-u 75 --max-grad-norm 2 --modelpath "./models/obs/"
```
## 250 labels
best_model_cifar10_250.pt - error rate 46.96%
```
& python "./main.py" --dataset cifar10 --num-labeled 250 --num-validation 100 --lr 0.03 --momentum 0.9 --wd 0.001 --train-batch 64 --total-iter 38400 --iter-per-epoch 128 --confidence-threshold 0.95 --similarity-threshold 0.9 --mu 7 --lambda-u 1 --lambda-pair-s 75 --lambda-pair-u 75 --max-grad-norm 2 --modelpath "./models/obs/"
```
# CIFAR 100
## 10000 labels
best_model_cifar100_10000.pt - error rate 43.54%
```
& python "./main.py" --dataset cifar100 --num-labeled 10000 --num-validation 2000 --lr 0.03 --momentum 0.9 --wd 0.001 --train-batch 64 --total-iter 38400 --iter-per-epoch 128 --confidence-threshold 0.95 --similarity-threshold 0.9 --mu 7 --lambda-u 1 --lambda-pair-s 75 --lambda-pair-u 75 --max-grad-norm 2 --modelpath "./models/obs/"
```
## 4000 labels
best_model_cifar100_4000.pt - error rate 61.59%
```
& python "./main.py" --dataset cifar100 --num-labeled 4000 --num-validation 1000 --lr 0.03 --momentum 0.9 --wd 0.001 --train-batch 64 --total-iter 38400 --iter-per-epoch 128 --confidence-threshold 0.95 --similarity-threshold 0.9 --mu 7 --lambda-u 1 --lambda-pair-s 75 --lambda-pair-u 75 --max-grad-norm 2 --modelpath "./models/obs/"
```