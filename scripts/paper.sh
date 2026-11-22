#!/usr/bin/env bash
# Commands for the experiments of the paper (Sec. III). Run from the repo root.
# Each run trains, then evaluates best.pth on the test set before and after
# temperature scaling (results.json, reliability diagrams, confusion matrices).
set -e

effnet() {  # EfficientNet-B0 from scratch, 60 epochs, OneCycle max LR 0.1, dropout 0.2
  for loss in ce logit_margin; do
    python tools/train_net.py data=road data.batch_size=256 \
      model=efficientnet_b0 model.dropout_rate=0.2 loss=$loss \
      optim=sgd optim.lr=0.1 optim.momentum=0.9 optim.weight_decay=5e-4 \
      scheduler=one_cycle train.max_epoch=60
  done
}

resnet50() {  # ResNet-50 ImageNet-pretrained, 100 epochs, OneCycle max LR 0.1
  for loss in ce logit_margin; do
    python tools/train_net.py data=road data.batch_size=256 \
      model=resnet50 model.num_classes=6 model.pretrained=True loss=$loss \
      optim=sgd optim.lr=0.1 optim.momentum=0.9 optim.weight_decay=5e-4 \
      scheduler=one_cycle train.max_epoch=100
  done
}

# Evaluate an existing checkpoint:
#   python tools/test_net.py model=efficientnet_b0 loss=ce test.checkpoint=/abs/path/best.pth

"${@:-effnet}"
