# SeesawLoss_pytorch

This implementation is based on [bamps53](https://github.com/bamps53)/**[SeesawLoss](https://github.com/bamps53/SeesawLoss)**. His implementation only involves mitigation factors, no compensation factors.I implemented this part.

## useage

```python
from loss_utils import DistibutionAgnosticSeesawLossWithLogits
num_labels = 10
loss_fn = DistibutionAgnosticSeesawLossWithLogits(num_labels=num_labels)
loss = loss_fn(logits, label)
```

`preds`: logits

`label`: not one-hot label

## Citation

1. This is unofficial pytorch implementation for SeesawLoss, which was proposed by Jiaqi Wang et. al. in their technical report for LVIS workshop at ECCV 2020.
