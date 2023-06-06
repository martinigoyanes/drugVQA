import torch
import torch.nn.functional as F

# Focal Loss with alpha=0.25 and gamma=2 (standard)
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(pred, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

# Label Smoothing with smoothing=0.1
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes=2, smoothing=0.1, dim=-1, weight = None):
        """if smoothing == 0, it's one-hot method
           if 0 < smoothing < 1, it's smooth method
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.weight = weight
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=self.dim)

        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            print(target.data.shape)
            print(target.data.unsqueeze(1).shape)
            print(pred.data.shape)
            print(true_dist.data.shape)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

'''
torch.Size([1])
torch.Size([1, 1])
torch.Size([1])
torch.Size([1])
Traceback (most recent call last):
  File "main.py", line 15, in <module>
    main()
  File "main.py", line 12, in main
    losses,accs,testResults = train(trainArgs)
  File "/Midgard/home/martinig/adv-comp-bio/trainAndTest.py", line 54, in train
    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
  File "/Midgard/home/martinig/miniconda3/envs/drugVQA/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/Midgard/home/martinig/adv-comp-bio/loss.py", line 44, in forward
    true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
'''