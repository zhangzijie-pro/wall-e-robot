class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0
        self.val = 0.0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += int(n)
        self.val = float(val)
        self.avg = self.sum / max(1, self.cnt)

    @property
    def avg(self):
        return self.sum / max(1, self.cnt)


def top1_accuracy(logits, targets):
    # logits: [B,C], targets: [B]
    pred = logits.argmax(dim=1)
    correct = (pred == targets).sum().item()
    return correct / max(1, targets.size(0))
