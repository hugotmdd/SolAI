import torch

def f1score(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_pred = torch.argmax(pred, dim=1)
    tp = torch.sum((y_pred == 1) * (y == 1))
    fn = torch.sum((y_pred == 0) * (y == 1))
    fp = torch.sum((y_pred == 1) * (y == 0))

    precision = tp / (tp + fp + 0.00001)
    recall = tp / (tp + fn + 0.00001)

    score = 2 * (precision * recall) / (precision + recall + 0.00001)

    return score

class Dice(torch.nn.Module):
    def __init__(self, sigmoid: bool = True):
        super(Dice, self).__init__()
        self.name = "Dice"
        self.sigmoid = sigmoid

    def forward(self, y_pred, y_true):
        if self.sigmoid:
            y_pred = (torch.sigmoid(y_pred) > 0.5).float()
        intersection = y_pred * y_true
        pred_gt_sum = y_pred + y_true

        numerator = 2 * torch.sum(intersection, dim=(2, 3))
        denominator = torch.sum(pred_gt_sum, dim=(2, 3)) + 0.00001

        return torch.mean(numerator / denominator)