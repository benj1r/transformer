import torch.nn.functional as F

class Tester:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def test(self):
        losses = []
        
        for batch_idx, (data, targets) in self.dataset:
            scores = self.model(data)

            loss = F.nll_loss(scores, targets)
            losses.append(loss.item())
            loss.backward()

            self.dataset.set_description("test | loss: {round(loss.item(), 2)}")
        
        return losses
