import os
import random

import numpy as np
import torch
from torch import nn
from transformers import AdamW

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 3)
        self.b1 = nn.Linear(3, 2)
        self.b2 = nn.Linear(3, 2)

    def forward(self, x, y):
        return self.a(x)

    def f1(self, x, y):
        feat = self.b1(x)
        return feat
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(feat, y)

    def f2(self, x, y):
        feat = self.b2(x)
        return feat
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(feat, y)


model = Model()
x = torch.randn([3, 2])
y = torch.randint(2, (3,))

a_optimizer = AdamW(model.a.parameters(), lr=1, eps=0.1)
b1_optimizer = AdamW(model.b1.parameters(), lr=1, eps=0.1)
b2_optimizer = AdamW(model.b2.parameters(), lr=1, eps=0.1)
all_optimizer = AdamW(model.parameters(), lr=1, eps=0.1)

model.train()
feat = model(x, y)
f1 = model.f1(feat, y)
# loss = model.f1(feat, y)
# loss.backward()
loss_fn = nn.CrossEntropyLoss()
f2 = model.f2(feat.detach(), y)
l1 = loss_fn(f1, y)
l2 = loss_fn(f2, y)
l1.backward()
l2.backward()
all_optimizer.step()
# b2_optimizer.step()

print(1)
