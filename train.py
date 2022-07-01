import model
import dataloader
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import time

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    
    for batch, batch_data in enumerate(dataloader):
        start_time = time.time()
        for data in batch_data:
            # prediction
            prediction, target = model(data)

            loss = loss_fn(prediction, target)

            # backpropagation
            writer.add_scalar("Loss/train", loss, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        used_time = time.time()-start_time
        if batch % 1 == 0:
            loss, current = loss, batch * len(batch_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], used time: {used_time:>7f} sec")

device="cuda" if torch.cuda.is_available() else "cpu"

first = 0
last = 33000
train_data = dataloader.PersonaChat_with_Images(split='train', image_path='./image_dataset', init=first, end=last)
train_data = Subset(train_data, torch.arange(first,last).tolist())
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=dataloader.collate_fn)

model = model.GenDial(device)
model = model.to(device)

num_epoch = 4
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()
writer = SummaryWriter()

for epoch in range(num_epoch):
    path = './model_checkpoint/model_'+str(epoch)+'.pt'
    print('epoch: {}'.format(epoch))
    train(train_dataloader, model, loss_fn, optimizer)
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
writer.flush()