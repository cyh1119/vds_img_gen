import torch
import model
import dataloader
from torch.utils.data import DataLoader, Subset
import time
from torch.utils.tensorboard import SummaryWriter

def NLLLoss(prediction, target):
    device="cuda" if torch.cuda.is_available() else "cpu"
    target_length = target.shape[0]
    vocab_length = int(prediction.shape[0] / target_length)

    # create binary mask
    target_binary_mat = torch.zeros([target_length, vocab_length], device = device)
    for target_idx, target_word in zip(target, target_binary_mat):
        target_word[target_idx] = 1
    target_binary = torch.flatten(target_binary_mat)

    log_prob = torch.log(prediction)
    sum_log_likelihood = torch.matmul(log_prob,target_binary.T)
    loss = -1 * sum_log_likelihood

    return loss

def train(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    #writer.add_graph(model)
    for batch, batch_data in enumerate(dataloader):
        start_time = time.time()
        for data in batch_data:
            # prediction
            prediction, target = model(data)

            loss = NLLLoss(prediction, target)
            # backpropagation
            writer.add_scalar("Loss/train", loss, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        used_time = time.time()-start_time
        if batch % 1 == 0:
            loss, current = loss, batch * len(batch_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], used time: {used_time:>7f} sec")
    return loss

device="cuda" if torch.cuda.is_available() else "cpu"

first = 0
last = 33000
train_data = dataloader.PersonaChat_with_Images(split='train', image_path='./image_dataset', init=first, end=last)
train_data = Subset(train_data, torch.arange(first,last).tolist())
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=dataloader.collate_fn)

model = model.GenDial(device)
model.load_state_dict(torch.load("./model_checkpoint/model_0.pt")['model_state_dict'])
model = model.to(device)

num_epoch = 4
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
writer = SummaryWriter()

for epoch in range(num_epoch):
    path = './model_checkpoint/model_'+str(epoch)+'.pt'
    print('epoch: {}'.format(epoch))
    train(train_dataloader, model, optimizer)
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
writer.flush()