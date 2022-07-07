import model
import dataset
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import time

def CrossEntropyLoss(prediction, target, text_length, response_length, device):
    '''
    input: list of prediction per text, list of 
    output: loss -> torch tensor
    '''
    batch, max_len, vocab_len = prediction.shape
    
    target_extended = torch.ones([batch, max_len], device=device)*-1e2
    for idx, (len_text, len_response) in enumerate(zip(text_length, response_length)):
        target_extended[idx][len_text-len_response-1:len_text-1] = target[idx][0]

    prediction = prediction.view(batch*max_len, vocab_len)
    target_extended = target_extended.view(batch*max_len)
    target_extended = target_extended.type(torch.long)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(prediction, target_extended)
    return loss

def train(dataloader, model, optimizer, device):
    size = len(dataloader.dataset)
    loss_sum = 0
    for batch, data in enumerate(dataloader):
        print(batch)
        start_time = time.time()
        # prediction
        prediction = model(data)
        target = data['response']
        text_length = [sum([len(text) for text in texts]) for texts in data['text']]
        response_length = [sum([len(text) for text in texts]) for texts in data['response']]
        loss = CrossEntropyLoss(prediction, target, text_length, response_length, device)
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        used_time = time.time()-start_time
        if batch % 1 == 0:
            loss, current = loss, (batch+1) * len(data['image'])
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}], used time: {used_time:>7f} sec")
        loss_sum += loss

    return loss_sum / len(dataloader)

def test(dataloader, model, device):
    #size = len(dataloader.dataset)
    model.eval()

    loss_sum = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # prediction
            prediction = model(data)
            
            # loss
            target = data['response']
            text_length = [sum([len(text) for text in texts]) for texts in data['text']]
            response_length = [sum([len(text) for text in texts]) for texts in data['response']]
            loss = CrossEntropyLoss(prediction, target, text_length, response_length, device)
            loss_sum += loss

    return loss_sum / len(dataloader)

device="cuda" if torch.cuda.is_available() else "cpu"

first = 0
last = 10

# import dataset
data = dataset.PersonaChat_with_Images(split='train', image_path='./image_dataset')
data = Subset(data, torch.arange(first,last).tolist())

# split into train / valid dataset
dataset_size = len(data)
train_size = int(dataset_size * 0.8)
valid_size = dataset_size - train_size

train_data, valid_data = random_split(data, [train_size, valid_size])

# load DataLoader
train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
valid_dataloader = DataLoader(valid_data, batch_size=2, collate_fn=dataset.collate_fn)

model = model.GenDial(device).to(device)

num_epoch = 50
lr = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
writer = SummaryWriter()

for epoch in range(num_epoch):
    path = './model_checkpoint/model_'+str(epoch)+'.pt'
    print('epoch: {}'.format(epoch))
    loss = train(train_dataloader, model, optimizer, device)
    writer.add_scalar("Loss/train", loss, epoch)
    loss = test(train_dataloader, model, device)
    writer.add_scalar("Loss/valid", loss, epoch)
    '''
    torch.save({
        'epoch':epoch,
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, path)
    '''
writer.flush()