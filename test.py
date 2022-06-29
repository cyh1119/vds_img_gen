import torch
import model
import dataloader
import json
from torch.utils.data import DataLoader, Subset

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            # prediction
            _, prob_gold = model(data['images'], data['history'], data['response'])
            # 손실함수 고치기
            loss = loss_fn(prob_gold)
            test_loss = loss.item()
        
        print("Test Loss: {}".format(test_loss/size))

    return test_loss

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.GenDial(device)
model.load_state_dict(torch.load("./model_checkpoint/model_0.pt")['model_state_dict'])
model = model.to(device)
model.eval()

first = 0
last = 33000

train_data = dataloader.PersonaChat_with_Images(split='train', image_path='./image_dataset', init=first, end=last)
train_data = Subset(train_data, torch.arange(first,last).tolist())
train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=dataloader.collate_fn)

data = train_data[0]

prediction, response_token = model(data)

prediction = prediction.reshape(-1, 50257)
prediction_idx = torch.argmax(prediction, dim=-1)

with open('./vocab.json','r') as f:
    vocab = json.load(f)
vocab = {v:k for k,v in vocab.items()}

output = []
for idx in prediction_idx:
    output.append(vocab[idx.item()])

print(output)
print(data['response'])