import torch
import model
import image_generator
import json


device="cuda" if torch.cuda.is_available() else "cpu"

model = model.GenDial(device).to(device)
output = model(["./result/best_image.png", "./result/best_image.png"], ['this is cat', 'oh it is so']) # output size: [3,77,50257]

with open('./vocab.json','r') as f:
    vocab = json.load(f)
vocab = {v:k for k,v in vocab.items()}

words = []
for word in output:
    word_index = torch.argmax(word)
    words.append(vocab[word_index.item()])

print(words)

'''
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
'''