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

