from torch.utils.data import Dataset, Subset
from datasets import load_dataset
import torch

def collate_fn(samples):
    max_length = 50 # arbitrary set max length as 50
    for sample in samples:
        while len(sample['images']) < max_length: 
            sample['images'].append(None)

        while len(sample['history']) < max_length:
            sample['history'].append(None)

    return samples

class PersonaChat_with_Images(Dataset):
    def __init__(self, split, image_path, init, end):
        self.split = split
        self.image_path = image_path
        
        if self.split=='train':
            self.text_dataset = load_dataset('bavard/personachat_truecased','full',split='train')
        
        elif self.split=='valid':
            self.text_dataset = load_dataset('bavard/personachat_truecased','full',split='validation')

        #elif split=='test':
        #self.text_dataset = Subset(self.text_dataset, torch.arange(init, end))
        #print(self.text_dataset.__getitem__(0))
        
    def __len__(self):
        return self.text_dataset.__len__()

    def __getitem__(self, idx):
        data = self.text_dataset.__getitem__(idx)

        data['history'] = ['<|image|>' + text + '<|endoftext|>' for text in data['history']] # add <|image|>, <|endoftext|> token to history
        data['candidates'][-1] = data['candidates'][-1]+'<|endoftext|>' # add <|endoftext|> token to response

        image_paths = []
        for idx in range(len(data['history'])):
            image_paths.append(self.image_path+'/image_'+str(data['conv_id'])+'_'+str(idx)+'.png')

        data['images'] = image_paths
        data['response'] = [data['candidates'][-1]]

        del(data['personality'])
        del(data['utterance_idx'])
        del(data['candidates'])
        return data

