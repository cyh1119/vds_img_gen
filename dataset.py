import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
from datasets import load_dataset

class tokenizer():
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token="<|image|>")

    def tokenize(self, texts):
        '''
        input: type->list, shape->(texts)
        output: type->list, shape->(texts, token_ids)
        '''
        result = []
        for text in texts:
            result.append(self.tokenizer(text, return_tensors='pt')['input_ids'].squeeze(0))
        return result


def collate_fn(samples):
    batch_image = [sample['image'] for sample in samples]
    batch_text = [sample['text'] for sample in samples]
    batch_response = [sample['response'] for sample in samples]

    batch = {'image':batch_image, 'text':batch_text, 'response':batch_response}

    return batch

class PersonaChat_with_Images(Dataset):
    def __init__(self, split, image_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.split = split
        self.image_path = image_path
        
        if self.split=='train':
            self.text_dataset = load_dataset('bavard/personachat_truecased','full',split='train')
        
        elif self.split=='valid':
            self.text_dataset = load_dataset('bavard/personachat_truecased','full',split='validation')

        #elif split=='test':
        #self.text_dataset = Subset(self.text_dataset, torch.arange(init, end))
        #print(self.text_dataset.__getitem__(0))
        
        self.tokenizer = tokenizer(self.device)

    def __len__(self):
        return self.text_dataset.__len__()

    def __getitem__(self, idx):
        data = self.text_dataset.__getitem__(idx)
        
        response = [data['candidates'][-1]+'<|endoftext|>'] # add <|endoftext|> token to response
        text = ['<|image|>' + text + '<|endoftext|>' for text in data['history']] # add <|image|>, <|endoftext|> token to history
        text.append(response[0])

        tokenized_text = self.tokenizer.tokenize(text)
        tokenized_response = self.tokenizer.tokenize(response)
        image_paths = []
        for idx in range(len(data['history'])):
            image_paths.append(self.image_path+'/image_'+str(data['conv_id'])+'_'+str(idx)+'.png')

        sample = {'image': image_paths,
                'text': tokenized_text,
                'response': tokenized_response
                }
        
        return sample

