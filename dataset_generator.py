import image_generator
from PIL import Image
import os
from datasets import list_datasets, load_dataset

dataset_train = load_dataset('bavard/personachat_truecased','full',split='train')
#dataset_val = load_dataset('bavard/personachat_truecased','full',split='validation')
generator = image_generator.dall_e()

result_path = './result'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

for idx, text in enumerate(dataset_train[0]['history']):
    print(f"Text: {text}\n")
    image_generated = generator.generate_best_image(text, 1)
    image_generated.save(result_path+'/image'+str(idx)+'.png','png')
