import image_generator
from PIL import Image
import os
import sys
from datasets import list_datasets, load_dataset

if len(sys.argv) != 3:
    print("Insufficient Arguments")
    sys.exit()

data_init_bound = int(sys.argv[1])
data_end_bound = int(sys.argv[2])
length = data_end_bound - data_init_bound

print('process image from {} to {} index'.format(data_init_bound, data_end_bound))

dataset_train = load_dataset('bavard/personachat_truecased','full',split='train')
dataset_val = load_dataset('bavard/personachat_truecased','full',split='validation')

generator = image_generator.dall_e()

result_path = './image_dataset'
if not os.path.isdir(result_path):
    os.mkdir(result_path)

for idx, data in enumerate(dataset_train):
    print("{} / {}".format(idx, length))
    if idx >= data_init_bound and idx < data_end_bound:
        if data['utterance_idx'] == 0:
            index = data['utterance_idx']*2
            print("Text: {}\n".format(data['history'][index]))
            image_generated = generator.generate_best_image(data['history'][data['utterance_idx']], n_predictions=1)
            image_generated.save(result_path+'/image_'+str(data['conv_id'])+'_'+str(0)+'.png','png') 

        else:
            index = data['utterance_idx']*2
            print("Text: {}\n".format(data['history'][index-1]))
            image_generated = generator.generate_best_image(data['history'][index-1], n_predictions=1)
            image_generated.save(result_path+'/image_'+str(data['conv_id'])+'_'+str(index-1)+'.png','png')

            print("Text: {}\n".format(data['history'][index]))
            image_generated = generator.generate_best_image(data['history'][index], n_predictions=1)
            image_generated.save(result_path+'/image_'+str(data['conv_id'])+'_'+str(index)+'.png','png')

    
    elif idx >= data_end_bound:
         break
