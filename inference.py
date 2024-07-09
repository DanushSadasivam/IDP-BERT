import torch
import yaml
import os
from model.network import create_model
from model.utils import test
from transformers import BertTokenizerFast
import numpy as np
from data.dataset import IDPBERTDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

with np.load(f'./data/inference_data.npz') as test:
    test_X = test['X']

tokenizer = BertTokenizerFast.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)

encoded_te = tokenizer(list(test_X), padding='max_length', max_length=602, return_attention_mask=True)
test_encodings, test_attention_masks = encoded_te['input_ids'], encoded_te['attention_mask']

test_dataset = IDPBERTDataset(input_ids=test_encodings, attention_masks=test_attention_masks, labels=np.zeros(test_X.shape))

test_loader = DataLoader(
    test_dataset,
    batch_size=config['batch_size'],
    shuffle=False
)

model = create_model(config)

model.load_state_dict(torch.load(f'./checkpoints/{config["run_name"]}/model.pt')['model_state_dict'], strict=False)
predictions = test(model, test_loader, device)

os.makedirs(f'./data/inference_results', exist_ok=True)

np.savez(f'./data/inference_results/{config["run_name"]}.npz', predictions = predictions)
    