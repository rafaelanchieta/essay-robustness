import random
import numpy as np
import re
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from util import Corpus

seed_val = 24

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
torch.backends.cudnn.deterministic = True

c = Corpus()
training, development, _ = c.read_splits()
prompts = pd.read_csv('essay-br/prompts.csv')


train_merge = pd.merge(training, prompts, left_on="prompt", right_on="id")
train = train_merge[['essay','description','competence']]

dev_merge = pd.merge(development, prompts, left_on="prompt", right_on="id")
dev = dev_merge[['essay','description','competence']]


# print(training['competence'].str[0])
# exit()

tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')


#Classe de manipulação de dataset:
class Dataset(Dataset):

    #Construtor da classe:
    def __init__(self, essays, prompts, tokenizer, labels, max_length=512):
        #Armazene as entradas que serão passadas ao modelo:
        self.essays = essays['essay'].values  # reviews["review_text"].values

        self.prompts = prompts

        #Armazene as labels que serão utilizadas para treino/validação/teste:
        self.labels = labels.values

        #Armazene o tokenizador:
        self.tokenizer = tokenizer

        #Armazene o tamanho máximo das sentenças:
        self.max_length = max_length

    #Retorna o número de instâncias:
    def __len__(self):
        return len(self.essays)

    #Retorna uma instância completa com base num índice:
    def __getitem__(self, index):
        #Obtenha a entrada do índice pertinente:
        #review = self.reviews[index]
        essay = self.essays[index]

        prompt = self.prompts[index]

        #Obtenha a label do índice pertinente:
        label = self.labels[index]

        #Tokenize a entrada:
        encoding = self.tokenizer(
          text_pair=prompt.tolist(),  # .tolist(),
          text=essay,
          add_special_tokens=True,
          max_length=self.max_length,
          return_token_type_ids=False,
          return_overflowing_tokens=False,
          padding='max_length',
          #pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )

        #Obtenha os códigos numéricos da sentença:
        input_ids = encoding['input_ids'].flatten()

        #Obtenha os códigos numéricos dos token types:
        # token_type_ids = encoding['token_type_ids'].flatten()

        #Obtenha a máscara de atenção da sentença:
        attention_mask = encoding['attention_mask'].flatten()

        #Transforme a label da instância em um tensor:
        label_tensor = torch.tensor(label, dtype=torch.float)


        #Retorne um dicionário com estes dados:
        return {
          'input_ids': input_ids,
          #'token_type_ids': token_type_ids,
          'attention_mask': attention_mask,
          'labels': label_tensor
        }


class BERTTimbauRegression(nn.Module):
    def __init__(self):
        super(BERTTimbauRegression, self).__init__()
        # Load the pre-trained BERT model
        self.bert = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        # Dropout layer
        self.dropout = nn.Dropout(0.4)

        # A Linear layer to get 4 continuous values (PROPOR)

        self.regressor = nn.Linear(1024, 1)

    def forward(self, input_ids, attention_mask):
        # Get the output from BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        pooled_output = outputs.pooler_output

        # Apply dropout
        pooled_output = self.dropout(pooled_output)

        # Pass through the regressor
        return self.regressor(pooled_output)


max_length = 512
batch_size = 8  # 16

train_set = Dataset(train, train[['description']].values, tokenizer, train['competence'].str[0], max_length)
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)

val_set = Dataset(dev, dev[['description']].values, tokenizer, dev['competence'].str[0], max_length)
val_loader = DataLoader(val_set, batch_size = batch_size, shuffle=False)


#Crie um dataset de testes:
dtoy = Dataset(train, train[['description']].values, tokenizer, train['competence'].str[0], max_length=45)

#Pegue uma instância do dataset:
data_inst = next(iter(dtoy))

#Imprima os componentes da instância:
print("Input IDs:", data_inst['input_ids'])
# print("Token Type IDs:", data_inst['token_type_ids'])
print("Attention Mask:", data_inst['attention_mask'])

# Crie o modelo:
from transformers import AdamW, get_linear_schedule_with_warmup

device = "mps" if torch.backends.mps.is_available() else "cpu"
BertPROPOR = BERTTimbauRegression()
BertPROPOR.to(device)

epochs = 6
total_steps = len(train_loader) * epochs
loss_function = torch.nn.MSELoss()
optimizer = AdamW(BertPROPOR.parameters(), lr=4e-5, eps=1e-8)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

import tqdm
def train_model(model, data_loader, loss_fn, optimizer, scheduler):
    #Coloque o modelo em modo de treinamento:
    model.train()

    #Inicialize o erro total da epoch:
    total_loss = 0
    total_preds = []

    #Para cada batch do data_loader, faça:
    for d in tqdm.tqdm(data_loader, desc='training...'):
        #Obtenha os dados da batch:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

        #Passe os dados pelo modelo:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        # for o in outputs:
        #     print(o.item())
        # outputs = outputs.cpu().detach().numpy()
        # print(outputs)
        # exit()
        # Obtenha as predições:
        # _, preds = torch.max(outputs, dim=1)
        # total_preds.extend([p.item() for p in preds])
        # print(total_preds)
        # print(outputs)
        # outputs = [round_to_nearest_grade(score) for score in outputs]
        # print(outputs)
        # outputs = np.array(outputs)
        # outputs = torch.FloatTensor(outputs).cuda()
        # outputs = torch.from_numpy(np.array(outputs)).float().to(device)
        #Calcule o erro:
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        #Propague o erro para o modelo, promovendo aprendizado:
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(data_loader)


def test_model(model, data_loader, loss_fn):
    #Coloque o modelo em modo de treinamento:
    model.eval()

    #Inicialize o erro total da epoch:
    total_loss = 0
    total_preds = []
    with torch.no_grad():
      #Para cada batch do data_loader, faça:
      for d in tqdm.tqdm(data_loader, desc='evaluating...'):
          #Obtenha os dados da batch:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          labels = d["labels"].to(device)

          #Passe os dados pelo modelo:
          outputs = model(input_ids=input_ids, attention_mask=attention_mask)

          #Obtenha as predições:
          #_, preds = torch.max(outputs, dim=1)
          # total_preds.extend([p.item() for p in preds])

          #Calcule o erro:
          loss = loss_fn(outputs, labels)
          total_loss += loss.item()

          # preds = outputs # .squeeze(1)
          preds = outputs.cpu().detach().numpy()
          total_preds.extend(preds)

    return total_loss / len(data_loader), total_preds

import os
output_dir = 'model_save/'


# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Treine o modelo:

best_valid_loss = float('inf')

losses_va, losses_tr = [],[]

for i in range(epochs):
    #Treine em dados de treinamento:
    #print('\nTreinando o modelo, epoch ', i)
    total_loss_tr = train_model(BertPROPOR, train_loader, loss_function, optimizer, scheduler)

    #Valide em dados de validação:
    #print('Validando o modelo, epoch ', i)
    total_loss_va, _ = test_model(BertPROPOR, val_loader, loss_function)

    losses_tr.append(total_loss_tr)
    losses_va.append(total_loss_va)

    if total_loss_va < best_valid_loss:
      print(f"Época: {i}")
      best_valid_loss = total_loss_va
      torch.save(BertPROPOR.state_dict(), output_dir + 'model_finetuned_bert.pt')

    #Imprima os erros de treinamento/validação:
    print('Erro de treinamento:', total_loss_tr)
    print('Erro de validação:', total_loss_va)