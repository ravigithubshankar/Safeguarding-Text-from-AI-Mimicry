from wordcloud import WordCloud

# 1. Distribution of Essay Lengths
plt.figure(figsize=(10, 6))
sns.histplot(train_essays['text'].apply(len), bins=30, kde=True, color=my_colors[0])
plt.title('Distribution of Essay Lengths')
plt.xlabel('Length of Essays')
plt.ylabel('Count')
plt.show()

# 2. Word Cloud of Essays
all_text = ' '.join(train_essays['text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(12, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Essays')
plt.show()

# 3. Distribution of Generated Essays Across Prompts
plt.figure(figsize=(12, 8))
sns.countplot(x='prompt_id', data=train_gen, hue='generated', palette=my_colors[1:])
plt.title('Distribution of Generated Essays Across Prompts')
plt.xlabel('Prompt ID')
plt.ylabel('Count')
plt.legend(title='Generated', labels=['Non-Generated', 'Generated'])
plt.show()

# 4. Comparing Lengths of Generated vs. Non-Generated Essays
plt.figure(figsize=(10, 6))
sns.histplot(train_gen['text'].apply(len), bins=30, kde=True, color=my_colors[2], alpha=0.6, label='Generated')
sns.histplot(train_essays[train_essays['generated'] == 0]['text'].apply(len), bins=30, kde=True, color=my_colors[3], alpha=0.6, label='Non-Generated')
plt.title('Comparison of Essay Lengths: Generated vs. Non-Generated')
plt.xlabel('Length of Essays')
plt.ylabel('Count')
plt.legend()
plt.show()

# 5. Comparison of Essay Lengths Across Prompts
plt.figure(figsize=(12, 8))
sns.boxplot(x='prompt_id', y=train_essays['text'].apply(len), data=train_essays, palette=my_colors[1:])
plt.title('Comparison of Essay Lengths Across Prompts')
plt.xlabel('Prompt ID')
plt.ylabel('Length of Essays')
plt.show()
#text-preproecssing
def preprocess_text(input_text):
    translator=str.maketrans('','',string.punctuation)
    text_no_punct=input_text.translate(translator)
    cleaned_text=re.sub(r'\s+',' ', text_no_punct)
    cleaned_text=cleaned_text.lower()
    return cleaned_text

from torch.utils.data import Dataset, DataLoader
import torch

#data loaders
class CustomDataset(Dataset):
    def __init__(self, data, max_len):
        super(CustomDataset, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL)
        self.max_len = max_len
        self.data = data
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = preprocess_text(row.text)
        inputs = self.tokenizer(text, max_length=self.max_len, padding="max_length", truncation=True)
        ids = torch.tensor(inputs['input_ids'], dtype=torch.long)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long)
        target = torch.tensor(row.generated, dtype=torch.float)
        
        return {
            'input_ids': ids,
            'attention_mask': mask,
            'target': target
        }

def to_device(batch):
    ids, mask, target = batch['input_ids'], batch['attention_mask'], batch['target']
    return ids.to(DEVICE), mask.to(DEVICE), target.to(DEVICE)

#transformer
class Transformers(nn.Module):
    def __init__(self,bert_model,layer_size):
        super(Transformers,self).__init__()
        self.Model=DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.ln1=nn.Linear(768,layer_size)
        self.ln2=nn.Linear(layer_size,1)
        
    def forward(self,ids,mask,verbose=False):
        text=self.Model(ids,mask)[0]
        out_ln1=self.ln1(text)
        out_ln2=self.ln2(out_ln1)
        out=out_ln2[:,0,:]
        
        if verbose:
            print("===============")
            print(clr.S+"Text Out Shape:"+clr.E, text.shape)
            print(clr.S+"After LN1 Shape:"+clr.E, out_ln1.shape)
            print(clr.S+"After LN2 Shape:"+clr.E, out_ln2.shape)
            print(clr.S+"Output Shape:"+clr.E, out.shape)
            
        return out


def get_loader(train,valid,batch_size):
    train_dataset=CustomDataset(train,MAX_LEN)
    valid_dataset=CustomDataset(valid,MAX_LEN)
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    valid_loader=DataLoader(valid_dataset,batch_size=batch_size,num_workers=8,shuffle=False)
    return train_loader,valid_loader
  
