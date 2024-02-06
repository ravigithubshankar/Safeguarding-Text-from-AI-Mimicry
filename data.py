train_essays=pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")
train_prompts=pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_prompts.csv")

train_generated=pd.read_csv("/kaggle/input/llm-generated-essays/ai_generated_train_essays.csv")
train_gpt=pd.read_csv("/kaggle/input/llm-generated-essays/ai_generated_train_essays_gpt-4.csv")
train_gen = pd.concat([train_generated, train_gpt], axis=0)\
                .reset_index(drop=True)

print(train_essays.head(5),"\n")
print(train_prompts.head(5),"\n")
print(train_generated.head(5),"\n")
print(train_gpt.head(5),"\n")

for k in range(len(train_prompts)):
    intro=train_prompts.loc[k,:]
    print(f"Prompt {k}","\n")
    print(f"Prompt_name:",intro.prompt_name,"\n")
    print(f"Instructions:",intro.instructions,"\n")
    print(f"Source_text:",intro.source_text,"\n")


#merge all data into one data file by using ETL techniques
train_data = pd.concat([train_essays, train_gen], axis=0)\
                .reset_index(drop=True)

SEED=34
MAX_LEN=512
LAYER_SIZE=712
NVALID=0.3 
LR=0.007
BATCH_SIZE=32
EPOCHS=10
BERT_MODEL='distilbert-base-uncased'

