import matplotlib.pyplot as plt
import seaborn as sns

def show_values_on_bars(axs, h_v="v", space=0.4):
    """
    Attach a text label above each bar in *axs*, displaying its height.

    Parameters:
    axs: Axes object or array of Axes objects.
        The axes to draw the annotations onto.
    h_v: str, optional (default: 'v')
        Whether to show values 'v'ertically or 'h'orizontally.
    space: float, optional (default: 0.4)
        Space between the text and the bar.

    Returns:
    None
    """
    if h_v == "v":
        for ax in axs:
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, value, ha="center", va="bottom")
    elif h_v == "h":
        for ax in axs:
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, value, ha="left", va="center")

my_colors = ["#A63D40", "#E9B872", "#90A959", "#6494AA", "#1C6E8C"]
plt.figure(figsize=(20, 10))
figure = sns.countplot(data=train_prompts,
                       x="prompt_id", palette=my_colors[1:])
show_values_on_bars([figure], h_v="v", space=0.4)
plt.title('[Comp Data] No. of texts based on prompt ID',
          weight="bold", size=20)

plt.xlabel("Prompt ID", size=18, weight="bold")
plt.ylabel("Count", size=18, weight="bold")

sns.despine(right=True, top=True, left=True)
plt.show()

#from preprocess_text(text-preprocessing)
print(f"Original:")
print(train_data["text"][1][:1000],"\n")

print(f"Cleaned:")
print(preprocess_text(train_data["text"][1][:1000]))

#dataloaders
data = CustomDataset(train_data.sample(n=6), MAX_LEN)
loader = DataLoader(data, batch_size=8, shuffle=False)

for i, batch in enumerate(loader):
    ids, mask, target = batch['input_ids'], batch['attention_mask'], batch['target']
    ids, mask, target = to_device(batch) 
    print(f"Batch: {i}")
    print("ids:", ids)
    print("target:", target)
    print("=" * 50)

# Define ANSI escape codes for color formatting
class clr:
    S = '\033[92m'  # green
    E = '\033[0m'   # reset

# Your code continues...
model = Transformers(BERT_MODEL, layer_size=200).to(DEVICE)
model.train()
for i, batch in enumerate(loader):
    ids, mask, target = to_device(batch)
    break

print(clr.S + "Input data shape:" + clr.E, len(ids), "texts.", "\n")

out = model(ids, mask, verbose=True)

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key= user_secrets.get_secret("kaggle")
os.environ["WANDB_API_KEY"] = api_key

# Login to wandb
wandb.login()

def trainer():
    config_defaults = {"model": BERT_MODEL,
                       "batch_size": BATCH_SIZE,
                       "layer_size" : LAYER_SIZE,
                       "lr" : LR,
                       "epochs" : EPOCHS}
    config_defaults.update(CONFIG)
    
    with wandb.init(project=COMP_ID, config=config_defaults):
        config = wandb.config
    
        # loaders
        train_loader, valid_loader = get_loader(train, valid, 
                                                 batch_size=config.batch_size)

        # model
        model = Transformers(BERT_MODEL, layer_size=config.layer_size).to(DEVICE)
        optimizer = Adam(model.parameters(), lr=config.lr)
        criterion = nn.MSELoss()

        print(clr.S + f"--- lr {config.lr} | epochs {config.epochs} ---" + clr.E)
        print(clr.S + f"--- batch {config.batch_size} | layer {config.layer_size} ---" + clr.E)

        for e in range(config.epochs):
            print(clr.S + f"- Epoch {e} -" + clr.E)

            # -Train the Model-
            model.train()

            losses = []
            for i, batch in enumerate(loader):
                ids, mask, target = to_device(batch) 

                optimizer.zero_grad()
                out = model(ids, mask)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                losses.append(loss.cpu().detach().numpy().tolist())

            # log training loss
            train_loss = np.mean(losses)
            wandb.log({"mean_train_loss": float(train_loss)}, step=e)
            print("Mean Train Loss:", train_loss)


            # -Evaluate the Model-
            model.eval() 

            valid_preds, valid_targets = [], []
            with torch.no_grad():
                for i, batch in enumerate(valid_loader):
                    ids, mask, target = to_device(batch)

                    out = model(ids, mask)

                    valid_preds.append(out.detach().cpu().numpy().ravel())
                    valid_targets.append(target.detach().cpu().numpy().ravel())


            # save results
            valid_preds = np.concatenate(valid_preds)
            valid_targets = np.concatenate(valid_targets)
            roc = roc_auc_score(valid_targets, valid_preds)

            wandb.log({"valid_ROC": float(roc),
                       "epoch": e})
            print(f"Valid ROC: {roc}")
            log_predictions(valid_preds,valid_targets,e)


train, valid = train_test_split(train_data, test_size=0.2, random_state=SEED)

print(clr.S+"Train:"+clr.E, "\n",
      train["generated"].value_counts() / train["generated"].value_counts().sum())
print(clr.S+"Valid:"+clr.E, "\n",
      valid["generated"].value_counts() / valid["generated"].value_counts().sum())

!export CUDA_LAUNCH_BLOCKING=1

trainer()
