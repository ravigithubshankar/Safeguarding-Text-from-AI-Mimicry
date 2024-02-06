import wandb

def artifact(run_name, artifact_name, path, data_type="dataset"):
    run = wandb.init(project=COMP_ID, name=run_name, config=CONFIG)
    artifact = wandb.Artifact(name=artifact_name, type=data_type)
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    wandb.finish()
    print(f"🐝 Artifact {artifact_name} has been saved successfully.")

def wandb_plot(x_data=None, y_data=None, x_name=None, y_name=None, title=None, log=None, plot="line"):
    data = [[label, val] for (label, val) in zip(x_data, y_data)]
    table = wandb.Table(data=data, columns=[x_name, y_name])
    
    if plot == "line":
        wandb.log({log: wandb.plot.line(table, x_name, y_name, title=title)})
    elif plot == "bar":
        wandb.log({log: wandb.plot.bar(table, x_name, y_name, title=title)})
    elif plot == "scatter":
        wandb.log({log: wandb.plot.scatter(table, x_name, y_name, title=title)})
    print(f"📊 Plot {plot} has been logged successfully.")

def wandb_hist(x_data=None, x_name=None, title=None, log=None):
    data = [[x] for x in x_data]
    table = wandb.Table(data=data, columns=[x_name])
    wandb.log({log: wandb.plot.histogram(table, x_name, title=title)})
    print(f"📊 Histogram {log} has been logged successfully.")

def log_predictions(predictions, targets, epoch):
    data = [[idx, target, pred] for idx, (target, pred) in enumerate(zip(targets, predictions))]
    table = wandb.Table(data=data, columns=["Index", "Target", "Prediction"])
    wandb.log({"predictions_epoch_" + str(epoch): table})


sweep_config={
    "method":"random",
    "metric":{
        "name":"valid_ROC",
        "goal":"maximize"
    },
    "parameters":{
        "batch_size":{
            "values":[8,16,32]
        },
        "layer_size":{
            "values":[100,150,200,300,768,816,987]
        }
    },
    "lr":{
        "distribution":"uniform",
                        "max":0.3 ,
                        "min":0.2
    },
    "epochs":{
        "values":[1,3,5,8,10]
    }
}
print(sweep_config)
sweep_id=wandb.sweep(sweep_config,project=COMP_ID)

start=time()

wandb.agent(sweep_id,trainer,count=10)
print(f"Sweeping Took:", round((time()-start)/60,1),"mins")

