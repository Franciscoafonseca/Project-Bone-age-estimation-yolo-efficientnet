from logs.checkpoint import save_checkpoint, save_best
from logs.logging import log_metrics, log_confusion_matrix, log_prediction_artifacts
from data.resize import model_resize_size
from data.dataset import TeethDataset, StagesDataset, transform_to_real_age
from models.models import *

import torch
import torch.distributed as dist
import os
import numpy as np
import pandas as pd

from torch import nn,optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error, median_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import v2
from utils.gradcam import generate_gradcam_heatmaps

def train(rank, world_size, dataset, config, run):
    print(f"Running on rank {rank}.")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
    else: 
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

    model = get_model(config).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
        print(f'Running on rank {rank}, using GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(rank)}')

    train_sampler, val_sampler, test_sampler, train_loader, val_loader, test_loader, class_weights = create_datasets(dataset, world_size, rank, config)

    if config["optimizer"]=="adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"]=="sgd-m":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )

    # Step LR decay: drop LR by gamma every lr_step_size epochs
    if config["use_lr_scheduler"]:    
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_step_size"],
            gamma=config["lr_gamma"],
        )

    if config['model_task'] == "binary":
        if config.get("use_class_weights", True):
            class_weights = class_weights.to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = nn.BCEWithLogitsLoss()
    elif config['model_task'] in ["multiclass", "stages"]:
        if config.get("use_class_weights", True):
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif config['model_task'] == "regression":
        criterion = nn.MSELoss()

    early_stopper = EarlyStop(patience=config.get("early_stop_patience", 10), min_delta=config.get("early_stop_min_delta", 0))
        
    epoch = 1
    best_f1 = 0
    best_mse = float('inf')
    best_epoch = 1
    model_name = f"{config['model_task']}_{config['model_name']}"

    os.makedirs("results", exist_ok=True)

    while epoch <= config["epochs"]:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        train_epoch(epoch, model, train_loader, optimizer, criterion, device, rank, run, config)

        if config["use_lr_scheduler"]:
            scheduler.step()

        metrics = validate_epoch(epoch, model, val_loader, criterion, device, rank, run, config)
        stop_flag = torch.tensor([0], device=device)

        if rank==0:
            stop = early_stopper.step(metrics['val/loss'])
            if stop:
                print(f"Early stopping at epoch {epoch}")
                save_checkpoint(run, epoch, model, "results", model_name+"_early_stop")
                stop_flag[0] = 1
            else:
                if epoch % config['save_interval'] == 0 :
                    save_checkpoint(run, epoch, model, "results", model_name)
                
                if config['model_task']=="binary" and metrics['val/macro_f1'] > best_f1:
                    best_epoch, best_f1 = save_best(run, epoch, model, metrics['val/macro_f1'], "results", model_name)
                elif config['model_task']=="multiclass" and metrics['val/macro_f1'] > best_f1:
                    best_epoch, best_f1 = save_best(run, epoch, model, metrics['val/macro_f1'], "results", model_name, sum_name="best_val_macro_f1")
                elif config['model_task'] == "regression" and metrics['val/mse'] < best_mse:
                    best_epoch, best_mse = save_best(run, epoch, model, metrics['val/mse'], "results", model_name, sum_name="best_val_mse")
        
        if world_size > 1:
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item() == 1:
                break

        epoch += 1

    export_predictions_to_excel(model, train_loader, criterion, device, config, rank, world_size, output_path=f"results/{run.name}", split="train", run=run)
    export_predictions_to_excel(model, val_loader, criterion, device, config, rank, world_size, output_path=f"results/{run.name}", split="val", run=run)

    if test_loader is not None:
        export_predictions_to_excel(model, test_loader, criterion, device, config, rank, world_size, output_path=f"results/{run.name}", split="test", run=run)
    
    if rank == 0 and config.get("generate_heatmaps", False):
        heatmap_loader = test_loader if test_loader is not None else val_loader
        heatmap_split = "test" if test_loader is not None else "val"
        heatmap_dir = f"results/{run.name}/heatmaps_{heatmap_split}"
        saved_heatmaps = generate_gradcam_heatmaps(
            model=model,
            data_loader=heatmap_loader,
            device=device,
            config=config,
            run=run,
            split=heatmap_split,
            output_dir=heatmap_dir,
            max_images=int(config.get("heatmap_max_images", 16)),
        )
        print(f"Saved {len(saved_heatmaps)} Grad-CAM heatmaps to {heatmap_dir}")

    if world_size > 1:
        dist.destroy_process_group()

    if rank==0:
        if config['model_task'] in ["binary", "multiclass"]:
            print(f"Best validation F1 score: {best_f1:.4f} at epoch {best_epoch}")
        elif config['model_task'] == "regression":
            print(f"Best validation MSE: {best_mse:.4f} at epoch {best_epoch}")

def train_epoch(epoch, model, train_loader, optimizer, criterion, device, rank, run, config):
    model.train()
    total_loss = 0.0
    all_probs = torch.empty(0)
    all_preds = torch.empty(0)
    all_labels = torch.empty(0)

    for batch in train_loader:
        optimizer.zero_grad()
        inputs, labels, img_ids, _, _ = batch
        inputs = inputs.to(device)

        if config['model_task'] == "binary":
            labels = labels.float().to(device)
            logits = model(inputs)
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            predictions = (probs>=0.5).long()

            if not torch.isfinite(logits).all():
                bad_rows = (~torch.isfinite(logits)).nonzero(as_tuple=False)[:, 0].unique()
                print(f"[Rank {rank}] bad dataset ids:", img_ids[bad_rows].tolist())
                print(f"Shape: {tuple(inputs[bad_rows][0].shape)} dtype: {inputs[bad_rows][0].dtype} device: {inputs[bad_rows][0].device}")
                print(f"Min: {inputs[bad_rows][0].min().item()}  Max: {inputs[bad_rows][0].max().item()}")
                print(f"Mean: {inputs[bad_rows][0].mean().item()}  Std: {inputs[bad_rows][0].std().item()}")
                print(f"NaN: {torch.isnan(inputs[bad_rows][0]).any().item()}  Inf: {torch.isinf(inputs[bad_rows][0]).any().item()}")
                raise RuntimeError("Non-finite logits")

        elif config['model_task'] in ["multiclass", "stages"]:
            labels = labels.long().to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)
        elif config['model_task'] == "regression":
            labels = labels.float().to(device)
            logits = model(inputs)
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, labels)
            predictions = logits.detach()
            predictions=transform_to_real_age(predictions)
            labels=transform_to_real_age(labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        if config["model_task"] in ["binary", "multiclass", "stages"]:
            all_probs = torch.cat([all_probs, probs.detach().cpu()]) #probably change these cats to array list

        all_preds = torch.cat([all_preds, predictions.detach().cpu()])
        all_labels = torch.cat([all_labels, labels.detach().cpu()])

    epoch_loss = total_loss / len(train_loader)
    all_probs, all_preds, all_labels, epoch_loss = gather_all_predictions(all_probs, all_preds, all_labels, epoch_loss)

    if rank == 0 and config["model_task"]=="binary":
        metrics=compute_binary_metrics(epoch, all_probs, all_preds, all_labels, epoch_loss, split="train")
        log_metrics(run, metrics)
        log_confusion_matrix(run, all_labels, all_preds, config, split="train")
    elif rank == 0 and config["model_task"]=="regression":
        metrics=compute_regression_metrics(epoch, all_preds, all_labels, epoch_loss, split="train")
        log_metrics(run, metrics)
    elif rank == 0 and config["model_task"] in ["multiclass", "stages"]:
        metrics=compute_multiclass_metrics(epoch, all_probs, all_preds, all_labels, epoch_loss, split="train")
        log_metrics(run, metrics)
        log_confusion_matrix(run, all_labels, all_preds, config, split="train")

def validate_epoch(epoch, model, val_loader, criterion, device, rank, run, config):
    model.eval()
    total_loss = 0.0
    all_probs = torch.empty(0)
    all_preds = torch.empty(0)
    all_labels = torch.empty(0)

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels, _, _, _ = batch
            inputs = inputs.to(device)

            if config['model_task'] == "binary":
                labels = labels.float().to(device)
                logits = model(inputs)
                if logits.ndim == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                predictions = (probs>=0.5).long()
            elif config['model_task'] in ["multiclass", "stages"]:
                labels = labels.long().to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                predictions = probs.argmax(dim=1)
            elif config['model_task'] == "regression":
                labels = labels.float().to(device)
                logits = model(inputs)
                if logits.ndim == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, labels)
                predictions = logits.detach()
                predictions=transform_to_real_age(predictions)
                labels=transform_to_real_age(labels)

            total_loss += loss.item()

            if config["model_task"] in ["binary", "multiclass", "stages"]:
                all_probs = torch.cat([all_probs, probs.detach().cpu()])

            all_preds = torch.cat([all_preds, predictions.detach().cpu()])
            all_labels = torch.cat([all_labels, labels.detach().cpu()])

    epoch_loss = total_loss / len(val_loader)
    all_probs, all_preds, all_labels, epoch_loss = gather_all_predictions(all_probs, all_preds, all_labels, epoch_loss)

    metrics={}
    
    if rank == 0 and config["model_task"]=="binary":
        metrics=compute_binary_metrics(epoch, all_probs, all_preds, all_labels, epoch_loss, split="val")
        log_metrics(run, metrics)
        log_confusion_matrix(run, all_labels, all_preds, config, split="val")
    elif rank == 0 and config["model_task"]=="regression":
        print(all_preds.mean().item(), all_preds.std().item())
        metrics=compute_regression_metrics(epoch, all_preds, all_labels, epoch_loss, split="val")
        log_metrics(run, metrics)
    elif rank == 0 and config["model_task"] in ["multiclass", "stages"]:
        metrics=compute_multiclass_metrics(epoch, all_probs, all_preds, all_labels, epoch_loss, split="val")
        log_metrics(run, metrics)
        log_confusion_matrix(run, all_labels, all_preds, config, split="val")

    return metrics

def export_predictions_to_excel(model, data_loader, criterion, device, config, rank, world_size, output_path, split, run=None):
    model.eval()
    records = []

    all_probs = torch.empty(0)
    all_preds = torch.empty(0)
    all_labels = torch.empty(0)

    total_loss=0.0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels, idx_batch, sex_batch, treatment_batch = batch
            inputs = inputs.to(device)

            if config['model_task'] == "binary":
                labels = labels.float().to(device)
                logits = model(inputs)
                if logits.ndim == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                predictions = (probs>=0.5).long()
            elif config['model_task'] in ["multiclass", "stages"]:
                labels = labels.long().to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                probs = torch.softmax(logits, dim=1)
                predictions = probs.argmax(dim=1)
            elif config['model_task'] == "regression":
                labels = labels.float().to(device)
                logits = model(inputs)
                if logits.ndim == 2 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, labels)
                predictions = logits.detach()
                predictions=transform_to_real_age(predictions)
                labels=transform_to_real_age(labels)

            batch_size = inputs.size(0)
            total_loss += loss.item()

            if config["model_task"] in ["binary", "multiclass"]:
                all_probs = torch.cat([all_probs, probs.detach().cpu()])

            all_preds = torch.cat([all_preds, predictions.detach().cpu()])
            all_labels = torch.cat([all_labels, labels.detach().cpu()])

            for i in range(batch_size):
                idx_val = idx_batch[i].item()

                sex_val = sex_batch[i]
                treatment_val = treatment_batch[i].item()

                label_val = labels[i].item()
                pred_val = predictions[i].item()

                record = {
                    "index": idx_val,
                    "sex": sex_val,
                    "treatment": treatment_val,
                    "label": label_val,
                    "prediction": pred_val,
                }

                if config['model_task'] == "binary":
                    record["probability"] = float(probs[i].item())
                elif config['model_task'] in ["multiclass", "stages"]:
                    record["probability_vector"] = probs[i].cpu().tolist()

                records.append(record)

    metrics={}
    epoch_loss=total_loss/len(data_loader)

    if dist.is_initialized() and world_size > 1:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, records)
        all_probs, all_preds, all_labels, epoch_loss = gather_all_predictions(all_probs, all_preds, all_labels, epoch_loss)
        if rank == 0:
            all_records = [r for sub in gathered for r in sub]
        else:
            return
    else:
        if rank != 0:
            return
        all_records = records
    
    if rank == 0 and config["model_task"]=="binary":
        metrics=compute_binary_metrics("last", all_probs, all_preds, all_labels, epoch_loss, split=split)
    elif rank == 0 and config["model_task"]=="regression":
        metrics=compute_regression_metrics("last", all_preds, all_labels, epoch_loss, split=split)
    elif rank == 0 and config["model_task"] in ["multiclass", "stages"]:
        metrics=compute_multiclass_metrics("last", all_probs, all_preds, all_labels, epoch_loss, split=split)

    if rank == 0:
        metrics.pop("epoch")
        os.makedirs(output_path, exist_ok=True)
        excel_path = output_path + f"/{split}_predictions.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pd.DataFrame(all_records).to_excel(
                writer,
                sheet_name="predictions",
                index=False
            )
            pd.DataFrame([metrics]).to_excel(
                writer,
                sheet_name="metrics",
                index=False
            )

        log_prediction_artifacts(run, all_records, metrics, split=split, excel_path=excel_path)
        print(f"Saved predictions to {output_path}/{split}_predictions.xlsx")

def gather_all_predictions(local_probs, local_preds, local_labels, local_epoch_loss):
    if not dist.is_initialized():
        return local_probs, local_preds, local_labels, local_epoch_loss

    world_size = dist.get_world_size()

    prob_list = [None for _ in range(world_size)]
    pred_list = [None for _ in range(world_size)]
    label_list = [None for _ in range(world_size)]
    epoch_loss_list = [None for _ in range(world_size)]

    if local_probs is not None:
        dist.all_gather_object(prob_list, local_probs)
        all_probs = torch.cat(prob_list, dim=0)
    else:
        all_probs = None

    if local_preds is not None:
        dist.all_gather_object(pred_list, local_preds)
        all_preds = torch.cat(pred_list, dim=0)
    else:
        all_preds = None

    if local_labels is not None:
        dist.all_gather_object(label_list, local_labels)
        all_labels = torch.cat(label_list, dim=0)
    else:
        all_labels = None

    if local_epoch_loss is not None:
        dist.all_gather_object(epoch_loss_list, local_epoch_loss)
        all_epoch_loss = sum(epoch_loss_list) / world_size
    else:
        all_epoch_loss = None

    return all_probs, all_preds, all_labels, all_epoch_loss

def gather_attribute(local_atr):
    if not dist.is_initialized():
        return local_atr

    world_size=dist.get_world_size()
    atr_list = [None for _ in range(world_size)]
    dist.all_gather_object(atr_list, local_atr)
    all_atr = torch.cat(atr_list, dim=0)

    return all_atr
    

def compute_binary_metrics(epoch, all_probs, all_preds, all_labels, avg_loss, split):

    mask = np.isfinite(all_labels) & np.isfinite(all_probs) & np.isfinite(all_preds)
    mask = mask.bool()
    all_labels = all_labels[mask]
    all_preds = all_preds[mask]
    all_probs = all_probs[mask]

    # roc_auc also fails if only one class present
    if len(np.unique(all_labels)) < 2:
        roc_auc=float("nan")
        print("only 1 label present")

    acc=(all_preds == all_labels).float().mean().item()
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1_1 = f1_score(all_labels, all_preds) 
    roc_auc = roc_auc_score(all_labels, all_probs)

    npv = precision_score(all_labels, all_preds, pos_label=0) # precision for negative class
    specificity = recall_score(all_labels, all_preds, pos_label=0) # recall for negative class
    f1_0 = f1_score(all_labels, all_preds, pos_label=0) # f1 for negative class

    precision_macro = precision_score(all_labels, all_preds, average='macro') 
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    metrics = {
        f"epoch": epoch,
        f"{split}/loss": avg_loss,
        f"{split}/accuracy": acc,
        f"{split}/precision": precision,
        f"{split}/recall": recall,
        f"{split}/f1_1": f1_1,
        f"{split}/roc_auc": roc_auc,
        f"{split}/negative_predictive_value": npv,
        f"{split}/specificity": specificity,
        f"{split}/f1_0": f1_0,
        f"{split}/macro_precision": precision_macro,
        f"{split}/macro_recall": recall_macro,
        f"{split}/macro_f1": f1_macro,
    }

    return metrics

def compute_regression_metrics(epoch, all_preds, all_labels, avg_loss, split):

    mask = np.isfinite(all_labels) & np.isfinite(all_preds)
    mask = mask.bool()
    all_labels = all_labels[mask]
    all_preds = all_preds[mask]

    bias = torch.mean(all_preds - all_labels).item()
    median_ae = median_absolute_error(all_labels, all_preds)
    mape = mean_absolute_percentage_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    mse = mean_squared_error(all_labels, all_preds)
    rmse = root_mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)

    metrics = {
        f"epoch": epoch,
        f"{split}/loss": avg_loss,
        f"{split}/bias": bias,
        f"{split}/median_absolute_error": median_ae,
        f"{split}/mape": mape,
        f"{split}/mae": mae,
        f"{split}/mse": mse,
        f"{split}/rmse": rmse,
        f"{split}/r2": r2,
    }

    return metrics

def compute_multiclass_metrics(epoch, all_probs, all_preds, all_labels, avg_loss, split):

    acc=(all_preds == all_labels).float().mean().item()

    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    precision_dict = {f"{split}/precision/{i}": p for i, p in enumerate(precision_score(all_labels, all_preds, average=None))}
    recall_dict = {f"{split}/recall/{i}": r for i, r in enumerate(recall_score(all_labels, all_preds, average=None))}
    f1_dict = {f"{split}/f1/{i}": f for i, f in enumerate(f1_score(all_labels, all_preds, average=None))}

    metrics = {
        f"epoch": epoch,
        f"{split}/loss": avg_loss,
        f"{split}/accuracy": acc,
        f"{split}/macro_precision": macro_precision,
        f"{split}/macro_recall": macro_recall,
        f"{split}/macro_f1": macro_f1,
        **precision_dict,
        **recall_dict,
        **f1_dict,
    }

    return metrics

def create_datasets(idx_dataset, world_size, rank, config):
    train_tfms = v2.Compose([
        v2.Resize(model_resize_size(config["model_name"])),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomAffine(degrees=7, translate=(0.02, 0.02), scale=(0.95, 1.05), fill=0),
        v2.RandomApply([v2.ColorJitter(brightness=0.08, contrast=0.08)], p=0.5),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.2),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.1, scale=(0.01, 0.05))
    ])

    val_tfms = v2.Compose([
        v2.Resize(model_resize_size(config["model_name"])),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    labels = []
    for i in range(len(idx_dataset)):
        _,y,_,_,_ = idx_dataset[i]
        labels.append(y)
    labels = np.array(labels)

    N = len(idx_dataset)
    rng = np.random.RandomState(config.get("seed", 42))
    idx = np.arange(N)
    rng.shuffle(idx)

    train_split = config.get("train_split", 0.8)
    val_split = config.get("val_split", 0.2)
    test_split = config.get("test_split", 0.0)

    total_split = train_split + val_split + test_split
    if not np.isclose(total_split, 1.0, atol=1e-6):
        raise ValueError(f"train_split + val_split + test_split must equal 1.0, got {total_split}")

    seed = config.get("seed", 42)
    task = config.get("model_task", "binary")
    
    if test_split > 0 and task in ["binary", "multiclass", "stages"]:
        # First split off the test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_split,
            random_state=seed,
        )
        train_val_pos, test_pos = next(sss_test.split(idx, labels[idx]))
        remaining_idx = idx[train_val_pos]
        test_idx = idx[test_pos]

        # Then split the remaining data into train/val using the relative proportions
        val_ratio = val_split / (train_split + val_split)
        sss_val = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed,
        )
        pos_train, pos_val = next(sss_val.split(remaining_idx, labels[remaining_idx]))
        train_idx = remaining_idx[pos_train]
        val_idx = remaining_idx[pos_val]
        
    elif test_split == 0 and task in ["binary", "multiclass", "stages"]:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_split,
            random_state=seed,
        )
        pos_train, pos_val = next(sss.split(idx, labels[idx]))
        train_idx = idx[pos_train]
        val_idx   = idx[pos_val]
        test_idx = None

    elif test_split>0 and task=="regression":
        ss_test = ShuffleSplit(
                n_splits=1,
                test_size=test_split,
                random_state=seed,
            )
        train_val_pos, test_pos = next(ss_test.split(idx))
        remaining_idx = idx[train_val_pos]
        test_idx      = idx[test_pos]

        val_ratio = val_split / (train_split + val_split)
        ss_val = ShuffleSplit(
            n_splits=1,
            test_size=val_ratio,
            random_state=seed,
        )
        pos_train, pos_val = next(ss_val.split(remaining_idx))
        train_idx = remaining_idx[pos_train]
        val_idx   = remaining_idx[pos_val]

    elif test_split == 0 and task == "regression":
        ss = ShuffleSplit(
            n_splits=1,
            test_size=val_split,
            random_state=seed,
        )
        pos_train, pos_val = next(ss.split(idx))
        train_idx = idx[pos_train]
        val_idx   = idx[pos_val]
        test_idx  = None

    use_class_weights = config.get("use_class_weights", False)

    if task == "binary" and use_class_weights:
        train_labels = labels[train_idx]
        pos = int((train_labels == 1).sum())
        neg = int((train_labels == 0).sum())
        class_weights = torch.tensor([neg / max(pos, 1)], dtype=torch.float32)
    elif task in ["multiclass", "stages"] and use_class_weights:
        train_labels = labels[train_idx]          
        num_classes = int(train_labels.max()) + 1

        class_counts = np.bincount(train_labels, minlength=num_classes)
        class_weights = class_counts.sum() / (num_classes * np.maximum(class_counts, 1))

        class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = torch.tensor([1.0], dtype=torch.float32)

    if task != "stages":
        train_ds = TeethDataset(
            ann_file="data/" + config["ann_file"],
            img_dir=os.path.join("data/preprocessed/", config["image_processing"]),
            transform=train_tfms,
            config=config
        )
        val_ds = TeethDataset(
            ann_file="data/" + config["ann_file"],
            img_dir=os.path.join("data/preprocessed/", config["image_processing"]),
            transform=val_tfms,
            config=config
        )
    else:
        train_ds = StagesDataset(
            ann_file="data/" + config["ann_file"],
            img_dir="data/stages_images/",
            transform=train_tfms,
            config=config
        )
        val_ds = StagesDataset(
            ann_file="data/" + config["ann_file"],
            img_dir="data/stages_images/",
            transform=val_tfms,
            config=config
        )


    train_dataset = Subset(train_ds, train_idx.tolist())
    val_dataset   = Subset(val_ds,   val_idx.tolist())
    test_dataset  = Subset(val_ds,   test_idx.tolist()) if test_idx is not None else None
    print(len(train_idx))
    print(len(val_idx))
    print(len(test_idx))
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler  = DistributedSampler(test_dataset,  num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if test_dataset is not None else None

    train_loader = DataLoader(
        train_dataset, batch_size=config.get("batch_size", 128), shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=config.get("num_workers", 4), pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get("batch_size", 128), shuffle=False,
        sampler=val_sampler, num_workers=config.get("num_workers", 4), pin_memory=True, drop_last=False
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset, batch_size=config.get("batch_size", 128), shuffle=False,
            sampler=test_sampler, num_workers=config.get("num_workers", 4), pin_memory=True, drop_last=False
        )

    return train_sampler, val_sampler, test_sampler, train_loader, val_loader, test_loader, class_weights

class EarlyStop:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0 
        self.best_loss = float('inf')
    
    def step(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.count = 0
        elif val_loss + self.min_delta > self.best_loss:
            self.count += 1
        return self.count >= self.patience
