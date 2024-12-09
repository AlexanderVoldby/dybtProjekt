import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, models
from tqdm import tqdm
from split import deterministic_split, random_split
from image_dataset import ImageDataset
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

def evaluate_model(model_name, datasets, variants, base_dir="/dtu/blackhole/12/145234/", batch_size=32):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    _datasets = []
    for dataset, variant in zip(datasets, variants):
        N_train_pets_real = 471
        N_train_cars_real = 2514
        n_train = N_train_cars_real if dataset=="cars" else N_train_pets_real

        target_dir = os.path.join(base_dir, f"{dataset}/{variant}")

        if dataset == "pets" and variant == "real-fewshot":
            target_dir = os.path.join(target_dir, "seed0")
            test_ratio = 0.2
            _, test_files = deterministic_split(target_dir, test_ratio=test_ratio)
            dataset_obj = ImageDataset(file_list=test_files, transform=test_transform, synthetic_label=0)
            _datasets.append(dataset_obj)
        elif variant == "real-fewshot":
            target_dir = os.path.join(target_dir, "best")
            test_ratio = 0.2
            _, test_files = deterministic_split(target_dir, test_ratio=test_ratio)
            dataset_obj = ImageDataset(file_list=test_files, transform=test_transform, synthetic_label=0)
            _datasets.append(dataset_obj)
        else:
            target_dir = os.path.join(target_dir, "train")
            _, test_files = random_split(target_dir, n_train)
            dataset_obj = ImageDataset(file_list=test_files, transform=test_transform, synthetic_label=1)
            _datasets.append(dataset_obj)
        if not os.path.isdir(target_dir):
            raise ValueError(f"Target directory does not exist: {target_dir}")


    combined_dataset = ConcatDataset(_datasets)
    dataloader_test = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model_path = f"models/{model_name}.pth"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Saved model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader_test, desc="Inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report = classification_report(all_labels, all_preds, output_dict=True, target_names=['Real', "Synthetic"])
    metrics = {
        "accuracy": report["accuracy"],  # Overall accuracy
        "class_0": {
            "precision": report["Real"]["precision"],
            "recall": report["Real"]["recall"],
            "f1": report["Real"]["f1-score"],
        },
        "class_1": {
            "precision": report["Synthetic"]["precision"],
            "recall": report["Synthetic"]["recall"],
            "f1": report["Synthetic"]["f1-score"],
        },
    }

    return metrics
