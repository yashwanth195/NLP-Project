import json
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import commentDataset, predictionDataset
from model import commentClassifier

seed = 67890

def set_seeds(seed=seed):
    '''
    Function to set random seeds.
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_param_no(model):
    '''
    Calculates and returns total and trainable parameters in the model.
    '''
    total_num = sum(p.numel() for p in model.parameters())  # Total params
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable params
    return {'Total': total_num, 'Trainable': trainable_num}


def get_train_loaders(train_address, pretrained_model, batch_size=32, num_workers=0, pin_memory=False):
    '''
    Loads training data into a DataLoader.
    '''
    trainset = commentDataset(train_address, pretrained_model)
    print("train_set_num:", len(trainset))  

    # Create DataLoader for training data
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=trainset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader


def get_test_loaders(test_dataset, test_mode, pretrained_model, batch_size=32, num_workers=0, pin_memory=False):
    '''
    Function to load test data into a DataLoader.
    '''
    testset = predictionDataset(test_dataset, test_mode, pretrained_model)  # Load the dataset
    print("test_set_num:", len(testset))  # Print number of samples in the test set

    # Create DataLoader for test data
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return test_loader


def train_model(model, loss_function, dataloader, optimizer, epoch):
    '''
    Function to train the model.
    '''
    losses, preds, labels = [], [], []  

    model.train()  # Set model to training mode
    set_seeds(seed + epoch)  

    for data in tqdm(dataloader):
        optimizer.zero_grad() 

        # Extract batch data and move to GPU
        input_ids, att_mask, label, comment_len, punc_num = [d.cuda() for d in data[:5]]
        pair_ids, comments = data[-2:]

        # Forward pass through the model
        logits = model(input_ids, att_mask, comment_len, punc_num)
        loss = loss_function(logits, label)  # Compute loss
        preds.append(torch.argmax(logits, 1).cpu().numpy()) 
        labels.append(label.cpu().numpy()) 
        losses.append(loss.item()) 

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

    # Concatenate all predictions and labels across batches
    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    # Calculate average metrics
    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_f1 = round(f1_score(labels, preds, average='macro') * 100, 2)
    avg_precision = round(precision_score(labels, preds, average='macro') * 100, 2)
    avg_recall = round(recall_score(labels, preds, average='macro') * 100, 2)
    each_f1 = f1_score(labels, preds, average=None)
    each_precision = precision_score(labels, preds, average=None)
    each_recall = recall_score(labels, preds, average=None)

    return avg_loss, avg_f1, avg_precision, avg_recall, each_f1, each_precision, each_recall


def prediction(model, dataloader):
    '''
    Function for making predictions using the model.
    '''
    preds = []  
    model.eval()  
    set_seeds(seed)  

    # Disable gradient calculations for inference
    with torch.no_grad():
        for data in tqdm(dataloader):
            # Extract batch data and move to GPU
            input_ids, att_mask, comment_len, punc_num = [d.cuda() for d in data[:4]]
            pair_ids = data[-1]  

            # Forward pass to obtain predictions
            logits = model(input_ids, att_mask, comment_len, punc_num)
            preds.append(torch.argmax(logits, 1).cpu().numpy())  

    # Concatenate all predictions across batches
    if preds:
        preds = np.concatenate(preds)

    print("prediction_num: ", len(preds))  # Log the number of predictions
    return preds


class Config(object):
    '''
    Configuration class to manage hyperparameters and paths.
    '''
    def __init__(self, mode):
        self.train_address = f'./dataset/all_training_data.json' 
        self.test_dataset = 'funcom'  
        self.test_mode = 'train'  
        self.pretrained_model = 'microsoft/codebert-base'  
        self.batch_size = 32  
        self.lr = 1e-4  
        self.bert_lr = 2e-5  
        self.class_num = 6  
        self.class_name = ['what', 'why', 'usage', 'done', 'property', 'others']  
        self.dropout = 0.2  
        self.epochs = 14  
        self.mode = mode  


if __name__ == '__main__':
    config = Config(mode='test')  
    set_seeds(seed)  

    # Load the model and move to GPU
    model = commentClassifier(config.pretrained_model, config.class_num, config.dropout)
    model.cuda()
    print("load the parameters of the pretrained classifier!")
    model.load_state_dict(torch.load("./trained_models/model_fold_9.pth"))  

    # Load test data
    test_loader = get_test_loaders(config.test_dataset, config.test_mode, config.pretrained_model, config.batch_size)
    predict_labels = prediction(model, test_loader)

    # Read input data and ensure predictions match the input length
    with open(f'./dataset/clean/{config.test_dataset}/{config.test_mode}/{config.test_dataset}.{config.test_mode}', 'r') as f:
        json_data = f.readlines()
    assert len(json_data) == len(predict_labels)

    label_num = [0, 0, 0, 0, 0, 0]  
    output_dir = f'./dataset/prediction/{config.test_dataset}/{config.test_mode}/'
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions and calculate label distribution
    with open(f'./dataset/prediction/{config.test_dataset}/{config.test_mode}/{config.test_dataset}.{config.test_mode}', 'w') as w:
        for json_line, label in tqdm(zip(json_data, predict_labels)):
            data_dict = json.loads(json_line.strip())
            output_dict = {'id': data_dict['id'], 'raw_code': data_dict['raw_code'].strip(),
                           'comment': data_dict['comment'].strip(), 'label': config.class_name[label]}
            if output_dict['label'] != 'others': 
                json.dump(output_dict, w)
                w.write('\n')

            label_num[label] += 1

    for name, num in zip(config.class_name, label_num):
        print(name, ':    ', num, 'ratio:', round(num / len(predict_labels), 2))