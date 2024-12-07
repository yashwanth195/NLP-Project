import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from dataloader import commentDataset
from tqdm import tqdm
from model import commentClassifier

seed = 12345

# Function to set the seed for random number generators in multiple libraries
def seed_everything(seed=seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_parameter_number(model):
    '''
    Function to calculate the total number of parameters in the model.
    '''
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def get_dialog_loaders(train_address, test_address, pretrained_model, batch_size=32, num_workers=0,
                       pin_memory=False):
    '''
    Function to create train and test data loaders.
    '''
    trainset = commentDataset(train_address, pretrained_model)
    print("train_set_num:", len(trainset))

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=trainset.collate_fn,
                              num_workers=num_workers, pin_memory=pin_memory)

    # Creating the test dataset and loading it into the DataLoader
    testset = commentDataset(test_address, pretrained_model)
    print("test_set_num:", len(testset))

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


def train_model(model, loss_function, dataloader, optimizer, epoch):
    '''
    Function to train the model for one epoch.
    '''
    losses, preds, labels = [], [], []

    model.train()

    seed_everything(seed + epoch)

    for data in tqdm(dataloader):
        # clear the grad
        optimizer.zero_grad()

        # Unpack the data and move tensors to the GPU
        input_ids, att_mask, label, comment_len, punc_num = [d.cuda() for d in data[:5]]
        pair_ids, comments = data[-2:]

        # Forward pass
        logits = model(input_ids, att_mask, comment_len, punc_num)
        loss = loss_function(logits, label)
        # Track predictions and labels for metrics calculation
        preds.append(torch.argmax(logits, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        # Backpropagate the gradients
        loss.backward()
        # optimize the parameters
        optimizer.step()
    # Concatenate the predictions and labels after all batches
    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    # Calculate the evaluation metrics
    avg_loss = round(np.sum(losses) / len(losses), 4)
    macro_f1 = round(f1_score(labels, preds, average='macro') * 100, 2)
    macro_precision = round(precision_score(labels, preds, average='macro') * 100, 2)
    macro_recall = round(recall_score(labels, preds, average='macro') * 100, 2)
    micro_f1 = round(f1_score(labels, preds, average='micro') * 100, 2)
    micro_precision = round(precision_score(labels, preds, average='micro') * 100, 2)
    micro_recall = round(recall_score(labels, preds, average='micro') * 100, 2)
    each_f1 = f1_score(labels, preds, average=None)
    each_precision = precision_score(labels, preds, average=None)
    each_recall = recall_score(labels, preds, average=None)

    return avg_loss, macro_f1, macro_precision, macro_recall, \
           micro_f1, micro_precision, micro_recall, each_f1, each_precision, each_recall


def evaluate_model(model, loss_function, dataloader):
    '''
    Function to evaluate the model on the test set.
    '''
    losses, preds, labels = [], [], []
    comment_list = []
    model.eval()

    seed_everything(seed)
    with torch.no_grad():
        for data in tqdm(dataloader):
            # Unpack the data and move tensors to the GPU
            input_ids, att_mask, label, comment_len, punc_num = [d.cuda() for d in data[:5]]
            pair_ids, comments = data[-2:]
            # Forward pass
            logits = model(input_ids, att_mask, comment_len, punc_num)
            loss = loss_function(logits, label)
            # Track predictions, labels, and comments
            preds.append(torch.argmax(logits, 1).cpu().numpy())
            labels.append(label.cpu().numpy())
            losses.append(loss.item())
            comment_list += comments

    # Concatenate the predictions and labels after all batches
    if preds:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)

    # Calculate the evaluation metrics
    avg_loss = round(np.sum(losses) / len(losses), 4)
    macro_f1 = round(f1_score(labels, preds, average='macro') * 100, 2)
    macro_precision = round(precision_score(labels, preds, average='macro') * 100, 2)
    macro_recall = round(recall_score(labels, preds, average='macro') * 100, 2)
    micro_f1 = round(f1_score(labels, preds, average='micro') * 100, 2)
    micro_precision = round(precision_score(labels, preds, average='micro') * 100, 2)
    micro_recall = round(recall_score(labels, preds, average='micro') * 100, 2)
    each_f1 = f1_score(labels, preds, average=None)
    each_precision = precision_score(labels, preds, average=None)
    each_recall = recall_score(labels, preds, average=None)

    # Store misclassified comments
    error_comment = []
    for p, l, comment in zip(preds, labels, comment_list):
        if p != l:
            error_comment.append((p, l, comment))

    return avg_loss, macro_f1, macro_precision, macro_recall, \
           micro_f1, micro_precision, micro_recall, each_f1, each_precision, each_recall, error_comment


class Config(object):
    '''
    Configuration class to set hyperparameters and file paths.
    '''
    def __init__(self, fold):
        self.train_address = f'./dataset/10fold/train_{fold}.json'
        self.test_address = f'./dataset/10fold/test_{fold}.json'
        self.pretrained_model = 'microsoft/codebert-base'
        self.batch_size = 32
        self.lr = 1e-4
        self.bert_lr = 2e-5
        self.class_num = 6  # 0:what, 1:why, 2:usage, 3:done, 4:property, 5:others
        self.class_name = ['what', 'why', 'usage', 'done', 'property', 'others']
        self.dropout = 0.2
        self.epochs = 14


if __name__ == '__main__':
    # Initialize variables to track performance across folds
    macro_p, macro_r, macro_f = 0, 0, 0
    micro_p, micro_r, micro_f = 0, 0, 0
    avg_each_p, avg_each_r, avg_each_f = [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
    # Iterate through 10-fold cross-validation
    for fold in range(10):
        config = Config(fold=fold)
        seed_everything(seed)

        model = commentClassifier(config.pretrained_model, config.class_num, config.dropout)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)
        # model.cuda()

        loss_function = nn.CrossEntropyLoss()

        unfreeze_layers = ['layer.11', 'pooler']
        for name, params in model.codeBert.named_parameters():
            params.requires_grad = False
            for unfrez in unfreeze_layers:
                if unfrez in name:
                    params.requires_grad = True

        print(get_parameter_number(model))
        # we adopt the Adam optimizer
        bert_params = list(map(id, model.codeBert.parameters()))
        bert_train_params = filter(lambda p: p.requires_grad, model.codeBert.parameters())
        downstream_params = filter(lambda p: id(p) not in bert_params, model.parameters())
        params = [
            {"params": downstream_params, "lr": config.lr},
            {"params": bert_train_params, "lr": config.bert_lr},
        ]

        optimizer = optim.Adam(params)
        train_loader, test_loader = get_dialog_loaders(config.train_address, config.test_address,
                                                       config.pretrained_model, config.batch_size)

        for e in range(config.epochs):
            train_loss, train_macro_f1, train_macro_pre, train_macro_rec, train_micro_f1, train_micro_pre, train_micro_rec, \
            train_each_f1, train_each_pre, train_each_rec = train_model(model, loss_function, train_loader, optimizer,
                                                                        e)

            print(
                'epoch: {}, train_loss: {}, train_macro_f1: {}, train_macro_pre: {}, train_macro_rec: {}, '
                'train_micro_f1: {}, train_micro_pre: {}, train_micro_rec: {}'.format(
                    e + 1,
                    train_loss, train_macro_f1, train_macro_pre, train_macro_rec, train_micro_f1, train_micro_pre,
                    train_micro_rec))

            if e == config.epochs - 1:
                test_loss, test_macro_f1, test_macro_pre, test_macro_rec, test_micro_f1, test_micro_pre, test_micro_rec, \
                test_each_f1, test_each_pre, test_each_rec, error_comment = evaluate_model(model, loss_function, test_loader)
                print(
                    'epoch: {}, test_loss: {}, test_macro_f1: {}, test_macro_pre: {}, test_macro_rec: {}, '
                    'test_micro_f1: {}, test_micro_pre: {}, test_micro_rec: {}'.format(
                        e + 1, test_loss, test_macro_f1, test_macro_pre, test_macro_rec, test_micro_f1, test_micro_pre,
                        test_micro_rec))

                for intent, p, r, f1 in zip(config.class_name, test_each_pre, test_each_rec, test_each_f1):
                    print(intent, ': ', 'P: ', p, 'R: ', r, 'F1: ', f1)

                macro_p += test_macro_pre
                macro_r += test_macro_rec
                macro_f += test_macro_f1
                micro_p += test_micro_pre
                micro_r += test_micro_rec
                micro_f += test_micro_f1
                for i in range(config.class_num):
                    avg_each_p[i] += test_each_pre[i]
                    avg_each_r[i] += test_each_rec[i]
                    avg_each_f[i] += test_each_f1[i]

            print("=========================================================================")

    print("training finish!!!")
    print(round(macro_p / 10, 2), round(macro_r / 10, 2), round(macro_f / 10, 2))
    print(round(micro_p / 10, 2), round(micro_r / 10, 2), round(micro_f / 10, 2))
    for p, r, f1 in zip(avg_each_p, avg_each_r, avg_each_f):
        print('P: ', round(p / 10, 2), 'R: ', round(r / 10, 2), 'F1: ', round(f1 / 10, 2))
