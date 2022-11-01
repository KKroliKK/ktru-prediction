import numpy as np
from IPython.display import clear_output
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

SEED = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


def create_labels(ktru_codes, code_to_index=None):
    if code_to_index == None:
        unique = np.unique(ktru_codes)
        labels = [np.where(unique == code)[0][0] for code in ktru_codes]
        index_to_code = {index: code for index, code in enumerate(unique)}
        code_to_index = {code: index for index, code in enumerate(unique)}
        return np.array(labels), index_to_code, code_to_index
    else:
        labels = [code_to_index[code] for code in ktru_codes]
        return np.array(labels)


def train_model(
        model, 
        model_id,
        train_loader, 
        valid_loader,
        num_epochs=10,
        print_res=False):
    
    model.double()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())

    loaders = {"train": train_loader, "valid": valid_loader}
    accuracy = {"train": [], "valid": []}
    best_accuracy = 0


    for epoch in range(num_epochs):
        if print_res == False:
            clear_output()
        print(f'\nEpoch: {epoch + 1}')
        for k, dataloader in loaders.items():
            epoch_correct = 0
            epoch_all = 0

            for x_batch, y_batch in dataloader:

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                if k == "train":
                    model.train()
                    optimizer.zero_grad()
                    outp = model(x_batch)
                else:
                    model.eval()
                    with torch.no_grad():
                        outp = model(x_batch)

                preds = outp.argmax(-1)
                correct = (preds == y_batch).sum()
                all = len(outp)
                epoch_correct += correct.item()
                epoch_all += all

                if k == "train":
                    loss = criterion(outp, y_batch)
                    loss.backward()
                    optimizer.step()

            print(f"Loader: {k}. Accuracy: {round(epoch_correct / epoch_all, 4)}")

            accuracy[k].append(epoch_correct / epoch_all)

            if k == 'valid' and accuracy['valid'][-1] > best_accuracy:
                torch.save(model.state_dict(), f'./models/_{model_id}.pt')
                best_accuracy = accuracy['valid'][-1]

    print('\nBest accuracy:\n')
    print('train:', round(max(accuracy['train']), 4))
    print('test:', round(max(accuracy['valid']), 4))
    model.load_state_dict(torch.load(f'./models/_{model_id}.pt', map_location=device))
    return round(max(accuracy['train']), 4), round(max(accuracy['valid']), 4)


def create_fcnn(input_dim, num_classes, hid_size=100):
    model = nn.Sequential(
        nn.BatchNorm1d(input_dim),
        nn.Linear(input_dim, hid_size),
        nn.BatchNorm1d(hid_size),
        nn.ReLU(),
        nn.Linear(hid_size, num_classes)
    )
    return model


class Node():
    def __init__(self, 
            ktru, 
            ktru_column, 
            sub_ktru_column,
            dataset,
            embeddings
        ):
        self.ktru = ktru
        self.ktru_column = ktru_column
        self.sub_ktru_column = sub_ktru_column
        self.dataset = dataset
        self.embeddings = embeddings

    def get_dataset(self, batch_size=4, num_workers=2):
        train_idxs = (self.dataset[self.ktru_column] == self.ktru) & (self.dataset['train'])
        test_idxs = (self.dataset[self.ktru_column] == self.ktru) & (~self.dataset['train'])

        x_train = self.embeddings[train_idxs]
        x_test = self.embeddings[test_idxs]

        y = self.dataset[self.sub_ktru_column]
        y_train, self.index_to_code, code_to_index = create_labels(y[train_idxs])
        y_test = create_labels(y[test_idxs], code_to_index)

        np.save(f'./index_to_code/{self.ktru}.npy', self.index_to_code)

        train_loader = DataLoader(list(zip(x_train, y_train)), 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                shuffle=True, 
                                drop_last=True)

        valid_loader = DataLoader(list(zip(x_test, y_test)), 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                shuffle=False, 
                                drop_last=True)

        return train_loader, valid_loader
        
    def train(self, num_epochs=10, print_res=False, batch_size=4):
        train_loader, valid_loader = self.get_dataset(batch_size)
        self.model = create_fcnn(
            input_dim=self.embeddings.shape[1],
            num_classes=len(self.index_to_code),
            hid_size=100
            )
        return train_model(
            self.model, 
            self.ktru, 
            train_loader, 
            valid_loader, 
            num_epochs=num_epochs, 
            print_res=print_res)


hierarchy = ['root', 'ktru_2', 'ktru_5', 'ktru_8', 'ktru_12', 'ktru_code']


def get_tree_info(dataset):
    sum = 0

    for i in range(len(hierarchy) - 1):
        codes = dataset[hierarchy[i]].unique()
        sum += len(codes)
        print(len(codes))

    print(f'\nTotal sum: {sum}')


def generate_models_args(dataset):
    models_args = []
    counter = 0
    for i in range(len(hierarchy) - 1):
        codes = dataset[hierarchy[i]].unique()

        for code in codes:
            args = {
                'code': code,
                'ktru': hierarchy[i],
                'ktru_succ': hierarchy[i + 1],
                'index': counter
            }
            models_args.append(args)
            counter += 1
    
    return models_args


def train_tree(dataset, embeddings, index=0, num_epochs=10, batch_size=4):
    accuracy = []
    models_args = generate_models_args(dataset)

    for args in tqdm(models_args[index:]):
        clear_output()
        print('Model', args['code'], 'index', args['index'])

        node = Node(
            ktru=args['code'],
            ktru_column=args['ktru'],
            sub_ktru_column=args['ktru_succ'],
            dataset=dataset,
            embeddings=embeddings
        )
        acc = node.train(num_epochs=num_epochs, print_res=True, batch_size=batch_size)
        accuracy.append(acc)

    return accuracy