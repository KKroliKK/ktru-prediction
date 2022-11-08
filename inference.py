import numpy as np
import torch
from training import create_fcnn
from torch.nn.functional import softmax
from training import generate_models_args


# TODO: add reading input_dim and hid_size from file


class Model():
    models_dict = None

    def __init__(self, ktru):
        self.ktru = ktru
        self.index_to_code = np.load(f'./index_to_code/{ktru}.npy', allow_pickle=True).item()

        self.model = create_fcnn(input_dim=600, num_classes=len(self.index_to_code))
        self.model.load_state_dict(torch.load(f'./models/_{ktru}.pt', map_location='cpu'))
        self.model.double()
        self.model.eval()

    def predict(self, embedding):
        with torch.no_grad():
            outp = self.model(torch.tensor(np.expand_dims(embedding, axis=0)))
        prob = softmax(outp, dim=1)
        pred = int(prob.argmax(-1))
        ktru = self.index_to_code[pred]

        if len(str(self.ktru)) != 12 and Model.models_dict != None:
            return Model.models_dict[ktru].predict(embedding)
        else:
            return ktru


def load_classificator(dataset):
    models_args = generate_models_args(dataset)
    models_dict = {}

    for args in models_args:
        models_dict[args['code']] = Model(args['code'])

    Model.models_dict=models_dict

    return models_dict['root']