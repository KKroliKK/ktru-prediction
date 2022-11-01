import numpy as np


class Model():
    def __init__(self, ktru):
        self.ktru = ktru
        self.index_to_code = np.load(f'./index_to_code/{ktru}.npy', allow_pickle=True).item()

        hid_size = 100
        self.model = nn.Sequential(
            nn.BatchNorm1d(600),
            nn.Linear(600, hid_size),
            nn.BatchNorm1d(hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, len(self.index_to_code))
        )
        self.model.load_state_dict(torch.load(f'./models/_{ktru}.pt', map_location='cpu'))
        self.model.double()
        self.model.eval()

    def predict(self, embedding, models_dict):
        with torch.no_grad():
            outp = self.model(torch.tensor(np.expand_dims(embedding, axis=0)))
        pred = int(outp.argmax(-1))
        ktru = self.index_to_code[pred]

        if len(str(self.ktru)) != 12:
            return models_dict[ktru].predict(embedding, models_dict)
        else:
            return ktru

model = Model('root')