import torch
from sklearn import Accuracy, F1Score, Precision, Recall
from model.model import Model
from model.config import model_config

def eval_model(model : Model, dataloader):
    model = Model(model_config())
    model.load_state_dict(torch.load(...))

    model.eval()
    for x, y in dataloader:
        with torch.no_grad():
            ...



    return ...