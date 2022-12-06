from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from model import TwoLayerModel, TestModel
from dataset import VolleyballDataset
from trainer import Trainer
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def predict_milestone(train_X):
    ##############################################################################################
    # MILESTONE TRAINING CONFIGURATIONS
    BATCH_SIZE = 8
    EPOCHS = 3
    LR = 0.1
    GRAPH=False
    PRINT_PRETRAINING_ACC=False
    USE_GPU=True
    MODEL_PATH = 'milestone_ep{}.pt'.format(EPOCHS)
    
    input_dims = len(train_X.iloc[0])
    print("input_dims:", input_dims)

    device = enable_gpu() if USE_GPU else 'cpu'
    print("Training on:", device)

    # model = TwoLayerModel(input_dims)
    model = TestModel(input_dims)
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Model loaded from", MODEL_PATH)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR) # torch.optim.Adam(model.parameters(), lr=LR)
    ##############################################################################################
    

    # Instantiate Dataloaders
    trainset = VolleyballDataset("train")
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valset = VolleyballDataset("val")
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True)
    testset = VolleyballDataset("test")
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    trainer = Trainer(model, train_dataloader, val_dataloader, test_dataloader, device)

    # Evaluate!
    milestone_val_preds, val_Y = trainer.predict(val_dataloader)
    milestone_val_acc = accuracy_score(val_Y, milestone_val_preds)
    print("Milestone Val Accuracy:", milestone_val_acc)

def main():
    train = pd.read_csv("processed_data/train.csv")
    # Train x and y
    train_X = train.loc[:, train.columns != 'point_won_by']
    predict_milestone(train_X)

main()