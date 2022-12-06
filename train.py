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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#model_args.use_multiprocessing = False



##############################################################################################
# TRAINING CONFIGURATIONS
MODEL_TO_TRAIN = "milestone" # "naive_classifier"
data_path = 'export_our_test_value_FINAL.csv'
##############################################################################################

def main():
    # Load train and val from csvs
    train = pd.read_csv("processed_data/train.csv")
    val = pd.read_csv("processed_data/val.csv")
    test = pd.read_csv("processed_data/test.csv")
    # Train x and y
    train_X = train.loc[:, train.columns != 'point_won_by']
    train_Y = train['point_won_by']

    # val x and y
    val_X = val.loc[:, val.columns != 'point_won_by']
    val_Y = val['point_won_by']

    # test x and y
    test_X = test.loc[:, test.columns != 'point_won_by']
    test_Y = test['point_won_by']

    print("len(train):", len(train_X))
    print("len(val):", len(val_X))

    # Separate Training
    if MODEL_TO_TRAIN == "baseline":
        print("Training Baseline Model...")
        train_test_baseline(train_X, train_Y, val_X, val_Y)
    elif MODEL_TO_TRAIN == "naive_classifier":
        print("Training Naive Classifier...")
        test_naive(train_X, train_Y, val_X, val_Y)
    elif MODEL_TO_TRAIN == "milestone":
        print("Training Two Layer Network...")
        train_test_milestone(train_X, train_Y, val_X, val_Y)
    else:
        print("ERROR: INVALID MODEL INPUT")

def train_test_baseline(train_X, train_Y, val_X, val_Y):
    print("Started Training")
    logReg = LogisticRegression(random_state=0, max_iter=1000000)

    logReg = logReg.fit(val_X.iloc[:3], val_Y.iloc[:3])
    pre_val_preds = logReg.predict(val_X)
    pre_val_acc = accuracy_score(val_Y, pre_val_preds)
    print("Before Training: Val Accuracy:", pre_val_acc)

    logReg = logReg.fit(train_X, train_Y)

    train_preds = logReg.predict(train_X)
    train_acc = accuracy_score(train_Y, train_preds)
    print("Train Accuracy:", train_acc)
    val_preds = logReg.predict(val_X)
    val_acc = accuracy_score(val_Y, val_preds)
    print("Val Accuracy:", val_acc)
    # test_preds = logReg.predict(test_X)
    # test_acc = accuracy_score(test_Y, test_preds)
    # print("Test Accuracy:", test_acc)

def test_naive(train_X, train_Y, val_X, val_Y):
    # Naive Classifier
    naive_train_pred = [1 for i in range(len(train_X))]
    naive_train_acc = accuracy_score(train_Y, naive_train_pred)
    print("Naive Train Accuracy:", naive_train_acc)

    naive_val_pred = [1 for i in range(len(val_X))]
    naive_val_acc = accuracy_score(val_Y, naive_val_pred)
    print("Naive Val Accuracy:", naive_val_acc)

    # test_test_pred = [1 for i in range(len(test_X))]
    # naive_test_acc = accuracy_score(val_Y, test_test_pred)
    # print("Naive Test Accuracy:", naive_test_acc)


def train_test_milestone(train_X, train_Y, val_X, val_Y):
    ##############################################################################################
    # MILESTONE TRAINING CONFIGURATIONS
    BATCH_SIZE = 32
    EPOCHS = 1
    LR = 0.1
    GRAPH=False
    PRINT_PRETRAINING_ACC=False
    USE_GPU=True
    MODEL_PATH='milestone_ep{}.pt'.format(EPOCHS)
    
    input_dims = len(train_X.iloc[0])
    print("input_dims:", input_dims)

    device = enable_gpu() if USE_GPU else 'cpu'
    print("Training on:", device)

    # model = TwoLayerModel(input_dims)
    model = TestModel(input_dims)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR) # torch.optim.Adam(model.parameters(), lr=LR)
    ##############################################################################################
    

    # Instantiate Dataloaders
    trainset = VolleyballDataset("train")
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valset = VolleyballDataset("val")
    val_dataloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    testset = VolleyballDataset("test")
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    trainer = Trainer(model, train_dataloader, val_dataloader, test_dataloader, device)
    # Pretraining vals
    if PRINT_PRETRAINING_ACC:
        milestone_pre_val_preds, val_Y = trainer.predict(val_dataloader)
        milestone_pre_val_acc = accuracy_score(val_Y, milestone_pre_val_preds)
        print("Before Training Milestone Val Accuracy:", milestone_pre_val_acc)
    # TRAIN!
    train_accs, val_accs, losses = trainer.train(EPOCHS, optimizer, create_graph=GRAPH)
    # trainer.fit(EPOCHS, LR, model, train_dataloader, val_dataloader)

    if GRAPH:
        plot_train_curve(train_accs, val_accs, losses)

    print("SIUUU DONE TRAINING")
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model successfully saved at", MODEL_PATH)
    milestone_val_preds, val_Y = trainer.predict(val_dataloader)
    milestone_val_acc = accuracy_score(val_Y, milestone_val_preds)
    print("Milestone Val Accuracy:", milestone_val_acc)


def plot_train_curve(train_accs, val_accs, losses):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.plot(train_accs, label="train_accs")
    plt.plot(val_accs, label="val_accs")
    plt.plot(losses, label="loss")
    plt.legend()
    plt.show()

def enable_gpu():
    # this ensures that the current MacOS version is at least 12.3+
    #print(torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    #print(torch.backends.mps.is_built())
    return torch.device("cuda:0")
    

if __name__ == '__main__':
    main()
