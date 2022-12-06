from torch import nn
import torch
from sklearn.metrics import accuracy_score

class Trainer():
    def __init__(self, 
            model, 
            train_dataloader, val_dataloader, test_dataloader, device):
        
        self.model = model.to(device)
        

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.loss_fn = nn.BCELoss() # nn.BCEWithLogitsLoss()

        self.device = device
    
    def train(self, num_epochs, optimizer, create_graph=False):
        train_accs = []
        val_accs = []
        losses = []
        loss_granular = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs.to(self.device))

                loss = self.loss_fn(outputs, labels.unsqueeze(1).to(self.device))
                loss.backward()
                optimizer.step()

                loss_granular.append(loss.to('cpu').detach().numpy())
                # print statistics
                running_loss += loss.item()
                if i % 1000 == 999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')
            print(f'[Epoch:{epoch + 1}] loss: {running_loss / len(self.train_dataloader):.3f}')
            if create_graph:
                # Predict after each epoch of training
                train_preds, train_Y = self.predict(self.train_dataloader)
                val_preds, val_Y = self.predict(self.val_dataloader)
                
                train_acc = accuracy_score(train_Y, train_preds)
                val_acc = accuracy_score(val_Y, val_preds)

                train_accs.append(train_acc)
                val_accs.append(val_acc)
                losses.append(running_loss)

        return train_accs, val_accs, loss_granular
    
    def fit(self, epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        train_loss = []
        train_accuracy = []

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            # Training Phase 
            for i, batch in enumerate(train_loader):
                images, labels = batch
                out = model(images.to(self.device))
    #             loss = model.training_step(batch)
                loss = nn.BCELoss()(out.squeeze(), labels.to(self.device))   #y_batch.unsqueeze(1)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item().to('cpu')

                if i % 1000 == 999:    # print every 2000 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')

            train_loss.append(epoch_loss/len(train_loader))

    #         print(train_loss)
        return  history, train_loss, train_accuracy

    def predict(self, dataloader):
        # returns predictions and corresponding labels
        Y = []
        preds = []
        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                # calculate outputs by running images through the network
                # convert input shape from [1, 11] to [11]
                # inputs = inputs.squeeze()
                outputs = self.model(inputs.to(self.device))
                # the class with the highest energy is what we choose as prediction
                predicted = (outputs.to('cpu').data.squeeze() > 0.5).float()
     
                preds.append(predicted)
                Y.append(labels.squeeze())
        return preds, Y


        





