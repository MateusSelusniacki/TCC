import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import math
import Grad as df
import time
import os

def setBalance(sx,sy):
    n_1 = np.count_nonzero(sy == 1)
    n_2 = np.count_nonzero(sy == 2)
    index1 = 0
    index2 = 0
    while(np.count_nonzero(sy == 0) > int(0.70*len(sy))):
        len_sy = len(sy)
        for i in range(len_sy):
            if(np.count_nonzero(sy == 0) < int(0.70*len(sy))):
                break
            len_syi = len(sy[i])
            index2 = 0
            for j in range(len_syi):
                if(np.count_nonzero(sy == 0) < int(0.70*len(sy))):
                   break
                if(sy[i][j] == 1 and np.count_nonzero(sy == 1) < int(0.16*len(sy))):
                    temp = np.concatenate((sy[0:i],np.array([sy[i]])))
                    sy = np.concatenate((temp,sy[i::]))
                    
                    temp = np.concatenate((sx[0:i],np.array([sx[i]])))
                    sx = np.concatenate((temp,sx[i::]))
                if(sy[i][j] == 2 and np.count_nonzero(sy == 2) < int(0.16*len(sy))):
                    temp = np.concatenate((sy[0:i],np.array([sy[i]])))
                    sy = np.concatenate((temp,sy[i::]))
                    
                    temp = np.concatenate((sx[0:i],np.array([sx[i]])))
                    sx = np.concatenate((temp,sx[i::]))
                    
                index2 += 1
            index1 += 1

    return sx,sy

# check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")    

#df.csv2xlsx('raw_data/VALE3_raw.csv')
#df.improved_input()

dataall = df.allData()
## não estou consirando GPUs abaixo, apenas CPU, 
## para GPU o código precisa ser adaptado, os 
## dados precisam ser submetidos à GPU explicitamente

target_column = df.get_output()
#dataall = df.get_dataframe_input()

dataall['label'] = target_column
# print(dataall)

data = dataall[['label','Open','Close']]

#data = pd.DataFrame({'label':dataall['label'].tolist(),'Open':dataall['Open'].tolist(),'Close':dataall['Close'].tolist()})

def generate_time_lags(df, n_lags):
    df_n = df.copy()
    for n in range(n_lags,0,-1):
        df_n[f"Open{n}"] = df_n["Open"].shift(n)
        df_n[f"Close{n}"] = df_n["Close"].shift(n)  
    df_n = df_n.iloc[2*n_lags:] 
    return df_n

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

data.to_excel('VALE3_raw.xlsx')

# Paulo, ao rotular os dados, favor considerar rótulos no intervalo 0,1,2 ao invés de -1, 0, 1. Assim não precisaremos fazer nenhum pós-processamento na rede

data_new = generate_time_lags(data, 7) # o número 3 indica quantos instantes anteriores estão sendo considerados no treinamento,   

X, y = feature_label_split(data_new,"label")

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

X_dev, X_test, y_dev, y_test = train_test_split(X.values, y.values, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, shuffle=False)
print(X_dev.shape,y_dev.shape,np.count_nonzero(X_dev == 0),np.count_nonzero(X_dev == 1),np.count_nonzero(X_dev == 2),np.count_nonzero(y_dev == 0),np.count_nonzero(y_dev == 1),np.count_nonzero(y_dev == 2))
print(X_test.shape,y_test.shape,np.count_nonzero(X_test == 0),np.count_nonzero(X_test == 1),np.count_nonzero(X_test == 2),np.count_nonzero(y_test == 0),np.count_nonzero(y_test == 1),np.count_nonzero(y_test == 2))
print(X_train.shape,y_train.shape,np.count_nonzero(X_train == 0),np.count_nonzero(X_train == 1),np.count_nonzero(X_train == 2),np.count_nonzero(y_train == 0),np.count_nonzero(y_train == 1),np.count_nonzero(y_train == 2))
print(X_val.shape,y_val.shape,np.count_nonzero(X_val == 0),np.count_nonzero(X_val == 1),np.count_nonzero(X_val == 2),np.count_nonzero(y_val == 0),np.count_nonzero(y_val == 1),np.count_nonzero(y_val == 2))

X_dev,y_dev = setBalance(X_dev,y_dev)
X_test,y_test = setBalance(X_test,y_test)
X_train,y_train = setBalance(X_train,y_train)
X_val,y_val = setBalance(X_val,y_val)

print(X_dev.shape,y_dev.shape,np.count_nonzero(X_dev == 0),np.count_nonzero(X_dev == 1),np.count_nonzero(X_dev == 2),np.count_nonzero(y_dev == 0),np.count_nonzero(y_dev == 1),np.count_nonzero(y_dev == 2))
print(X_test.shape,y_test.shape,np.count_nonzero(X_test == 0),np.count_nonzero(X_test == 1),np.count_nonzero(X_test == 2),np.count_nonzero(y_test == 0),np.count_nonzero(y_test == 1),np.count_nonzero(y_test == 2))
print(X_train.shape,y_train.shape,np.count_nonzero(X_train == 0),np.count_nonzero(X_train == 1),np.count_nonzero(X_train == 2),np.count_nonzero(y_train == 0),np.count_nonzero(y_train == 1),np.count_nonzero(y_train == 2))
print(X_val.shape,y_val.shape,np.count_nonzero(X_val == 0),np.count_nonzero(X_val == 1),np.count_nonzero(X_val == 2),np.count_nonzero(y_val == 0),np.count_nonzero(y_val == 1),np.count_nonzero(y_val == 2))

'''
print(len(X_dev),len(y_dev))
(np.count_nonzero(y_dev == 2),np.count_nonzero(y_dev == 1),np.count_nonzero(y_dev == 0))
print(np.count_nonzero(y_test == 2),np.count_nonzero(y_test == 1),np.count_nonzero(y_test == 0))
print(np.count_nonzero(y_train == 2),np.count_nonzero(y_train == 1),np.count_nonzero(y_train == 0))
print(np.count_nonzero(y_val == 2),np.count_nonzero(y_val == 1),np.count_nonzero(y_val == 0))

#print(np.count_nonzero(y_test == 1),len(y_test))

print(len(y_test),np.bincount(np.reshape(y_test,y_test.size))[1])
print(len(y_train),np.bincount(np.reshape(y_train,y_train.size))[1])
'''
batch_size = 10

train_features = torch.Tensor(X_train)
train_targets = torch.Tensor(y_train)
val_features = torch.Tensor(X_val)
val_targets = torch.Tensor(y_val)
test_features = torch.Tensor(X_test)
test_targets = torch.Tensor(y_test)

#print(len(X_train),len(y_train))

train = TensorDataset(train_features, train_targets)
val = TensorDataset(val_features, val_targets)
test = TensorDataset(test_features, test_targets)

train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)


#print(train_targets.shape)
#print(train_features.shape)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, output_dim, dropout_prob, batch_size):
        super(GRUModel, self).__init__()

        self.layers = layers
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim, hidden_dim, layers, batch_first=True, dropout=dropout_prob)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def get_model(model, model_params):
    models = {
#         "rnn": RNNModel,
#         "lstm": LSTMModel,
        "gru": GRUModel,
    }
    return models.get(model.lower())(**model_params)

class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """
    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)
#         print("Yhat: ", yhat.shape)
#         print("Y: ",y.view(-1).long().shape)

        # Computes loss
        loss = self.loss_fn(yhat, y.view(-1).long())
        
#         loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        #model_path = './'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(yhat, y_val.view(-1).long()).item()
#                     val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

#             if (epoch <= 10) | (epoch % 50 == 0):
            print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:}\t Validation loss: {validation_loss:}"
            )

#         torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()

import torch.optim as optim

input_dim = 1
output_dim = 3
hidden_dim = 32
layer_dim = 2
batch_size = 10
dropout = 0.0
n_epochs = 50
learning_rate = 0.0001
weight_decay = 1e-6

model_params = {'input_dim': input_dim,
                'hidden_dim' : hidden_dim,
                'layers' : layer_dim,
                'output_dim' : output_dim,
                'dropout_prob' : dropout,
                'batch_size' : batch_size
                }

model = get_model('gru', model_params)

# loss_fn = nn.MSELoss(reduction="mean")
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print(val_loader,train_loader)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_dim)
opt.plot_losses()

predictions, values = opt.evaluate(
    test_loader_one,
    batch_size=1,
    n_features=input_dim
)

predictions = np.argmax(np.array(predictions).reshape([-1,output_dim]), axis=1)
values = np.array(values).reshape(-1)
np.set_printoptions(threshold=np.inf)

unique, counts = np.unique(values, return_counts=True)

unique, counts = np.unique(predictions, return_counts=True)

def format_predictions(predictions, values):
#     vals = np.concatenate(values, axis=0).ravel()
#     preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": values, "prediction": predictions}) #, index=df_test.head(len(vals)).index)
    return df_result

df_result = format_predictions(predictions, values)
df_result

from sklearn.metrics import accuracy_score, confusion_matrix
acc = accuracy_score(predictions, values)
print("Accuracy: ", acc)

confMatrix = confusion_matrix(predictions, values)

#print(predictions,values)
print(confMatrix)
print(predictions)  
print(values)

print(np.count_nonzero(values == 0),np.count_nonzero(values == 1),np.count_nonzero(values == 2))