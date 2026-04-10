import flwr as fl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

#pytorch model
class Net(nn.Module):
    def __init__(self, input_size=7, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

#dataset cleaning
class CustomDataset(Dataset):
    def __init__(self, file_path):
        #Read 
        df = pd.read_csv(file_path)

        #Dropping irrelevant columns(TransactionID and CustomerID)
        df = df.drop(columns=["TransactionID", "CustomerID"])

        #Converting Timestamp to numeric (UNIX time)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df["Timestamp"] = df["Timestamp"].astype("int64") // 10**9

        #Encoding categorical columns
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].astype("category").cat.codes

        #Normalizing features to zero mean and unit variance
        X = df.drop(columns=["FraudLabel"]).values.astype(np.float32)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        self.X = torch.tensor((X - mean) / std, dtype=torch.float32)
        self.y = torch.tensor(df["FraudLabel"].values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


#Flower Client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

        #Compute class weights to handle severe class imbalance (~99% non-fraud)
        ds = trainloader.dataset
        y_train = ds.y if hasattr(ds, 'y') else ds.dataset.y[list(ds.indices)]
        class_counts = torch.bincount(y_train)
        total = y_train.shape[0]
        class_weights = total / (len(class_counts) * class_counts.float())
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for X, y in self.trainloader:
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total = 0, 0
        loss_total = 0.0
        with torch.no_grad():
            for X, y in self.valloader:
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss_total += loss.item() * X.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        accuracy = correct / total
        loss = loss_total / total
        return float(loss), len(self.valloader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    dataset = CustomDataset("datasets/client2_dataset.csv")

    #Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    trainset, valset = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = DataLoader(valset, batch_size=32)

    model = Net()

    client = FlowerClient(model, trainloader, valloader)

    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client()
    )
