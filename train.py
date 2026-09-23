import pandas as pd
import numpy as np

from preprocessing import *

from LogReg_pytorch import LogisticRegression_torch
from from_scratch import LogReg_Linear_from_scratch
from sklearn.linear_model import LogisticRegression

from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv", )

df_prepr = basic_preprocess_data(df)

X_train, X_test, X_val, Y_train, Y_val, Y_test = prepare_for_training(df, Y_col="Churn")
X_train_torch, X_test_torch, X_val_torch, Y_train_torch, Y_val_torch, Y_test_torch = prepare_for_training(df, Y_col="Churn", pytorch_bool=True)

x_dim = X_train_torch.shape[1]

## models
basic_model = LogisticRegression()
torch_model = LogisticRegression_torch(x_dim)
scratch_model = LogReg_Linear_from_scratch()



## training baci model
basic_model.fit(X_train, Y_train)
y_probas_class_1_basic = basic_model.predict_proba(X_test)[:, 1]



## training torch model
batch_size = 256
train_dataset = TensorDataset(X_train_torch, Y_train_torch)
test_dataset = TensorDataset(X_test_torch, Y_test_torch)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


## optimizer
# optimizer = torch.optim.SGD(model_torch.parameters())
optimizer = torch.optim.Adam(torch_model.parameters(), lr=0.001)
## BCE loss
criterion = torch.nn.BCEWithLogitsLoss()   #### we need to apply sigmoid in our model
# criterion = torch.nn.BCELoss()

epochs = 100
total_losses_per_epoch = []
for epoch in range(epochs):
    torch_model.train()
    epoch_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()

        y_pred_logits = torch_model(features).view(-1)
        # y_pred_proba = torch.sigmoid(y_pred_logits).squeeze()

        labels = labels.float().view(-1)
        loss = criterion(y_pred_logits, labels)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * features.size(0)
    epoch_loss /= len(train_loader.dataset)
    total_losses_per_epoch.append(epoch_loss)
    if (epoch+1) % 10 == 0:
        print("Epoch ", epoch+1, " Loss = ", loss.item())

torch_model.eval()

with torch.no_grad():
    y_logits_class_1_torch = torch_model(X_test_torch).squeeze()
    y_probas_class_1_torch = torch.sigmoid(y_logits_class_1_torch)





## training from scratch model
epochs = 500
lr = 0.01
scratch_model.fit(X_train, Y_train, epochs, lr)

y_logits_class_1_scratch = scratch_model.forward(X_test)
y_probas_class_1_scratch = scratch_model.get_prob_class_1_from_logits_from_scratch(y_logits_class_1_scratch)



