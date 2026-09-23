import pandas as pd
import torch
import numpy as np

from model_evaluation import model_stat_metrics, threshold_calib
from preprocessing import prepare_for_training

from LogReg_pytorch import LogisticRegression_torch
from from_scratch import LogReg_Linear_from_scratch
from sklearn.linear_model import LogisticRegression

from torch.utils.data import TensorDataset, DataLoader

from imblearn.over_sampling import SMOTE


TORCH_EPOCHS = 2000
TORCH_BATCH_SIZE = 256
TORCH_LEARNING_RATE = 1e-3

SCRATCH_EPOCHS = 20_000
SCRATCH_LEARNING_RATE = 1e-2



def train_torch_model(X_train, Y_train, input_dim, class_weight: dict=None):
    """Train the PyTorch logistic-regression model."""
    model = LogisticRegression_torch(x_dim=input_dim)

    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=TORCH_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=TORCH_LEARNING_RATE)
    if class_weight is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        w_0, w_1 = class_weight[0], class_weight[1]

        weight = torch.tensor(w_1 / w_0, dtype=torch.float32)

        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)


    losses = []

    for epoch in range(TORCH_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for features, labels in train_loader:
            optimizer.zero_grad()

            logits = model(features).view(-1)
            loss = criterion(logits, labels.view(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * features.size(0)

        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)

        if (epoch + 1) % 100 == 0:
            print("Epoch ", epoch + 1, " Loss = ", epoch_loss)

    return model, losses







def main():
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    X_train, X_test, X_val, Y_train, Y_val, Y_test = prepare_for_training(df, Y_col="Churn")
    X_train_torch, X_test_torch, X_val_torch, Y_train_torch, Y_val_torch, Y_test_torch = prepare_for_training(df, Y_col="Churn", pytorch_bool=True)

    statistics = {}


    ## bacic model
    basic_model = LogisticRegression()
    basic_model.fit(X_train, Y_train)
    y_probas_val_class_1_basic = basic_model.predict_proba(X_val)[:, 1]
    y_probas_test_class_1_basic = basic_model.predict_proba(X_test)[:, 1]

    accuracy_basic, balanced_accuracy_basic, precision__minority_basic, recall__minority_basic, f1__minority_basic = model_stat_metrics(Y_test, y_probas_test_class_1_basic, mdl_name='BASIC', print_bool=True)
    best_t_basic = threshold_calib(Y_val, y_probas_val_class_1_basic)

    accuracy_basic_t, balanced_accuracy_basic_t, precision__minority_basic_t, recall__minority_basic_t, f1__minority_basic_t = model_stat_metrics(Y_test, y_probas_test_class_1_basic, t=best_t_basic, mdl_name='BASIC', print_bool=True)

    statistics["basic"] = {"acc": accuracy_basic, "bal_acc": balanced_accuracy_basic, "prec_minor": precision__minority_basic, "recall_minor": recall__minority_basic, "f_1_minor": f1__minority_basic}
    statistics["basic_t"] = {"acc": accuracy_basic_t, "bal_acc": balanced_accuracy_basic_t, "prec_minor": precision__minority_basic_t, "recall_minor": recall__minority_basic_t, "f_1_minor": f1__minority_basic_t}



    ## torch model
    torch_model, torch_losses = train_torch_model(X_train_torch, Y_train_torch, input_dim=X_train_torch.shape[1],)

    with torch.no_grad():
        y_logits_val_class_1_torch = torch_model(X_val_torch).squeeze()
        y_probas_val_class_1_torch = torch.sigmoid(y_logits_val_class_1_torch)
        y_probas_val_class_1_torch = np.array(y_probas_val_class_1_torch)

        y_logits_test_class_1_torch = torch_model(X_test_torch).squeeze()
        y_probas_test_class_1_torch = torch.sigmoid(y_logits_test_class_1_torch)
        y_probas_test_class_1_torch = np.array(y_probas_test_class_1_torch)

    accuracy_torch, balanced_accuracy_torch, precision__minority_torch, recall__minority_torch, f1__minority_torch = model_stat_metrics(Y_test, y_probas_test_class_1_torch, mdl_name='TORCH', print_bool=True)
    best_t_torch = threshold_calib(Y_val, y_probas_val_class_1_torch)

    accuracy_torch_t, balanced_accuracy_torch_t, precision__minority_torch_t, recall__minority_torch_t, f1__minority_torch_t = model_stat_metrics(Y_test, y_probas_test_class_1_torch, t=best_t_torch, mdl_name='TORCH', print_bool=True)

    statistics["torch"] = {"acc": accuracy_torch, "bal_acc": balanced_accuracy_torch, "prec_minor": precision__minority_torch, "recall_minor": recall__minority_torch, "f_1_minor": f1__minority_torch}
    statistics["torch_t"] = {"acc": accuracy_torch_t, "bal_acc": balanced_accuracy_torch_t, "prec_minor": precision__minority_torch_t, "recall_minor": recall__minority_torch_t, "f_1_minor": f1__minority_torch_t}



    ## from scratch model
    scratch_model = LogReg_Linear_from_scratch()
    scratch_model.fit(X_train, Y_train, SCRATCH_EPOCHS, SCRATCH_LEARNING_RATE)

    y_logits_val_class_1_scratch = scratch_model.forward(X_val)
    y_probas_val_class_1_scratch = scratch_model.get_prob_class_1_from_logits_from_scratch(y_logits_val_class_1_scratch)

    y_logits_test_class_1_scratch = scratch_model.forward(X_test)
    y_probas_test_class_1_scratch = scratch_model.get_prob_class_1_from_logits_from_scratch(y_logits_test_class_1_scratch)


    accuracy_scratch, balanced_accuracy_scratch, precision__minority_scratch, recall__minority_scratch, f1__minority_scratch = model_stat_metrics(Y_test, y_probas_test_class_1_scratch, mdl_name='FROM SCRATCH', print_bool=True)
    best_t_scratch = threshold_calib(Y_val, y_probas_val_class_1_scratch)

    accuracy_scratch_t, balanced_accuracy_scratch_t, precision__minority_scratch_t, recall__minority_scratch_t, f1__minority_scratch_t = model_stat_metrics(Y_test, y_probas_test_class_1_scratch, t=best_t_scratch, mdl_name='FROM SCRATCH', print_bool=True)

    statistics["scratch"] = {"acc": accuracy_scratch, "bal_acc": balanced_accuracy_scratch, "prec_minor": precision__minority_scratch, "recall_minor": recall__minority_scratch, "f_1_minor": f1__minority_scratch}
    statistics["scratch_t"] = {"acc": accuracy_scratch_t, "bal_acc": balanced_accuracy_scratch_t, "prec_minor": precision__minority_scratch_t, "recall_minor": recall__minority_scratch_t, "f_1_minor": f1__minority_scratch_t}







    ### Class weighting

    empirical_class_weights = {0:1, 1:3}

    # basic
    # basic_model_weights = LogisticRegression(class_weight='balanced')
    basic_model_weights_2 = LogisticRegression(class_weight=empirical_class_weights)
    # basic_model_weights.fit(X_train, Y_train)
    basic_model_weights_2.fit(X_train, Y_train)

    # y_probas_test_w_class1 = basic_model_weights.predict_proba(X_test)[:, 1]
    y_probas_test_w_2_class1 = basic_model_weights_2.predict_proba(X_test)[:, 1]

    # model_stat_metrics(Y_test, y_probas_test_w_class1, mdl_name="BASIC-balanced_weights", print_bool=True)
    accuracy_basic_w, balanced_accuracy_basic_w, precision__minority_basic_w, recall__minority_basic_w, f1__minority_basic_w = model_stat_metrics(Y_test, y_probas_test_w_2_class1, mdl_name="BASIC-1/3_weights", print_bool=True)

    # y_probas_validation_w_class1 = basic_model_weights.predict_proba(X_val)[:, 1]
    y_probas_validation_w_2_class1 = basic_model_weights_2.predict_proba(X_val)[:, 1]

    # best_t_w = threshold_calib(Y_val, y_probas_validation_w_class1)
    best_t_w_2 = threshold_calib(Y_val, y_probas_validation_w_2_class1)

    # model_stat_metrics(Y_test, y_probas_test_w_class1, t=best_t_w, mdl_name="BASIC-balanced_weights", print_bool=True)
    accuracy_basic_w__t, balanced_accuracy_basic_w__t, precision__minority_basic_w__t, recall__minority_basic_w__t, f1__minority_basic_w__t = model_stat_metrics(Y_test, y_probas_test_w_2_class1, t=best_t_w_2, mdl_name="BASIC-1/3_weights", print_bool=True)

    statistics["basic_w"] = {"acc": accuracy_basic_w, "bal_acc": balanced_accuracy_basic_w, "prec_minor": precision__minority_basic_w, "recall_minor": recall__minority_basic_w, "f_1_minor": f1__minority_basic_w}
    statistics["basic_w__t"] = {"acc": accuracy_basic_w__t, "bal_acc": balanced_accuracy_basic_w__t, "prec_minor": precision__minority_basic_w__t, "recall_minor": recall__minority_basic_w__t, "f_1_minor": f1__minority_basic_w__t}



    # Pytorch
    torch_model_w, torch_losses_w = train_torch_model(X_train_torch, Y_train_torch, input_dim=X_train_torch.shape[1], class_weight=empirical_class_weights)

    with torch.no_grad():
        y_logits_val_class_1_torch_w = torch_model_w(X_val_torch).squeeze()
        y_probas_val_class_1_torch_w = torch.sigmoid(y_logits_val_class_1_torch_w)
        y_probas_val_class_1_torch_w = np.array(y_probas_val_class_1_torch_w)

        y_logits_test_class_1_torch_w = torch_model_w(X_test_torch).squeeze()
        y_probas_test_class_1_torch_w = torch.sigmoid(y_logits_test_class_1_torch_w)
        y_probas_test_class_1_torch_w = np.array(y_probas_test_class_1_torch_w)

    accuracy_torch_w, balanced_accuracy_torch_w, precision__minority_torch_w, recall__minority_torch_w, f1__minority_torch_w = model_stat_metrics(Y_test, y_probas_test_class_1_torch_w, mdl_name='TORCH-1/3_weights', print_bool=True)
    best_t_torch_w = threshold_calib(Y_val, y_probas_val_class_1_torch_w)

    accuracy_torch_w__t, balanced_accuracy_torch_w__t, precision__minority_torch_w__t, recall__minority_torch_w__t, f1__minority_torch_w__t = model_stat_metrics(Y_test, y_probas_test_class_1_torch_w, t=best_t_torch_w, mdl_name='TORCH-1/3_weights', print_bool=True)

    statistics["torch_w"] = {"acc": accuracy_torch_w, "bal_acc": balanced_accuracy_torch_w, "prec_minor": precision__minority_torch_w, "recall_minor": recall__minority_torch_w, "f_1_minor": f1__minority_torch_w}
    statistics["torch_w__t"] = {"acc": accuracy_torch_w__t, "bal_acc": balanced_accuracy_torch_w__t, "prec_minor": precision__minority_torch_w__t, "recall_minor": recall__minority_torch_w__t, "f_1_minor": f1__minority_torch_w__t}



    # scratch
    scratch_model_weights = LogReg_Linear_from_scratch(binary_class_weight=empirical_class_weights)
    scratch_model_weights.fit(X_train, Y_train, SCRATCH_EPOCHS, SCRATCH_LEARNING_RATE)

    y_logits_test_class_1_scratch_w = scratch_model_weights.forward(X_test)
    y_probas_test_class_1_scratch_w = scratch_model_weights.get_prob_class_1_from_logits_from_scratch(y_logits_test_class_1_scratch_w)

    accuracy_scratch_w, balanced_accuracy_scratch_w, precision__minority_scratch_w, recall__minority_scratch_w, f1__minority_scratch_w = model_stat_metrics(Y_test, y_probas_test_class_1_scratch_w, mdl_name="SCRATCH-1/3-weights", print_bool=True)

    y_logits_val_class_1_scratch_w = scratch_model_weights.forward(X_val)
    y_probas_val_class_1_scratch_w = scratch_model_weights.get_prob_class_1_from_logits_from_scratch(y_logits_val_class_1_scratch_w)

    best_t_scratc_w = threshold_calib(Y_val, y_probas_val_class_1_scratch_w)

    accuracy_scratch_w__t, balanced_accuracy_scratch_w__t, precision__minority_scratch_w__t, recall__minority_scratch_w__t, f1__minority_scratch_w__t = model_stat_metrics(Y_test, y_probas_test_class_1_scratch_w, t=best_t_scratc_w, mdl_name="SCRATCH-1/3_weights", print_bool=True)

    statistics["scratch_w"] = {"acc": accuracy_scratch_w, "bal_acc": balanced_accuracy_scratch_w, "prec_minor": precision__minority_scratch_w, "recall_minor": recall__minority_scratch_w, "f_1_minor": f1__minority_scratch_w}
    statistics["scratch_w__t"] = {"acc": accuracy_scratch_w__t, "bal_acc": balanced_accuracy_scratch_w__t, "prec_minor": precision__minority_scratch_w__t, "recall_minor": recall__minority_scratch_w__t, "f_1_minor": f1__minority_scratch_w__t}






    ### SMOTE
    # basic
    smote = SMOTE(random_state=42)

    X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
    X_train_torch_smote, Y_train_torch_smote = torch.tensor(X_train_smote.values, dtype=torch.float32), torch.tensor(Y_train_smote.values, dtype=torch.float32)


    print("\nBefore SMOTE:")
    print(Y_train.value_counts())

    print("\nAfter SMOTE:")
    print(Y_train_smote.value_counts())

    basic_model_smote = LogisticRegression()
    basic_model_smote.fit(X_train_smote, Y_train_smote)

    y_probas_validation__basic_smote = basic_model_smote.predict_proba(X_val)[:, 1]
    y_probas_test__basic_smote = basic_model_smote.predict_proba(X_test)[:, 1]

    accuracy_basic_smote, balanced_accuracy_basic_smote, precision__minority_basic_smote, recall__minority_basic_smote, f1__minority_basic_smote = model_stat_metrics(Y_test, y_probas_test__basic_smote, mdl_name="BASIC-SMOTE", print_bool=True)

    best_t__basic_smote = threshold_calib(Y_val, y_probas_validation__basic_smote)

    accuracy_basic_smote__t, balanced_accuracy_basic_smote__t, precision__minority_basic_smote__t, recall__minority_basic_smote__t, f1__minority_basic_smote__t = model_stat_metrics(Y_test, y_probas_test__basic_smote,t=best_t__basic_smote, mdl_name="BASIC-SMOTE", print_bool=True)

    statistics["basic_smote"] = {"acc": accuracy_basic_smote, "bal_acc": balanced_accuracy_basic_smote, "prec_minor": precision__minority_basic_smote, "recall_minor": recall__minority_basic_smote, "f_1_minor": f1__minority_basic_smote}
    statistics["basic_smote__t"] = {"acc": accuracy_basic_smote__t, "bal_acc": balanced_accuracy_basic_smote__t, "prec_minor": precision__minority_basic_smote__t, "recall_minor": recall__minority_basic_smote__t, "f_1_minor": f1__minority_basic_smote__t}



    # Pytorch
    torch_model_smote, torch_losses_smote = train_torch_model(X_train_torch_smote, Y_train_torch_smote, input_dim=X_train_torch.shape[1],)

    with torch.no_grad():
        y_logits_val_class_1_torch_smote = torch_model_smote(X_val_torch).squeeze()
        y_probas_val_class_1_torch_smote = torch.sigmoid(y_logits_val_class_1_torch_smote)
        y_probas_val_class_1_torch_smote = np.array(y_probas_val_class_1_torch_smote)

        y_logits_test_class_1_torch_smote = torch_model_smote(X_test_torch).squeeze()
        y_probas_test_class_1_torch_smote = torch.sigmoid(y_logits_test_class_1_torch_smote)
        y_probas_test_class_1_torch_smote = np.array(y_probas_test_class_1_torch_smote)

    accuracy_torch_smote, balanced_accuracy_torch_smote, precision__minority_torch_smote, recall__minority_torch_smote, f1__minority_torch_smote = model_stat_metrics(Y_test, y_probas_test_class_1_torch_smote, mdl_name='TORCH-SMOTE', print_bool=True)
    best_t_torch_smote = threshold_calib(Y_val, y_probas_val_class_1_torch_smote)

    accuracy_torch_smote__t, balanced_accuracy_torch_smote__t, precision__minority_torch_smote__t, recall__minority_torch_smote__t, f1__minority_torch_smote__t = model_stat_metrics(Y_test, y_probas_test_class_1_torch_smote, t=best_t_torch_smote, mdl_name='TORCH-SMOTE', print_bool=True)

    statistics["torch_smote"] = {"acc": accuracy_torch_smote, "bal_acc": balanced_accuracy_torch_smote, "prec_minor": precision__minority_torch_smote, "recall_minor": recall__minority_torch_smote, "f_1_minor": f1__minority_torch_smote}
    statistics["torch_smote__t"] = {"acc": accuracy_torch_smote__t, "bal_acc": balanced_accuracy_torch_smote__t, "prec_minor": precision__minority_torch_smote__t, "recall_minor": recall__minority_torch_smote__t, "f_1_minor": f1__minority_torch_smote__t}


    # scratch
    scratch_model_smote = LogReg_Linear_from_scratch()
    scratch_model_smote.fit(X_train_smote, Y_train_smote, SCRATCH_EPOCHS, SCRATCH_LEARNING_RATE)

    y_logits_test_class_1__scratch_smote = scratch_model_smote.forward(X_test)
    y_probas_test_class_1__scratch_smote = scratch_model_smote.get_prob_class_1_from_logits_from_scratch(y_logits_test_class_1__scratch_smote)

    accuracy_scratch_smote, balanced_accuracy_scratch_smote, precision__minority_scratch_smote, recall__minority_scratch_smote, f1__minority_scratch_smote = model_stat_metrics(Y_test, y_probas_test_class_1__scratch_smote, mdl_name="SCRATCH-SMOTE", print_bool=True)

    y_logits_val_class_1_scratch_smote = scratch_model_smote.forward(X_val)
    y_probas_val_class_1_scratch_smote = scratch_model_smote.get_prob_class_1_from_logits_from_scratch(
        y_logits_val_class_1_scratch_smote)

    best_t_scratch_smote = threshold_calib(Y_val, y_probas_val_class_1_scratch_smote)

    accuracy_scratch_smote__t, balanced_accuracy_scratch_smote__t, precision__minority_scratch_smote__t, recall__minority_scratch_smote__t, f1__minority_scratch_smote__t = model_stat_metrics(Y_test, y_probas_test_class_1__scratch_smote, t=best_t_scratch_smote, mdl_name="SCRATCH-SMOTE", print_bool=True)

    statistics["scratch_smote"] = {"acc": accuracy_scratch_smote, "bal_acc": balanced_accuracy_scratch_smote, "prec_minor": precision__minority_scratch_smote, "recall_minor": recall__minority_scratch_smote, "f_1_minor": f1__minority_scratch_smote}
    statistics["scratch_smote__t"] = {"acc": accuracy_scratch_smote__t, "bal_acc": balanced_accuracy_scratch_smote__t, "prec_minor": precision__minority_scratch_smote__t, "recall_minor": recall__minority_scratch_smote__t, "f_1_minor": f1__minority_scratch_smote__t}


if __name__ == "__main__":
    main()


