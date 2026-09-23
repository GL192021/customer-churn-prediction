

def model_eval(Y_real, y_probas_class_1, t=0.5, after_thr_calib=False, print_bool=False, mdl_name=None):
    eps = 1e-12
    y_preds = (y_probas_class_1 >= t).astype(int)
    TP = ((Y_real == 1) & (y_preds == 1)).sum().item()
    FP = ((Y_real == 0) & (y_preds == 1)).sum().item()
    TN = ((Y_real == 0) & (y_preds == 0)).sum().item()
    FN = ((Y_real == 1) & (y_preds == 0)).sum().item()

    accuracy = (TP + TN) / (TP + FP + TN + FN + eps)
    precision__minority_Yes_1 = TP / (TP + FP + eps)
    recall__minority_Yes_1 = TP / (TP + FN + eps)
    recall_0 = TN / (TN + FP + eps)
    balanced_accuracy = 0.5 * (recall__minority_Yes_1 + recall_0)
    f1__minority_Yes_1 = 2 * TP / (2 * TP + FP + FN + eps)
    if print_bool:
        print("")
        print("=" * 100)
        if mdl_name:
            print("Model ", mdl_name)
        if after_thr_calib:
            print('Evaluation on test set AFTER threshold calibration')
            print('Calibration criterion: F1 maximization')
        print(f"Accuracy:           {accuracy:.3%}")
        print(f"Balanced Accuracy:  {balanced_accuracy:.3%}")
        print(f"Precision (class 1-minority):{precision__minority_Yes_1:.3%}")
        print(f"Recall (class 1-minority):   {recall__minority_Yes_1:.3%}")
        print(f"F1 score (class 1-minority): {f1__minority_Yes_1:.3%}")
        print("=" * 100)
        print("")
    return accuracy, balanced_accuracy, precision__minority_Yes_1, recall__minority_Yes_1, f1__minority_Yes_1


def threshold_calib(Y_real, y_prob_minority, print_bool=False):
    candidate_thresholds = sorted(set(y_prob_minority))
    best_t = None
    best_f1 = 0
    prec_for_best_t = 0
    recall_for_best_t = 0
    eps = 1e-12

    for t in candidate_thresholds:
        accuracy, balanced_accuracy, precision__minority_Yes_1, recall__minority_Yes_1, f1__minority_Yes_1 = model_eval(Y_real, y_prob_minority, t=t, print_bool=False)

        if f1__minority_Yes_1 > best_f1:
            best_f1 = f1__minority_Yes_1
            best_t = t
            prec_for_best_t = precision__minority_Yes_1
            recall_for_best_t = recall__minority_Yes_1

    if print_bool:
        print("Best threshold:", best_t)
        print("Best validation F1:", best_f1)
        print("Validation Precision for 'best t': ", prec_for_best_t)
        print("Validation Recall for 'best t': ", recall_for_best_t)

    return best_t

# def stronger_predictor(model, X, Y, ):
#     non_binary_cat_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
#                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
#     feature_groups = {}
#     ### group one-hot encoded columns
#     for f in non_binary_cat_cols:
#         cols = [c for c in X_train.columns if c.startswith(f + "_")]
#
#         if cols:
#             feature_groups[f] = cols
#     print(feature_groups)
#
#     baseline_pred = model.predict(X_test)
#     baseline_score = model_eval()
#     print("Baseline balanced accuracy:", baseline_score)
#
#     importance = {}
#     rng = np.random.default_rng(42)
#     for f, c in feature_groups.items():
#         # print(c)
#         drops = []
#
#         for i in range(50):
#             X_perm = X_test.copy()
#             perm = rng.permutation(len(X_perm))
#
#             # shuffle all one-hot encoded columns belonging to the feature together
#             X_perm.loc[:, c] = X_perm[c].to_numpy()[perm]
#
#             y_perm_pred = model.predict(X_perm)
#
#             perm_score = balanced_accuracy_score(
#                 Y_test,
#                 y_perm_pred
#             )
#
#             drops.append(baseline_score - perm_score)
#
#         importance[f] = {
#             "mean_drop": np.mean(drops),
#             "std": np.std(drops)
#         }
#
#     importance_df = pd.DataFrame(importance).T
#
#     importance_df = importance_df.sort_values(
#         by="mean_drop",
#         ascending=False
#     )
#
#     print(importance_df)
#
#     importance_df.plot.bar()
#     plt.xlabel("Drop in balanced accuracy")
#     plt.title("Grouped Permutation Feature Importance")
#     plt.show()