import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, precision_recall_curve, \
    roc_curve
from IPython.display import display
from ml_ids.visualization import plot_confusion_matrix
import gc
from ml_ids.model_selection import split_x_y, train_val_test_split
from ml_ids.transform.sampling import upsample_minority_classes, downsample
from ml_ids.transform.preprocessing import create_pipeline
from collections import Counter

def predict(model, X, y):
    preds = model.predict(X, batch_size=8196)
    mse = np.mean(np.power(X - preds, 2), axis=1)

    return pd.DataFrame({'y_true': y, 'rec_error': mse})


def evaluate_pr_roc(pred):
    pr_auc = average_precision_score(pred.y_true, pred.rec_error)
    roc_auc = roc_auc_score(pred.y_true, pred.rec_error)
    return pr_auc, roc_auc


def plot_evaluation_curves(pred):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    precisions, recalls, thresholds = precision_recall_curve(pred.y_true, pred.rec_error)
    fpr, tpr, _ = roc_curve(pred.y_true, pred.rec_error)
    pr_auc, roc_auc = evaluate_pr_roc(pred)

    # plot precision / recall curve
    ax1.plot(recalls, precisions, label='auc={}'.format(pr_auc))
    ax1.set_title('Precision / Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend(loc='lower right')

    # plot ROC curve
    ax2.plot(fpr, tpr, label='auc={}'.format(roc_auc))
    ax2.set_title('ROC Curve')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel("False Positive Rate")
    ax2.legend(loc='lower right')


def plot_pr_threshold_curves(pred, pr_plot_lim=[0, 1]):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    precisions, recalls, thresholds = precision_recall_curve(pred.y_true, pred.rec_error)

    # plot precision / recall for different thresholds
    ax1.plot(thresholds, precisions[:-1], label="Precision")
    ax1.plot(thresholds, recalls[:-1], label="Recall")
    ax1.set_title('Precision / Recall of different thresholds')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Precision / Recall')
    ax1.legend(loc='lower right')

    # plot precision / recall for different thresholds
    ax2.plot(thresholds, precisions[:-1], label="Precision")
    ax2.plot(thresholds, recalls[:-1], label="Recall")
    ax2.set_title('Precision / Recall of different thresholds')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Precision / Recall')
    ax2.set_xlim(pr_plot_lim)
    ax2.legend(loc='lower right')


def best_precision_for_target_recall(pred, target_recall):
    precisions, recalls, thresholds = precision_recall_curve(pred.y_true, pred.rec_error)
    return thresholds[np.argmin(recalls >= target_recall)]


def get_misclassifications(y, pred_binary):
    misclassifications = y[y.label_is_attack != pred_binary]

    mc_df = pd.merge(pd.DataFrame({'misclassified': misclassifications.label.value_counts()}),
                     pd.DataFrame({'total': y.label.value_counts()}),
                     how='left', left_index=True, right_index=True)
    mc_df['percent_misclassified'] = mc_df.apply(lambda x: x[0] / x[1], axis=1)
    return mc_df.sort_values('percent_misclassified', ascending=False)


def print_performance(y, pred, threshold):
    pred_binary = (pred.rec_error >= threshold).astype('int')

    print('Classification Report:')
    print('======================')
    print(classification_report(pred.y_true, pred_binary))

    print('Confusion Matrix:')
    print('=================')
    plot_confusion_matrix(pred.y_true, pred_binary, np.array(['Benign', 'Attack']), size=(5, 5))
    plt.show()

    print('Misclassifications by attack category:')
    print('======================================')
    mc_df = get_misclassifications(y, pred_binary)
    display(mc_df)


def filter_benign(X, y):
    return X[y.label_is_attack == 0]

def transform_data(dataset,
                   attack_samples,
                   imputer_strategy,
                   scaler,
                   benign_samples=None,
                   random_state=None):

    cols_to_impute = dataset.columns[dataset.isna().any()].tolist()

    train_data, val_data, test_data = train_val_test_split(dataset,
                                                           val_size=0.1,
                                                           test_size=0.1,
                                                           stratify_col='label_cat',
                                                           random_state=random_state)

    if benign_samples:
        train_data = downsample(train_data, default_nr_samples=benign_samples, random_state=random_state)

    X_train_raw, y_train = split_x_y(train_data)
    X_val_raw, y_val = split_x_y(val_data)
    X_test_raw, y_test = split_x_y(test_data)

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train_raw.shape))
    print('Val:      {}'.format(X_val_raw.shape))
    print('Test:     {}'.format(X_test_raw.shape))

    print('\nTraining labels:')
    print('================')
    print(y_train.label.value_counts())
    print('\nValidation labels:')
    print('==================')
    print(y_val.label.value_counts())
    print('\nTest labels:')
    print('============')
    print(y_test.label.value_counts())

    del train_data, val_data, test_data
    gc.collect()

    pipeline, get_col_names = create_pipeline(X_train_raw,
                                              imputer_strategy=imputer_strategy,
                                              imputer_cols=cols_to_impute,
                                              scaler=scaler)

    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    column_names = get_col_names()

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train.shape))
    print('Val:      {}'.format(X_val.shape))
    print('Test:     {}'.format(X_test.shape))

    print('\nMissing values:')
    print('===============')
    print('Training: {}'.format(np.count_nonzero(np.isnan(X_train))))
    print('Val:      {}'.format(np.count_nonzero(np.isnan(X_val))))
    print('Test:     {}'.format(np.count_nonzero(np.isnan(X_test))))

    print('\nScaling:')
    print('========')
    print('Training: min={}, max={}'.format(np.min(X_train), np.max(X_train)))
    print('Val:      min={}, max={}'.format(np.min(X_val), np.max(X_val)))
    print('Test:     min={}, max={}'.format(np.min(X_test), np.max(X_test)))

    X_train, y_train = upsample_minority_classes(X_train,
                                                 y_train,
                                                 min_samples=attack_samples,
                                                 random_state=random_state)

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train.shape))

    print('\nTraining labels:')
    print('================')
    print(Counter(y_train))

    return X_train, y_train, X_val, y_val, X_test, y_test, column_names
