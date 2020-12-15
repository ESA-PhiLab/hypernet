from pathlib import Path
from tensorflow import keras
import tensorflow.keras.backend as K
from data_gen import load_image_paths, DG_38Cloud
import losses
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from plotly import express as px
import numpy as np


def datagen_to_gt_array(datagen):
    ret = []
    # Keras internals force usage of range(len()) here instead of enumerate
    for i in range(len(datagen)):
        ret.append(datagen[i][1])

    return np.concatenate(ret)


def make_roc(y_gt, y_pred, output_dir):
    fpr, tpr, thr = roc_curve(y_gt, y_pred)

    fig = px.area(
        x=fpr, y=tpr, hover_data={"treshold": thr},
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.write_html(str(output_dir/"roc.html"))


def make_activation_hist(y_pred, output_dir):
    fig = px.histogram(y_pred.ravel(), log_y=True)
    fig.layout.update(showlegend=False)
    fig.write_html(str(output_dir/"activation_hist.html"))


def make_precission_recall(y_gt, y_pred, output_dir):
    fpr, fre, thr = precision_recall_curve(y_gt, y_pred)

    fig = px.area(
        x=fre, y=fpr, hover_data={"treshold": np.insert(thr, 0, 1)},
        title='Precision-Recall Curve',
        labels=dict(x='Recall', y='Precision'),
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=1, y1=0
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.write_html(str(output_dir/"prec_recall.html"))


def make_validation_insights(model, datagen, output_dir):
    output_dir.mkdir(exist_ok=True)

    y_gt = datagen_to_gt_array(datagen).ravel()
    y_pred = np.round(model.predict_generator(datagen).ravel(), decimals=5)

    make_activation_hist(y_pred, output_dir)
    make_roc(y_gt, y_pred, output_dir)
    make_precission_recall(y_gt, y_pred, output_dir)


def main():
    train_size = 0.8
    dpath = Path("../datasets/clouds/38-Cloud/38-Cloud_training")
    batch_size=8

    _, val_files = load_image_paths(
        dpath,
        (train_size, 1-train_size)
    )
    valgen = DG_38Cloud(
        files=val_files,
        batch_size=batch_size,
    )

    mpath = Path("/media/ML/mlflow/beetle/artifacts/34/4848bf5b4c464af7b6be5abb0d70f842/"
                 + "artifacts/model/data/model.h5")
    model = keras.models.load_model(
        mpath, custom_objects={
            "jaccard_index_loss": losses.Jaccard_index_loss(),
            "jaccard_index_metric": losses.Jaccard_index_metric(),
            "dice_coeff_metric": losses.Dice_coef_metric(),
            "recall": losses.recall,
            "precision": losses.precision,
            "specificity": losses.specificity,
            "f1_score": losses.f1_score
        }
    )
    make_validation_insights(model, valgen, Path("./artifacts"))


if __name__ == "__main__":
    main()
