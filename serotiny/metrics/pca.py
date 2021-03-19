import pandas as pd
import logging
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def calculate_pca(save_dir, test_dataloader, enc_layers, save=True):
    """
    Return the features that explain a minimum
    variance of the entire dataset
    """

    x_features, _, _ = test_dataloader.get_all_items()
    x_features = x_features.view(-1, enc_layers[0]).cpu().numpy()

    i = enc_layers[0]
    # for i in range(x_features.shape[-1]):
    model = PCA(n_components=i).fit(x_features)
    x_pc = model.transform(x_features)
    n_pcs = model.components_.shape[0]

    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    try:
        initial_features_names = all_features.columns.values
    except:
        initial_features_names = [i for i in range(enc_layers[0])]

    most_important_names = [
        initial_features_names[most_important[i]] for i in range(n_pcs)
    ]

    dic = {
        "PC": [],
        "Most_important_feature": [],
        "PC_explained_variance": [],
        "All_components": [],
    }
    for i in range(n_pcs):
        dic["PC"].append("PC" + str(i))
        dic["Most_important_feature"].append(most_important_names[i])
        dic["PC_explained_variance"].append(model.explained_variance_[i])
        dic["All_components"].append(model.components_[i])

    df = pd.DataFrame(dic)

    if save is True:
        path_csv = save_dir / Path("pca.csv")
        df.to_csv(path_csv)
        LOGGER.info(f"Saved: {path_csv}")

    return df
