import traceback
import pytest
from serotiny_steps.train_classifier import train_classifier

def test_train_classifier():
    """
    Iterate through available classification networks and test we can run them
    """
    models = {
        2: [
            "BasicCNN",
            #"FullyConnected",
            #"ResNet18",
        ],
        3: [
            "BasicCNN",
            #"ResNet18",
        ]
    }
    for dimensionality in [2, 3]:
        for model in models[dimensionality]:
            try:
                train_classifier(
                    datasets_path="/tmp",
                    output_path="/tmp",
                    datamodule="DummyDatamodule",
                    model=model,
                    batch_size=16,
                    num_gpus=0,
                    precision=32,
                    num_workers=4,
                    num_epochs=1,
                    lr=1e-3,
                    optimizer="Adam",
                    lr_scheduler="ReduceLROnPlateau",
                    test=True,
                    tune_bool=False,
                    x_label="dummy_x",
                    y_label="dummy_y",
                    classes=list(range(10)),
                    dimensionality=dimensionality,
                    length=20, #kwarg for dummy datamodule
                    dims=[32]*dimensionality, #kwarg for dummy datamodule
                    channels=[0, 0, 0] #kwarg for dummy datamodule
                )
            except Exception as e:
                pytest.fail(f"`train_classifier` failed to run, for "
                            f"dimensionality {dimensionality} and model {model}.\n"
                            f"Printing traceback:\n{traceback.print_exc()}")


if __name__ == "__main__":
    test_train_classifier()
