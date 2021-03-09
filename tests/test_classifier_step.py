import traceback
import pytest
from serotiny_steps.train_classifier import train_classifier

def test_train_classifier():
    models = {
        2: [
            "BasicCNN_2D",
            "BasicNeuralNetwork",
            "ResNet18Network",
        ],
        3: [
            "BasicCNN_3D",
            "ResNet18_3D",
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
                    num_gpus=1,
                    num_workers=4,
                    num_epochs=1,
                    lr=1e-3,
                    optimizer="Adam",
                    scheduler="ReduceLROnPlateau",
                    test=True,
                    tune_bool=False,
                    x_label="dummy_x",
                    y_label="dummy_y",
                    classes=list(range(10)),
                    dimensionality=dimensionality,
                )
            except Exception as e:
                pytest.fail(f"`train_classifier` failed to run, for "
                            f"dimensionality {dimensionality} and model {model}.\n"
                            f"Printing traceback:\n{traceback.print_exc()}")


if __name__ == "__main__":
    test_train_classifier()
