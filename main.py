from src.data_loader import load_datasets
from src.model import build_convmixer
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import plot_loss, plot_accuracy


DATA_DIR = "data/colored_images"


def main():

    train_ds, val_ds = load_datasets(DATA_DIR)

    model = build_convmixer()

    history = train_model(model, train_ds, val_ds)

    plot_loss(history)
    plot_accuracy(history)

    evaluate_model(model, val_ds)


if __name__ == "__main__":
    main()
