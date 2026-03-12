import matplotlib.pyplot as plt


def plot_loss(history):

    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")

    plt.legend()
    plt.title("Training Loss")

    plt.show()


def plot_accuracy(history):

    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")

    plt.legend()
    plt.title("Training Accuracy")

    plt.show()
