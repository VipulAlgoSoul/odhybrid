import matplotlib.pyplot as plt
import torch


def visualize_data(train_dataloader, id=0):
    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[id].squeeze()
    img=img.permute(1,2,0)
    # label = train_labels[id]
    label="ll"
    plt.imshow(img, cmap="gray")
    plt.show(block=True)
    # print(f"Label: {label}")


def visualize_asarray(traindata):

    for i, (images, labels) in enumerate(traindata):
        print(type(images), images.shape)

        img = images.squeeze()
        plt.imshow(img.permute(1, 2, 0))
        plt.show(block=True)
        # print(f"Label: {label}")
