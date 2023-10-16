import matplotlib.pyplot as plt
import torch
from prettytable import PrettyTable
from torchviz import make_dot


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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

def save_model_architecture(model,op,save_path, save_flag=True):
    if save_flag:
        make_dot(op, params=dict(list(model.named_parameters())), show_saved=True). \
            render(save_path, format="png")
