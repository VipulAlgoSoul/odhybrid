#create a model
# pass in input size
# pass in otput size
# create model create training
from visualize.visualize import count_parameters

import torch
from torchsummary import summary
from torchview import draw_graph
from torchvision.models import resnet18, GoogLeNet, densenet, vit_b_16
import graphviz
from prettytable import PrettyTable
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime



# when running on VSCode run the below command
# svg format on vscode does not give desired result
# graphviz.set_jupyter_format('png')
from .createmodel import myconvnet

class tinkerit():
    """ class to create initiale and model"""
    def __init__(self, dataloader_tr):
        self.dataloader = dataloader_tr
        train_features, train_labels = next(iter(dataloader_tr))
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        self.tri_dl = dataloader_tr
        self.val_dl = dataloader_tr

        self.inp_shape = train_features.size()
        self.out_shape = train_labels.size()
        self.out_vector = self.out_shape[-1]*self.out_shape[-2]*self.out_shape[-3]

        self.inp= torch.rand(self.inp_shape)
        self.out = torch.rand(self.out_shape)
        print("The input shape and output shape are", self.inp_shape, self.out_shape)

        self.inp_chnls = self.inp_shape[1]
        self.out_chnls = self.out_shape[1]

        if torch.cuda.is_available():
            print("Cuda is available Input in CUDA")
            inp_t = self.inp.cuda()
            out_t = self.out.cuda()
            # self.tri_dl = self.tri_dl.cuda()
            # self.val_dl = self.val_dl.cuda()
        else:
            print("cuda not found , going with cpu")

        self.loss = torch.nn.MSELoss()

    def train_one_epoch(self, epoch_index, tb_writer, optimizer):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.tri_dl):
            # Every data instance is an input + label pair
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                inputs = inputs/255
                labels = labels.cuda()

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            print(inputs.dtype)
            print(inputs)
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss(outputs, labels)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(self.tri_dl) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss
    def train_tinker(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Initializing in a separate cell so we can easily add more epochs to the same run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
        epoch_number = 0

        EPOCHS = 5

        best_vloss = 1_000_000.

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch_number + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch_number, writer, optimizer)

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.val_dl):
                    vinputs, vlabels = vdata
                    if torch.cuda.is_available():
                        vinputs = vinputs.cuda()
                        vinputs = vinputs/255
                        vlabels = vlabels.cuda()

                    voutputs = self.model(vinputs)
                    vloss = self.loss(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch
            # for both training and validation
            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number + 1)
            writer.flush()

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                torch.save(self.model.state_dict(), model_path)

            epoch_number += 1

    def do(self):
        print("___"*20)
        mymodel = myconvnet(input_channels = self.inp_chnls, out_vector = self.out_vector, out_shape = self.out_shape[1::])

        if torch.cuda.is_available():
            print("Cuda is available model in cuda")
            mymodel = mymodel.cuda()

        # print(mymodel)
        # # print(summary(mymodel,self.inp_shape))
        # op = mymodel(self.inp)
        # print(op.shape)
        count_parameters(mymodel)
        # make_dot(op, params=dict(list(mymodel.named_parameters())), show_saved=True).\
        #     render("E:\YOLOv10\YOLOv10//tinkering\mymodel", format="png")

        self.model = mymodel
        self.train_tinker()




