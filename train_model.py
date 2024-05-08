from yolo_model import YOLO
from loss import Loss
from dataset import DataLoader
from utils import save_checkpoint
import torch.optim as optim
import torch
import time
import os

train_img_files_path = "bdd100k/bdd100k/images/100k/train/"
train_target_files_path = "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json"
learning_rate = 1e-5
batch_size = 10
num_epochs = 80
load_size = 1000
num_boxes = 2
lambda_coord = 5
lambda_no_obj = 0.5
load_model_file = "yolo.pt"

# Dataset parameters
category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                 "truck", "train", "other person", "bus", "car", "rider", "motorcycle",
                 "bicycle", "trailer"]


split_size = 14

# Other parameters
cell_dim = int(448 / split_size)
num_classes = len(category_list)


def TrainNetwork(num_epochs, split_size, batch_size, load_size, num_boxes, num_classes,
                 train_img_files_path, train_target_files_path, category_list, model,
                 device, optimizer, load_model_file, lambda_coord, lambda_no_obj):

    model.train()

    # Initialize the DataLoader for the train dataset
    data = DataLoader(train_img_files_path, train_target_files_path, category_list,
                      split_size, batch_size, load_size)

    track_loss = {}  # Used for tracking the loss
    torch.save(track_loss, "track_loss.pt")  # Initialize the log file

    for epoch in range(num_epochs):
        epoch_losses = []  # Stores the loss progress

        print("LOADING DATA FOR NEW EPOCH")
        print("")
        data.LoadFiles()

        while len(data.img_files) > 0:
            all_batch_losses = 0.

            print("LOADING NEW BATCHES")
            print("Remaining files:" + str(len(data.img_files)))
            print("")
            data.LoadData()  # Loads new batches

            for batch_idx, (img_data, target_data) in enumerate(data.data):
                img_data = img_data.to(device)
                target_data = target_data.to(device)

                optimizer.zero_grad()

                predictions = model(img_data)

                yolo_loss = Loss(predictions, target_data, split_size, num_boxes,
                                      num_classes, lambda_coord, lambda_no_obj)
                yolo_loss.loss()
                loss = yolo_loss.final_loss
                all_batch_losses += loss.item()

                loss.backward()
                optimizer.step()

                print('Train Epoch: {} of {} [Batch: {}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch + 1, num_epochs, batch_idx + 1, len(data.data),
                    (batch_idx + 1) / len(data.data) * 100., loss))
                print('')

            epoch_losses.append(all_batch_losses / len(data.data))
            print("Loss progress so far:", epoch_losses)
            print("")

        track_loss = torch.load('track_loss.pt')
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        track_loss['Epoch: ' + str(epoch + 1)] = mean_loss
        torch.save(track_loss, 'track_loss.pt')
        print(f"Mean loss for this epoch was {sum(epoch_losses) / len(epoch_losses)}")
        print("")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename=load_model_file)

        time.sleep(10)


def main():
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')

    # Initialize model
    model = YOLO(split_size, num_boxes, num_classes).to(device)

    # Define the learning method for updating the model weights
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Start the training process
    print("###################### STARTING TRAINING ######################")
    print("")
    TrainNetwork(num_epochs, split_size, batch_size, load_size, num_boxes,
                 num_classes, train_img_files_path, train_target_files_path,
                 category_list, model, device, optimizer, load_model_file,
                 lambda_coord, lambda_no_obj)


if __name__ == "__main__":
    main()