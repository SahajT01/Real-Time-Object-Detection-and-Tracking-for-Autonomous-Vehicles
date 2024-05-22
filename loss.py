import torch

def IoU(target, prediction):
    i_x1 = max(target[0], prediction[0])
    i_y1 = max(target[1], prediction[1])
    i_x2 = min(target[2], prediction[2])
    i_y2 = min(target[3], prediction[3])
    intersection = max(0, (i_x2 - i_x1)) * max(0, (i_y2 - i_y1))
    union = ((target[2] - target[0]) * (target[3] - target[1])) + ((prediction[2] - prediction[0]) *
                                                                   (prediction[3] - prediction[1])) - intersection
    iou_value = intersection / union
    return iou_value


def MidtoCorner(mid_box, cell_h, cell_w, cell_dim):
    centre_x = mid_box[0] * cell_dim + cell_dim * cell_w
    centre_y = mid_box[1] * cell_dim + cell_dim * cell_h
    width = mid_box[2] * 448
    height = mid_box[3] * 448
    x1 = int(centre_x - width / 2)
    y1 = int(centre_y - height / 2)
    x2 = int(centre_x + width / 2)
    y2 = int(centre_y + height / 2)

    corner_box = [x1, y1, x2, y2]
    return corner_box

class Loss():
    def __init__(self, predictions, targets, split_size, num_boxes, num_classes,
                 lambda_coord, lambda_no_obj):

        self.predictions = predictions
        self.targets = targets
        self.split_size = split_size
        self.cell_dim = int(448 / split_size)  # Dimension of a single cell
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_no_obj = lambda_no_obj
        self.final_loss = torch.tensor(0.0, device=predictions.device)

    def loss(self):

        for sample in range(self.predictions.shape[0]):
            mid_loss = 0  # center loss
            dim_loss = 0  # width and height loss
            conf_loss = 0  # confidence score loss when obj
            conf_loss_no_obj = 0  # confidence score loss when no obj
            class_loss = 0  # class score loss
            for cell_h in range(self.split_size):
                for cell_w in range(self.split_size):
                    if self.targets[sample, cell_h, cell_w, 0] != 1:
                        conf_loss_no_obj += self.no_obj_loss(sample, cell_h, cell_w)
                    else:
                        mid_loss_local, dim_loss_local, conf_loss_local, class_loss_local = self.obj_loss(sample,
                                                                                                          cell_h,
                                                                                                          cell_w)
                        mid_loss += mid_loss_local
                        dim_loss += dim_loss_local
                        conf_loss += conf_loss_local
                        class_loss += class_loss_local

            # Calculate the final loss by summing the other losses and applying
            # the hyperparameters lambda_coord and lambda_noobj
            # print("mid_loss type:", type(mid_loss), "value:", mid_loss)
            # print("dim_loss type:", type(dim_loss), "value:", dim_loss)
            # print("conf_loss type:", type(conf_loss), "value:", conf_loss)
            # print("conf_loss_noobj type:", type(conf_loss_noobj), "value:", conf_loss_noobj)
            # print("class_loss type:", type(class_loss), "value:", class_loss)
            # print("final_loss before update:", self.final_loss)
            #print("Adding mid_loss...")
            #print("final_loss device:", self.final_loss.device, "dtype:", self.final_loss.dtype, "shape:",self.final_loss.shape)
            #print("mid_loss device:", mid_loss.device, "dtype:", mid_loss.dtype, "shape:", mid_loss.shape)
            # self.final_loss += self.lambda_coord * mid_loss
            # print("Current final_loss:", self.final_loss)
            #
            # print("Adding dim_loss...")
            # self.final_loss += self.lambda_coord * dim_loss
            # print("Current final_loss:", self.final_loss)
            #
            # print("Adding conf_loss...")
            # self.final_loss += self.lambda_noobj * conf_loss_noobj
            # print("Current final_loss:", self.final_loss)
            #
            # print("Adding other loss...")
            # self.final_loss += conf_loss + class_loss
            # print("Current final_loss:", self.final_loss)


            self.final_loss += (self.lambda_coord * mid_loss + self.lambda_coord * dim_loss
                                + self.lambda_no_obj * conf_loss_no_obj + conf_loss + class_loss)

    def no_obj_loss(self, sample, cell_h, cell_w):

        loss_value = 0.
        for box in range(self.num_boxes):
            loss_value += (0 - self.predictions[sample, cell_h, cell_w, box * 5]) ** 2
        return loss_value

    def obj_loss(self, sample, cell_h, cell_w):
        # Finds the box with the highest IoU with respect to the ground-truth and
        # stores its index in best_box
        if self.num_boxes != 1:
            best_box = self.find_best_box(sample, cell_h, cell_w)
        else:
            best_box = 0

        # Calculates the loss for the centre coordinates
        x_loss = torch.square(self.targets[sample, cell_h, cell_w, 1] -
                              self.predictions[sample, cell_h, cell_w, 1 + best_box * 5])
        y_loss = torch.square(self.targets[sample, cell_h, cell_w, 2] -
                              self.predictions[sample, cell_h, cell_w, 2 + best_box * 5])
        mid_loss_local = x_loss + y_loss

        # Calculates the loss for the width and height values
        w_loss = torch.square(torch.sqrt(self.targets[sample, cell_h, cell_w, 3]) -
                              torch.sqrt(self.predictions[sample, cell_h, cell_w, 3 + best_box * 5]))
        h_loss = torch.square(torch.sqrt(self.targets[sample, cell_h, cell_w, 4]) -
                              torch.sqrt(self.predictions[sample, cell_h, cell_w, 4 + best_box * 5]))
        dim_loss_local = w_loss + h_loss

        # Calculates the loss of the confidence score
        conf_loss_local = torch.square(1 - self.predictions[sample, cell_h, cell_w, best_box * 5])

        # Calculates the loss for the class scores
        class_loss_local = 0.
        for c in range(self.num_classes):
            class_loss_local += torch.square(self.targets[sample, cell_h, cell_w, 5 + c] -
                                             self.predictions[sample, cell_h, cell_w, 5 * self.num_boxes + c])

        return mid_loss_local, dim_loss_local, conf_loss_local, class_loss_local

    def find_best_box(self, sample, cell_h, cell_w):
        # Transform the box coordinates into the corner format
        t_box_coords = MidtoCorner(self.targets[sample, cell_h, cell_w, 1:5],
                                   cell_h, cell_w, self.cell_dim)

        best_box = 0
        max_iou = 0.
        for box in range(self.num_boxes):
            # Transform the box coordinates into the corner format
            p_box_coords = MidtoCorner(self.predictions[sample, cell_h, cell_w,
                                       1 + box * 5:5 + box * 5], cell_h, cell_w, self.cell_dim)

            box_score = IoU(t_box_coords, p_box_coords)
            if box_score > max_iou:
                max_iou = box_score
                best_box = box  # Store the box index with the highest IoU

        return best_box