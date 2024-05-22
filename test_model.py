import torch
import os
from dataset import DataLoader
from yolo_model import YOLO

test_img_path = "bdd100k/bdd100k/images/100k/val/"
test_target_path = "bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json"
batch_size = 10
load_size = 1000
num_boxes = 2
lambda_coord = 5
lambda_no_obj = 0.5
load_model_file = "YOLO_model.pt"
iou_threshold_nms = 0.9
iou_threshold_map = 0.7
threshold = 0.5
use_nms = True  # This can be set to True or False if you prefer boolean for clarity
split_size = 14

category_list = ["other vehicle", "pedestrian", "traffic light", "traffic sign",
                "truck", "train", "other person", "bus", "car", "rider", "motorcycle",
                "bicycle", "trailer"]

cell_dim = int(448/split_size)
num_classes = len(category_list)

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

def test_model(test_img_path, test_target_path, category_list, split_size,
               batch_size, load_size, model, cell_dim, num_boxes, num_classes, device,
               iou_threshold_nms, threshold, use_nms):

    model.eval()

    print("DATA IS BEING LOADED FOR TESTING")
    print("")
    # Initialize the DataLoader for the test dataset
    data = DataLoader(test_img_path, test_target_path, category_list,
                      split_size, batch_size, load_size)
    data.LoadFiles()

    all_pred_boxes = []
    all_target_boxes = []

    train_idx = 0  # Tracks the sample index for each image in the test dataset

    while len(data.img_files) > 0:
        print("LOADING NEW TEST BATCHES")
        print("Remaining testing files:" + str(len(data.img_files)))
        print("")
        data.LoadData()

        for batch_idx, (img_data, target_data) in enumerate(data.data):
            img_data = img_data.to(device)
            target_data = target_data.to(device)

            with torch.no_grad():
                predictions = model(img_data)

            print('Extracting bounding boxes')
            print('Batch: {}/{} ({:.0f}%)'.format(batch_idx + 1, len(data.data),
                                                  (batch_idx + 1) / len(data.data) * 100.))
            print('')
            pred_boxes = extract_boxes(predictions, num_classes, num_boxes,
                                       cell_dim, threshold)
            target_boxes = extract_boxes(target_data, num_classes, 1, cell_dim,
                                         threshold)

            for sample_idx in range(len(pred_boxes)):
                if use_nms:
                    # Applies non max suppression to the bounding box predictions
                    nms_boxes = non_max_suppression(pred_boxes[sample_idx],
                                                    iou_threshold_nms)
                else:
                    # Use the same list without changing anything
                    nms_boxes = pred_boxes[sample_idx]

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for box in target_boxes[sample_idx]:
                    all_target_boxes.append([train_idx] + box)

                train_idx += 1

    pred = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    for prediction in all_pred_boxes:
        cls_idx = prediction[1]
        pred[cls_idx] += 1
    print(pred)
    pred = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}
    for prediction in all_target_boxes:
        cls_idx = prediction[1]
        pred[cls_idx] += 1
    print(pred)

def extract_boxes(yolo_tensor, num_classes, num_boxes, cell_dim, threshold):

    all_bounding_boxes = []  # Stores the final output

    for sample_idx in range(yolo_tensor.shape[0]):
        bounding_boxes = []  # Stores all bounding boxes of a single image
        for cell_height in range(yolo_tensor.shape[1]):
            for cell_width in range(yolo_tensor.shape[2]):

                # Used to extract the bounding box with the highest confidence
                best_box = 0
                max_confidence = 0.
                for box_idx in range(num_boxes):
                    if yolo_tensor[sample_idx, cell_height, cell_width, box_idx * 5] > max_confidence:
                        max_confidence = yolo_tensor[sample_idx, cell_height, cell_width, box_idx * 5]
                        best_box = box_idx
                conf = yolo_tensor[sample_idx, cell_height, cell_width, best_box * 5]
                if conf < threshold:
                    continue

                # Used to extract the class with the highest score
                best_class = 0
                max_confidence = 0.
                for class_idx in range(num_classes):
                    if yolo_tensor[sample_idx, cell_height, cell_width, num_boxes * 5 + class_idx] > max_confidence:
                        max_confidence = yolo_tensor[sample_idx, cell_height, cell_width, num_boxes * 5 + class_idx]
                        best_class = class_idx

                cords = MidtoCorner(yolo_tensor[sample_idx, cell_height, cell_width,
                                    best_box * 5 + 1:best_box * 5 + 5], cell_height, cell_width, cell_dim)
                x1 = cords[0]
                y1 = cords[1]
                x2 = cords[2]
                y2 = cords[3]

                bounding_boxes.append([best_class, conf, x1, y1, x2, y2])
        all_bounding_boxes.append(bounding_boxes)
    return all_bounding_boxes


def non_max_suppression(bounding_boxes, iou_threshold):

    assert type(bounding_boxes) == list

    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1], reverse=True)
    bounding_boxes_after_nms = []

    while bounding_boxes:
        chosen_box = bounding_boxes.pop(0)

        bounding_boxes = [
            box
            for box in bounding_boxes
            if box[0] != chosen_box[0]
               or IoU(
                chosen_box[2:],
                box[2:]
            )
               < iou_threshold
        ]

        bounding_boxes_after_nms.append(chosen_box)

    return bounding_boxes_after_nms


def main():
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda')

    # Initialize model
    model = YOLO(split_size, num_boxes, num_classes).to(device)

    # Load model weights
    print("Loading Model Weights ...")
    print("")
    model_weights = torch.load(load_model_file)
    model.load_state_dict(model_weights["state_dict"])

    # Start the validation process
    print("Testing Started ...")
    print("")
    test_model(test_img_path, test_target_path, category_list, split_size,
               batch_size, load_size, model, cell_dim, num_boxes, num_classes, device,
               iou_threshold_nms, iou_threshold_map, threshold, use_nms)


if __name__ == "__main__":
    main()
