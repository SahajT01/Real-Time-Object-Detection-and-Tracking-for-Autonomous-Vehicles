# Real-Time-Object-Detection-and-Tracking-for-Autonomous-Vehicles

bdd100k_labels_release, saved model and saved loss can be found at https://drive.google.com/drive/folders/14d1emTph7pBAwAqPOqoy5F4W_FXh5mXb?usp=sharing

# Task Distribution

Our project team comprised three members: Sahaj Totla, Sai Anuraghvav Savadam, and Sai Naveen Chanumolu, each contributing their unique expertise to different aspects of the project. Here's a detailed breakdown of the tasks and responsibilities handled by each member.

## 1. Sahaj Totla

### Data Processing
Sahaj Totla was responsible for the crucial initial step of data processing. This involved cleaning and preparing the dataset to ensure that it was suitable for training the YOLO model. Sahaj implemented various data augmentation techniques to enhance the robustness of the model. These preprocessing steps were essential for increasing the diversity of the training data and improving the model's ability to generalize to different real-world scenarios.

### Model Training
After the data was prepared and architecture was finalized by Sai Anuraghav Savadam, Sahaj took charge of training the YOLO model. This involved setting up the training pipeline, selecting appropriate hyperparameters, and using the Adam optimizer to ensure efficient convergence. Sahaj conducted two training sessions, monitored the model's performance through various epochs, and utilized early stopping to prevent overfitting.

### Frontend Development
Additionally, Sahaj developed the frontend using React. This interface allows users to upload videos for object detection. The frontend provided a seamless experience for users to interact with the system, visualizing both the original and processed videos side by side.

## 2. Sai Anuraghav Savadam

### Model Architecture
Sai Anuraghav Savadam focused on the design and implementation of the YOLO model's architecture. He added additional convolutional layers to improve feature extraction and increased the depth of existing layers to enhance the model's capacity.

### Inference
He also handled the inference phase, ensuring that the trained model could accurately detect and classify objects in real-time. He optimized the inference pipeline to minimize latency, allowing the model to process each frame efficiently.

### Backend Development
For the backend, he used Flask to create a robust server-side application. This backend handles video uploads, processes the videos using the trained model, and returns the results to the frontend. He ensured that the backend was bug-free, providing a smooth and efficient processing experience for users.

## 3. Sai Naveen Chanumolu

### Testing
Sai Naveen Chanumolu was in charge of the comprehensive testing of the system. He rigorously tested the model's performance on various datasets, ensuring that it could accurately detect objects under different conditions.

### Evaluation
He also conducted a detailed evaluation of the model. He used metrics such as Precision, Recall, F1 Score, and Mean Average Precision (mAP) to assess the model's accuracy and reliability.

### Inference for Both Models
In addition to his primary roles, Sai Naveen contributed to the inference phase alongside Sai Anuraghav. Together, they ensured that the model's predictions were accurate and optimized for real-time performance.
