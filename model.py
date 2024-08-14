import torch
import torch.nn as nn
import torchvision
import tqdm
import os
import torchvision.models as models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
CHANNEL =  0
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.training_completed = False

    def is_trained(self):
        return self.training_completed

    def create_model(self):
        raise NotImplementedError("Subclasses must implement the create_model method.")

    def fit(self, X, y, device):

        pass

    def predict_proba(self, X):
        pass

    def predict(self, X):
        pass

    def ConfusionMatrix(self, predictions,labels):
        self.labels = labels.cpu().numpy()
        self.predictions = predictions.cpu().numpy()
        conf_matrix = confusion_matrix(self.labels, self.predictions)
        label = sorted(set(self.labels))
        return conf_matrix
    def predict_confidence(self, X):
        pass

class TabularClassifier(Model):
    def __init__(self, model_name):
        super(TabularClassifier, self).__init__(model_name)
        self.model = model_name
        self.labels = None
        self.predictions = None

    def fit(self, X_train, y_train):
        # Split the data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.bird, random_state=42)

        # Fit the provided model to the training data
        self.model.fit(X_train, y_train)

        # Evaluate the model on the test set
        # predictions = self.model.predict(X_test)
        # self.ConfusionMatrix(predictions, y_test)

    def predict_proba(self, X):
        # Return probability estimates for the test data
        return self.model.predict_proba(X)

    def predict(self, X):
        # Return predicted labels for the test data
        return self.model.predict(X)

    def ConfusionMatrix(self, predictions, labels):
        self.labels = labels
        self.predictions = predictions
        conf_matrix = confusion_matrix(self.labels, self.predictions)
        label = sorted(set(self.labels))
        return conf_matrix

    def predict_confidence(self, X):
        # Example: Return predicted labels along with confidence scores
        confidence_scores = self.model.predict_proba(X)
        predicted_labels = self.model.predict(X)
        return predicted_labels, confidence_scores

class Ensemble(Model):
    def __init__(self, model_name, model_list):
        super(Ensemble, self).__init__(model_name)
        self.model_list = model_list

    def combined_pred(self,dataset):
        self.predictions = []
        # self.labels = []
        for model in self.model_list:
            prediction = model.predict_proba(dataset)
            self.predictions.append(prediction)
            # self.labels.append(label)

        combined_predictions = np.stack([pred.cpu().numpy() for pred in self.predictions], axis=1)
        return combined_predictions

    def fit(self,device,dataset):
        #dataset  = dataset.train_loader
        models_to_fit = [model for model in self.model_list if not model.is_trained()]
        print(models_to_fit)
        if not models_to_fit:
            print("All models already fitted")
        else:
            self.dataset = dataset
            self.device = device
            for model in models_to_fit:
                model.fit(device, dataset)

class Voting(Ensemble):
    def __init__(self, model_list ):
        super(Voting, self).__init__("Voting",model_list)
        self.model_list = model_list

    def predict_proba(self, dataset): #dataset for testdataset
        combined_predictions = self.combined_pred(dataset=dataset.test_loader)
        # Use np.argmax to get the indices of the maximum values along axis 0 (samples)
        majority_votes = np.argmax(combined_predictions, axis=1)
        #print(majority_votes)
        return majority_votes

    def predict(self, dataset):
        self.labels = []
        for model in self.model_list:
            label = model.predict(dataset)
            self.labels.append(label)

        combined_labels = np.stack([pred.cpu().numpy() for pred in self.labels], axis=1)
        # Use np.argmax to get the indices of the maximum values along axis 0 (samples)
        majority_votes = np.argmax(combined_labels, axis=1)
        return majority_votes

class Stacking(Ensemble):
    def __init__(self, base_model_list,  meta_model  ):
        super(Stacking, self).__init__("Stacking",base_model_list)
        self.meta_model = meta_model
        self.combined_prediction = None


    def fit(self,device,dataset,is_stacked):
        super(Stacking, self).fit(device=device, dataset=dataset.train_loader)
        self.combined_prediction=super(Stacking,self).combined_pred(dataset.val_loader)
        labels = dataset.get_labels(train=False)

        if not is_stacked:
            self.meta_model.fit(self.combined_prediction,labels)

    def predict_proba(self, dataset): #dataset for testdataset
        self.combined_prediction = super(Stacking, self).combined_pred(dataset.test_loader)
        predict_proba=self.meta_model.predict_proba(self.combined_prediction)
        return  predict_proba

    def predict(self,dataset):
        self.combined_prediction = super(Stacking, self).combined_pred(dataset.test_loader)
        predict = self.meta_model.predict(self.combined_prediction)
        return predict

# List to store class names
called_classes = []
# Decorator to append the class name when the method is called
# def append_class_name(func):
#     def wrapper(self, *args, **kwargs):
#         class_name = self.__class__.__name__
#         called_classes.append(class_name)
#         return func(self, *args, **kwargs)
#     return wrapper


class ImageClassifier(Model):
    def __init__(self, model_name, num_classes, num_epochs, save_dir):
        super(ImageClassifier, self).__init__(model_name)
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.labels = None
        self.predictions = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train = False
        self.save_dir = save_dir
        self.log_file = self.save_dir+'/training.log'
        self.probabilities = None

    def create_model(self):
        raise NotImplementedError("Subclasses must implement the create_model method.")

    # @append_class_name
    def fit(self, dataset):
        early_stopping = EarlyStopping(patience=5, delta=0.001)
        self.dataset = dataset
        # self.device = device
        if self.model is None:
            raise ValueError("Model not created. Call create_model() before fitting.")
        if not os.path.exists(self.log_file):
            open(self.log_file, 'w').close()  # Create log file if not present
        with open(self.log_file, 'w') as log_file:
            print(f"==============Performing Training on Train Dataset with {self.model_name}==============")
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            self.model = self.model.to(self.device)  # Move the model to the device
            self.train = True
            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss = 0.0
                pbar = tqdm.tqdm(enumerate(self.dataset, 0), total=len(self.dataset))
                for i, data in pbar:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device),  labels.to(self.device)
                    #print(inputs.shape)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    pbar.set_description(
                        f"Classifier: {self.model_name}, Epoch {epoch + 1}, Loss: {running_loss / (i + 1):.4f}")
                epoch_loss = running_loss / len(self.dataset)
                log_file.write(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}\n")  # Write loss to log file
                torch.cuda.empty_cache()
                early_stopping(running_loss)
                torch.save(self.model.state_dict(), self.save_dir+f'/{self.model_name}_{epoch}_model_weights.pth')

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                if epoch > 0:
                    prev_model_weights_path = self.save_dir+f'{self.model_name}_{epoch - 1}_model_weights.pth'
                    if os.path.exists(prev_model_weights_path):
                        os.remove(prev_model_weights_path)  # Remove previous model file
        self.training_completed = True



    def predict_proba(self, data_loader):
        print("==============Predicting Probabilities on Test Dataset==============")
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # Initialize a list to store predictions
        labels_list = []  # Initialize a list to store ground truth labels
        max_probabilities = []  # Initialize a list to store maximum probabilities
        probabilities  = []

        with torch.no_grad():  # Disable gradient calculation for inference
            for data in tqdm.tqdm(data_loader, total=len(data_loader)):
                if len(data) == 2:
                    images, labels = data
                else:
                    images, labels = data[0], data[1]
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, 1)

                predictions.append(predicted)  # Append the predictions to the list
                labels_list.append(labels)  # Append the ground truth labels to the list
                max_probabilities.append(max_probs)
                # probs = probs.detach().cpu().numpy()
                probabilities.append(probs)


        # Convert lists of tensors to single tensors
        self.predictions = torch.cat(predictions, dim=0)
        self.labels = torch.cat(labels_list, dim=0)
        self.probabilities = torch.cat(max_probabilities, dim=0)
        self.probs = torch.cat(probabilities,dim=0)
        self.probs = self.probs.cpu().numpy()


        return self.probs

    def predict(self, dataset):
        if self.labels is None:
            raise ValueError("Model is Evaluated. Call predict_proba() before predict().")
        self.dataset = dataset
        return  self.labels

    def load_model(self,model_path):
        """
        Load a PyTorch model from a file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            torch.nn.Module: Loaded PyTorch model.
        """
        # Load the model state_dict
        model_state_dict = torch.load(model_path)
        # Load the state_dict into the model
        self.model.load_state_dict(model_state_dict)
        print("================== Model Loaded Successfully======================")

        # If the model was trained on GPU, move it back to GPU
        # if torch.cuda.is_available():
        #     self.model.cuda()

        return self.model

    def save_csv(self,output_csv):
        df = pd.DataFrame({'File Path': self.paths, 'True Label': self.labels, 'Predicted Label': self.predictions})

        # Save DataFrame to CSV file
        df.to_csv(output_csv, index=False)

        print(f"Predictions saved to {output_csv}")

    def get_probability(self):
        return self.probabilities

class AlexNet(ImageClassifier):
        def __init__(self,num_classes, num_epochs, channel, save_dir):
            self.channel = channel
            global CHANNEL
            CHANNEL = self.channel
            super(AlexNet, self).__init__("AlexNet", num_classes, num_epochs, save_dir)
            self.features = nn.Sequential(
                nn.Conv2d(self.channel, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.classifier(x)
            return x
        def create_model(self):
            self.features[0] = nn.Conv2d(self.channel, 64, kernel_size=11, stride=2, padding=2)
            self.model = nn.Sequential(self.features, self.avgpool, nn.Flatten(), self.classifier)


class ResNet(ImageClassifier):
    def __init__(self, num_classes, num_epochs, channel, save_dir):
        super(ResNet, self).__init__("ResNet", num_classes, num_epochs,save_dir)  # Initialize Model attributes
        self.channel = channel
        global CHANNEL
        CHANNEL = self.channel

    def create_model(self):
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(self.channel, 64, kernel_size=7, stride=2, padding=3)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, self.num_classes)

class InceptionV3(ImageClassifier):
    def __init__(self, num_classes, num_epochs, channel, save_dir):
        super(InceptionV3, self).__init__("InceptionV3", num_classes, num_epochs, save_dir)
        self.channel = channel
        global CHANNEL
        CHANNEL = self.channel

    def create_model(self):
        self.model = models.inception_v3(pretrained=False)
        # Modify first convolutional layer to match input size
        # self.model.Conv2d_1a_3x3.conv = nn.Conv2d(self.channel, 32, kernel_size=3, stride=2, bias=False                           )
        # Modify last fully connected layer to match the number of classes
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


class ConvAutoEncoder(nn.Module):
    def __init__(self,channel):
        super(ConvAutoEncoder, self).__init__()
        self.channel = channel

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.channel, 16, 3, stride=2, padding=1),  # 1x28x28 -> 16x14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 16x14x14 -> 32x7x7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)  # 32x7x7 -> 64x1x1
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),  # 64x1x1 -> 32x7x7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # 32x7x7 -> 16x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, self.channel, 3, stride=2, padding=1, output_padding=1),  # 16x14x14 -> 1x28x28
            nn.Sigmoid()  # Using Sigmoid for pixel values between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x