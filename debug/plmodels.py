import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from torchvision import models
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.training_completed = False

    def is_trained(self):
        return self.training_completed


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
        self.model.fit(X_train, y_train)

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


class ImageClassifier(pl.LightningModule, Model):
    def __init__(self, model_name, num_classes, max_epochs= 10, learning_rate=1e-3, pretrained = True):
        super(ImageClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.model = None
        self.max_epochs = max_epochs
        self.pretrained = pretrained

        # Load the model dynamically
        self.model = self.load_model(self.model_name, self.num_classes, self.pretrained)
        self.learning_rate = learning_rate

    def load_model(self, model_name, num_classes, pretrained):
        """
        Dynamically loads a model based on the model_name provided.

        Args:
            model_name (str): Name of the model to load (e.g., 'resnet18', 'vgg16').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.

        Returns:
            nn.Module: The loaded model.
        """
        if model_name == "ConvAutoEncoder":
            # Use your custom ConvAutoEncoder model
            model = ConvAutoEncoder(channel=3)  # Assuming num_classes corresponds to input channels
        else:
            model = getattr(models, model_name)(pretrained=pretrained)

            # For models with an 'fc' layer (like ResNet)
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

            # For models with a 'classifier' layer (like AlexNet, DenseNet, VGG)
        elif hasattr(model, 'classifier'):
            # Check if 'classifier' is a Sequential layer (as in AlexNet or VGG)
            if isinstance(model.classifier, nn.Sequential):
                num_ftrs = model.classifier[-1].in_features  # Last layer of the Sequential
                model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

            # If 'classifier' is a single Linear layer (as in DenseNet)
            elif isinstance(model.classifier, nn.Linear):
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, num_classes)

            else:
                raise ValueError(f"Unsupported classifier type: {type(model.classifier)}")

        else:
            raise ValueError("Unsupported model type: No 'fc' or 'classifier' layer found")

        return model



    def forward(self, x):
        return self.model(x)

    def get_logits_from_model_output(self,outputs):
        """
        Extracts logits from model output, handling different output types.
        """
        if isinstance(outputs, tuple):
            # Some models might return a tuple (e.g., ResNet)
            return outputs[0]  # Usually the first element is the logits

        elif hasattr(outputs, 'logits'):
            # For models like Inception V3
            return outputs.logits

        elif isinstance(outputs, torch.Tensor):
            # If the output is directly a tensor
            return outputs

        else:
            raise ValueError("Unsupported model output type: {}".format(type(outputs)))

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self(inputs)
        # Get logits from the model output
        logits = self.get_logits_from_model_output(outputs)

        loss = self.criterion(logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # Predict step to return predictions and labels
    def predict_step(self, batch, batch_idx):
        inputs, labels = batch  # Extract inputs and labels
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        outputs = self(inputs)
        return outputs, labels  # Return both predictions and true labels

    # New fit method
    def fit(self, train_dataloader, val_dataloader=None):
        """
        Trains the model using the provided dataloaders and PyTorch Lightning's Trainer.

        Args:
            train_dataloader: Dataloader for training data.
            val_dataloader: Dataloader for validation data (optional).
            max_epochs: Number of epochs to train (default: 10).
        """

        # Create the PyTorch Lightning Trainer
        trainer = pl.Trainer(max_epochs=self.max_epochs,callbacks=[EarlyStopping(monitor="train_loss", mode="min")])

        # Fit the model using the trainer and dataloaders
        if val_dataloader is not None:
            trainer.fit(self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self, train_dataloaders=train_dataloader)

    def predict(self, dataloader):
        """Makes predictions on the provided dataloader."""
        trainer = pl.Trainer()
        predictions = trainer.predict(self, dataloaders=dataloader)

        # Initialize a list to collect all predictions
        all_predictions = []

        for batch_predictions in predictions:
            # If the predictions are a tuple, get the first element
            if isinstance(batch_predictions, tuple):
                batch_predictions = batch_predictions[0]  # or another index depending on your model output

            # Ensure predictions are 2D
            if batch_predictions.dim() == 1:
                batch_predictions = batch_predictions.unsqueeze(1)

            all_predictions.append(batch_predictions)

        # Convert to a single tensor
        all_predictions = torch.cat(all_predictions, dim=0)

        # Get class labels by taking argmax
        class_labels = all_predictions.argmax(dim=1)

        return class_labels

    def predict_proba(self, dataloader):
        """Makes probability predictions on the provided dataloader."""
        # Initialize the PyTorch Lightning Trainer
        trainer = pl.Trainer()
        predictions = trainer.predict(self, dataloaders=dataloader)

        # Initialize a list to collect all predictions
        all_predictions = []

        for batch_predictions in predictions:
            # If the predictions are a tuple, get the first element
            if isinstance(batch_predictions, tuple):
                batch_predictions = batch_predictions[0]  # or another index depending on your model output
            # Ensure predictions are 2D
            if batch_predictions.dim() == 1:
                batch_predictions = batch_predictions.unsqueeze(1)

            all_predictions.append(batch_predictions)

        # Concatenate all batch predictions into a single tensor
        all_predictions = torch.cat(all_predictions, dim=0)

        # Apply softmax to get probabilities
        probabilities = torch.softmax(all_predictions, dim=1)

        # Convert the tensor to a NumPy array
        return probabilities.cpu().detach().numpy()


# class AutoEncoderLightning(pl.LightningModule):
#     def __init__(self, model_name,num_classes,max_epochs=10, model_kwargs=None, learning_rate=1e-3):
#         """
#         Args:
#             model_class (class): The class of the autoencoder model to instantiate (e.g., 'ConvAutoEncoder').
#             model_kwargs (dict): Optional keyword arguments for the model class instantiation.
#             learning_rate (float): Learning rate for the optimizer.
#         """
#         super(AutoEncoderLightning, self).__init__()
#         self.model_class = model_name  # The class of the autoencoder model
#         self.learning_rate = learning_rate
#         self.criterion = nn.MSELoss()  # Reconstruction loss
#         self.max_epochs = max_epochs
#         self.model_kwargs = model_kwargs if model_kwargs else {}
#
#         # Dynamically load the autoencoder model
#         self.model = self.load_model(self.model_class, self.model_kwargs)
#
#     def load_model(self, model_class, model_kwargs):
#         """
#         Dynamically loads and instantiates a model based on the provided class and keyword arguments.
#         """
#         return model_class(**model_kwargs)
#
#     def forward(self, x):
#         return self.model(x)
#
#     def fit(self, train_dataloader, val_dataloader=None):
#         """
#         Trains the model using the provided dataloaders and PyTorch Lightning's Trainer.
#
#         Args:
#             train_dataloader: Dataloader for training data.
#             val_dataloader: Dataloader for validation data (optional).
#             max_epochs: Number of epochs to train (default: 10).
#         """
#         # Create the PyTorch Lightning Trainer
#         trainer = pl.Trainer(max_epochs=self.max_epochs)
#
#         # Fit the model using the trainer and dataloaders
#         if val_dataloader is not None:
#             trainer.fit(self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
#         else:
#             trainer.fit(self, train_dataloaders=train_dataloader)
#
#     def training_step(self, batch, batch_idx):
#         inputs, _ = batch  # For autoencoders, the target is the same as the input
#         inputs = inputs.to(self.device)
#         outputs = self(inputs)
#         loss = self.criterion(outputs, inputs)  # Reconstruction loss
#         self.log('train_loss', loss)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         inputs, _ = batch  # For autoencoders, the target is the same as the input
#         inputs = inputs.to(self.device)
#         outputs = self(inputs)
#         loss = self.criterion(outputs, inputs)  # Reconstruction loss
#         self.log('val_loss', loss)
#         return loss
#
#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.learning_rate)

class ConvAutoEncoder(pl.LightningModule):
    def __init__(self,max_epochs=1):
        super(ConvAutoEncoder, self).__init__()
        self.max_epochs = max_epochs
        self.channel = 3
        self.individual_losses = []  # List to store training losses

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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # Forward pass
        x_hat = self(x)

        # Calculate loss for each input in the batch (no reduction)
        losses = nn.functional.mse_loss(x_hat, x, reduction='none')

        # Sum the losses across dimensions (if multi-dimensional input/output)
        individual_losses = losses.view(losses.size(0), -1).mean(dim=1)

        # Store or process individual losses as needed
        self.individual_losses.extend(individual_losses.detach().cpu().numpy())

        # Log average loss for the batch
        average_loss = individual_losses.mean()
        self.log('train_loss', average_loss)

        return average_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def return_training_loss(self):
        """Return the training losses as a NumPy array."""
        return np.array(self.individual_losses)

    def fit(self, train_dataloader, val_dataloader=None):
        """
        Trains the model using the provided dataloaders and PyTorch Lightning's Trainer.

        Args:
            train_dataloader: Dataloader for training data.
            val_dataloader: Dataloader for validation data (optional).
            max_epochs: Number of epochs to train (default: 10).
        """
        self.individual_losses = []
        # Create the PyTorch Lightning Trainer
        trainer = pl.Trainer(max_epochs=self.max_epochs,callbacks=[EarlyStopping(monitor="train_loss", mode="min")])

        # Fit the model using the trainer and dataloaders
        if val_dataloader is not None:
            trainer.fit(self, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        else:
            trainer.fit(self, train_dataloaders=train_dataloader)
        self.max_epochs = 1