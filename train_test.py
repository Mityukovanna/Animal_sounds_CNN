import torch
from dataset import AnimalSoundsDataset
from model import ConvNet
import torchaudio
from torch import nn
import os

# FORMULA TO CALCULATE THE OUTPUT SIZE
# (width size - filter size + 2*padding)/stride + 1
# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
num_epochs = 20
batch_size = 64
learning_rate = 0.00005

def create_data_loader(train_dataset, batch_size):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True  # shuffle the data directly in the data loader
    )
    return train_loader

def create_test_data_loader(test_dataset, batch_size):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def train_single_epoch(model, data_loader, criterion, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = criterion(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")

def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")

def test(model, data_loader, device, num_classes):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    # Initialize a list to store correct predictions per class
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    with torch.no_grad():  # No need to compute gradients during testing
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            
            # Forward pass
            prediction = model(input)
            
            # Get the index of the predicted class
            _, predicted_classes = torch.max(prediction, 1)
            
            # Count the correct predictions overall
            correct_predictions += (predicted_classes == target).sum().item()
            total_predictions += target.size(0)
            
            # Count correct predictions per class
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted_classes[i] == label:
                    class_correct[label] += 1

    # Overall accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Print class-specific accuracy
    for i in range(num_classes):
        if class_total[i] > 0:  # To avoid division by zero
            class_accuracy = (class_correct[i] / class_total[i]) * 100
            print(f"Class {i} Accuracy: {class_accuracy:.2f}%")
        else:
            print(f"Class {i} has no samples in the test set.")


def main():

    # instantiating dataset object and creating data loader
    ANNOTATIONS_FILE = r"C:\Users\admin\Desktop\python projects\neural_zoo\labels.csv"  
    AUDIO_DIR = r"C:\Users\admin\Desktop\python projects\neural_zoo\Wav_files" 
    SAMPLE_RATE = 16000

    # dataset transforms (building a mel_spectrogram)
    # parameters of teh spectrogram explained: https://stackoverflow.com/questions/62584184/understanding-the-shape-of-spectrograms-and-n-mels
    # one more link to understand mel spectrograms: https://importchris.medium.com/how-to-create-understand-mel-spectrograms-ff7634991056#:~:text=Each%20of%20these%20values%20in,second%20there%20are%2022%2C050%20samples.
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024, # frame size ()
        hop_length=512,
        n_mels = 64
    )

    # Debug: Check if paths exist
    if not os.path.exists(ANNOTATIONS_FILE):
        raise FileNotFoundError(f"Labels CSV not found at: {ANNOTATIONS_FILE}")

    if not os.path.exists(AUDIO_DIR):
        raise FileNotFoundError(f"Audio directory not found at: {AUDIO_DIR}")

    # Load dataset
    full_dataset = AnimalSoundsDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE)

    # Split dataset into training and test sets
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

    # Create data loaders
    train_dataloader = create_data_loader(train_dataset, batch_size)
    test_dataloader = create_test_data_loader(test_dataset, batch_size)
    # constructing model
    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss() # multiclass classification --> cross-entropy loss
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    train(model, train_dataloader, criterion, optimiser, device, num_epochs)

    # Test the model after training
    test(model, test_dataloader, device, num_classes=5)

    # save model
    torch.save(model.state_dict(), "CNN.pth")
    print("Trained CNN saved at CNN.pth")


if __name__ == "__main__":
    main()