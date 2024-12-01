import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.datasets import ImageFolder



import matplotlib.pyplot as plt
import numpy as np



transform_normal = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])



transforms_augmented_random = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

transforms_augmented_random_darkSaturation = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])


# Direktes dynamisches Laden bei Bedarf
combined_training_data = ConcatDataset([
    ImageFolder(root="Bilder", transform=transform_normal),
    ImageFolder(root="Bilder", transform=transforms_augmented_random),
    ImageFolder(root="Bilder", transform=transforms_augmented_random_darkSaturation)
])





# Prozentuale Aufteilung des kombinierten Datasets
total_size = len(combined_training_data)
train_size = int(0.7 * total_size)  # 70% für Training
val_size = total_size - train_size  # 30% für Validierung

train_data, val_data = random_split(combined_training_data, [train_size, val_size])

# DataLoader für Training und Validierung
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

"""
# Optional: Überprüfen, ob die Daten korrekt geladen wurden
images, labels = next(iter(train_loader))
plt.imshow(images[0].permute(1, 2, 0))  # Permutieren für die Anzeige
plt.title(f"Label: {labels[0].item()}")
plt.show()


class cnnNeuralnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_relu_pool = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLu(),
            nn.MaxPool2d(kernel_size=2, stride=2),
                        
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 4) 
    
        )    
        
    def forward(self, x):
        x = self.cnn_relu_pool(x)
        x = self.classification(x)
        return x
    
model = cnnNeuralnetwork()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.torch.optim.SGD(model.parameters(), lr=1e-2)


def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
         X, y = X.to("cpu"), y.to("cpu")

    # Compute prediction error
    pred = model(X) # model(X) calls model.forward(x)
    loss = loss_func(pred, y) # Calculate error between prediction and ground truth

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Logging
    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")
            pred = model(X)
            test_loss += loss_func(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"Test Error:\n\tAccuracy: {accuracy:>0.1f}%\n\tAvg loss: {test_loss:>8f}\n")
    return accuracy, test_loss


epochs = 10
history = {
    "epoch": [],
    "accuracy": [],
    "loss": [],
}
for e in range(epochs):
  print(f"Epoch {e+1}\n", 40*"-")
  train(train_loader, model, loss_func, optimizer)
  acc, loss = test(val_loader, model, loss_func)

  history["epoch"].append(e)
  history["accuracy"].append(acc)
  history["loss"].append(loss)
print("Done!")

        
"""        
