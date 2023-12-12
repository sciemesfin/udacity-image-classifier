import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import json

def build_model():
    model = models.vgg16(pretrained=True)
    # model = models.vgg16(weights='imagenet')

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Define a new, untrained feed-forward network as a classifier
    classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    return model

def load_data(data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x])
        for x in ['train', 'valid', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=64, shuffle=True)
        for x in ['train', 'valid', 'test']
    }

    return dataloaders, image_datasets

def train(model, dataloaders, criterion, optimizer, epochs, device):
    steps = 0
    print_every = 5

    model.to(device)

    for e in range(epochs):
        running_loss = 0

        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss, accuracy = test_model(model, dataloaders['valid'], criterion, device)
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

def test_model(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()

            ps = torch.exp(outputs)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def save_checkpoint(model, image_datasets, filepath):
    checkpoint = {
        'arch': 'vgg16',
        'class_to_idx': image_datasets['train'].class_to_idx,
        'model_state_dict': model.state_dict(),
        'classifier': model.classifier
    }

    torch.save(checkpoint, filepath)

def main():
    parser = argparse.ArgumentParser(description='Image Classifier Training')
    parser.add_argument('data_dir', type=str, help='Path to the dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save the trained model')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataloaders, image_datasets = load_data(args.data_dir)
    model = build_model()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train(model, dataloaders, criterion, optimizer, args.epochs, device)
    save_checkpoint(model, image_datasets, args.save_dir)

if __name__ == "__main__":
    main()
