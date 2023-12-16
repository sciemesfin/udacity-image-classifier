import torch
from torch import nn, optim
from torchvision import models, transforms, datasets
from collections import OrderedDict
from PIL import Image
import json

def nn_setup(arch='vgg16', hidden_units=2048, output_units=102, lr=0.001, device='cuda'):
    # Choose the model architecture based on user input
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088  # VGG16 input size
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024  # Densenet121 input size
    else:
        raise ValueError("Invalid architecture. Please choose 'vgg16' or 'densenet121'.")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 256),
        nn.ReLU(),
        nn.Linear(256, output_units),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    model.to(device)

    return model, criterion, optimizer

def train_and_validate(model, trainloader, validloader, criterion, optimizer, epochs=2, print_every=10):
    steps = 0

    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for inputs, labels in validloader:
                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {e + 1}/{epochs}.. "
                      f"Loss: {running_loss / print_every:.3f}.. "
                      f"Validation Loss: {valid_loss / len(validloader):.3f}.. "
                      f"Accuracy: {accuracy / len(validloader):.3f}")
                running_loss = 0
                model.train()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Classifier Training")
    parser.add_argument("data_dir", help="Path to the directory containing train, validation, and test data")
    parser.add_argument("--save_dir", default=".", help="Directory to save the trained model checkpoint")
    parser.add_argument("--arch", choices=["vgg16", "densenet121"], default="vgg16", help="Model architecture")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--hidden_units", type=int, default=2048, help="Number of hidden units in the classifier")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training if available")

    args = parser.parse_args()

    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'
    test_dir = args.data_dir + '/test'

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
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
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }

    # Use GPU if available and specified by the user
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    model, criterion, optimizer = nn_setup(arch=args.arch, hidden_units=args.hidden_units, lr=args.learning_rate, device=device)

    train_and_validate(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, epochs=args.epochs)

    checkpoint = {
        'structure': args.arch,
        'model_state_dict': model.state_dict(),
        'classifier': model.classifier,
        'class_to_idx': image_datasets['train'].class_to_idx,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }

    save_path = args.save_dir + '/checkpoint.pth'
    torch.save(checkpoint, save_path)

    print(f"Model trained and checkpoint saved to {save_path}")
