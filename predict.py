import torch
from torchvision import models, transforms
from PIL import Image
import json

def load_model(filepath, device='cuda'):
    checkpoint = torch.load(filepath, map_location=device)
    
    if checkpoint['structure'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print("Architecture not recognized.")
        return None
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer'])
  
    model.epochs = checkpoint['epochs']
    
    model.to(device)
    
    return model

def process_image(image_path):
    img_pil = Image.open(image_path)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])
    
    image = img_transforms(img_pil)
    
    return image

def predict(image_path, model, topk=5):
    image = process_image(image_path)
    model.eval()
    
    # Convert image to PyTorch tensor
    image = torch.unsqueeze(image, 0)
    
    # Perform inference without gradient tracking
    with torch.no_grad():
        output = model(image)
    
    # Calculate probabilities and indices of the topk predictions
    probabilities, indices = torch.topk(torch.exp(output), topk)
  
    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices.numpy()[0]]
    
    # Convert PyTorch tensor to NumPy array for printing
    probabilities = probabilities.numpy()[0]
    
    return probabilities, classes

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Image Classifier Prediction")
    parser.add_argument("image_path", help="Path to the image for prediction")
    parser.add_argument("checkpoint_path", help="Path to the model checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Top K most likely classes")
    parser.add_argument("--category_names", default="cat_to_name.json", help="Path to the category names mapping file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for prediction if available")

    args = parser.parse_args()

    # Use GPU if available and specified by the user
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

    # Load model
    model = load_model(args.checkpoint_path, device=device)

    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Make prediction
    probs, classes = predict(args.image_path, model, topk=args.top_k)

    # Display results
    print("Probabilities:", probs)
    print("Classes:", classes)
    print("Class Names:", [cat_to_name[class_] for class_ in classes])
