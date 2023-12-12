import torch
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
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
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    image = process_image(image_path).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        output = model(image)

    probabilities = torch.nn.functional.softmax(output.data, dim=1)
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices[0].tolist()]

    return top_probabilities[0].tolist(), top_classes

def main():
    parser = argparse.ArgumentParser(description="Predict flower class using a trained deep learning model")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to category names mapping file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available")

    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)

    if args.gpu and torch.cuda.is_available():
        model.to('cuda')

    top_probs, top_classes = predict(args.image_path, model, args.top_k)

    class_names = [cat_to_name[cls] for cls in top_classes]

    print(f"Top {args.top_k} predictions:")
    for prob, cls_name in zip(top_probs, class_names):
        print(f"Class: {cls_name}, Probability: {prob:.4f}")

if __name__ == "__main__":
    main()
