from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Image to predict')
    parser.add_argument('checkpoint', type=str, help='Model checkpoint to use')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K predictions')
    parser.add_argument('--category_names', type=str, default="cat_to_name.json", help='JSON file containing label names')
    parser.add_argument('--gpu', default=False, help='Use GPU if available')
    return parser.parse_args()

def load_checkpoint(filepath, device):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.classifier = checkpoint['classifier']
    model.state_dict =  checkpoint['state_dict']
    model.class_to_idx =  checkpoint['class_to_idx']
    return model

def process_image(image):
    pil_image = Image.open(image).convert("RGB")
    adjustments = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    pil_image = adjustments(pil_image)
    return pil_image

def predict(image_path, model, device, topk=1):
    image = process_image(image_path)
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        top_prob = top_prob.exp()
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
    return top_prob.numpy()[0], mapped_classes

def main():
    args = parse_args()
    device = torch.device("cpu")
    if args.gpu:
        device = torch.device("cuda:0")
    print('Using', device)
    model = load_checkpoint(args.checkpoint, device)
    #model = model.to(device)
    image = args.input
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    probs, classes = predict(image, model, device, args.top_k)
    print(probs, [cat_to_name[name] for name in classes])


if __name__ == "__main__":
    main()