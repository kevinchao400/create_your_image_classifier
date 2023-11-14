import numpy as np
import argparse
import json
import PIL
from torchvision import datasets, transforms, models
from train import gpu_check
import modeling

def arg_parser():
    parser = argparse.ArgumentParser(description='predict.py')
    parser.add_argument('input', action='store', default='./flowers/test/1/image_06743.jpg')
    parser.add_argument('--dir', dest='data_dir', action='store', default='./flowers/')
    parser.add_argument('checkpoint', action='store', default='./checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', action='store', type=int, default=5)
    parser.add_argument('--category_names', dest='category_names', action='store', default='cat_to_name.json')
    parser.add_argument('--gpu', dest='gpu', action='store', default='gpu')

    args = parser.parse_args()
    return args

def load_checkpoint(path='checkpoint.pth'):
    # Load the saved file
    checkpoint = torch.load(path)
    # Load Defaults if none specified
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    # Load stuff from checkpoint
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):

    pil_image = PIL.Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    image = img_transforms(pil_image)
    return image

def predict(image_path, model, topk=5, device='gpu'):

    model.to(device)
    model.eval()
    img = process_image(image_path).unsqueeze_(0)
    image = img.float()

    with torch.no_grad():
        logps = model.forward(image.cuda())
    
    # convert to linear scale
    ps = torch.exp(logps).data

    # Find the top 5 results
    top5 = ps.topk(topk)

    # convert indices to classes
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}

    probs = top5[0][0].cpu().numpy()
    classes = [idx_to_class[each] for each in top5[1][0].cpu().numpy()]

    return probs, classes

def main():

    args = arg_parser()

    model = load_checkpoint(args.checkpoint)
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    device = gpu_check(args.gpu)
    image_tensor = process_image(args.input)

    probs, classes = predict(image_tensor, model, args.top_k, device)
    names = [cat_to_name[i] for i in classes]
    
    print('Predictions: ')
    for i in range(len(probs)):
        print("{} with a probability of {}".format(names[i], probs[i]))
            
    print("Prediction completed")

if __name__ == '__main__':
    main()
    





