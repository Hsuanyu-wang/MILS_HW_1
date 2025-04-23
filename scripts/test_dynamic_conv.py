import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from models.dynamic_conv import ResNet34_Dynamic
from models.resnet_34 import ResNet34
from custom_dataset import TxtImageDataset

def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])
    test_set = TxtImageDataset(args.test_txt, args.image_root, transform)
    # test_set = torchvision.datasets.ImageFolder(root=args.test_path, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = ResNet34_Dynamic(num_classes=args.num_classes) if args.dynamic else ResNet34(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.weights))
    model = model.to(device)
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--image_root', type=str, required=True)
    parser.add_argument('--test_txt', type=str, required=True)

    args = parser.parse_args()
    test(args)
