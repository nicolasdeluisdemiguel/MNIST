import sys
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
from model import SimpleMNISTNet

def predict_image(image_path, model_path="mnist_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model = SimpleMNISTNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    image = Image.open(image_path).convert("L")
    image = ImageOps.invert(image)

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item() * 100

    print(f"predicted digit: {predicted_class}")
    print(f"confidence: {confidence:.2f}%")

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    predict_image(img_path)
