from model import SimpleMNISTNet
from train import train_model

if __name__ == "__main__":
    print("start training...")
    
    model = SimpleMNISTNet()
    
    train_model(model, epochs=5, batch_size=64, lr=0.001, save_path="mnist_model.pth")
