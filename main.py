import torch
import train
import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ",device)

print("Training started...")
train.train(device)

print("Testing started...")
test.test(device)

