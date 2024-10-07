import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  # Add this import
import torch.optim as optim
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import torch.nn.functional as F

app = Flask(__name__)

# Define your UNet model (or use an existing implementation)
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (contracting path)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Decoder (expansive path)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)
        x2 = self.middle(x1)
        
        # Decoder
        x3 = self.decoder(x2)
        
        return x3



# Load the trained model weights
model = UNet()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()

# Define data transformations (ensure it's the same as during training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image file
        file = request.files['image']
        if file:
            # Read the image and perform inference
            image = Image.open(file).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output = model(image)
                predicted_mask = torch.sigmoid(output)
                # Convert the tensor to a NumPy array and scale it to 0-255 range
                predicted_mask = (predicted_mask * 255).byte().cpu().numpy()

                # Create a PIL Image object from the NumPy array
                predicted_mask_image = Image.fromarray(predicted_mask[0, 0])

                # Save the image
                predicted_mask_image.save('predicted_mask.png')
                # Process the predicted_mask as needed

            return render_template('result.html', predicted_mask=predicted_mask)
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
