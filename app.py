import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
import os

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

latent_dim = 100
n_classes = 10
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

@st.cache_resource
def load_model():
    model_path = "generator.pth"
    if not os.path.exists(model_path):
        return None
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

generator = load_model()

st.set_page_config(layout="wide")
st.title("Handwritten Digit Generation")

if generator is None:
    st.error("Model file (generator.pth) not found. Please add it to your GitHub repository.")
else:
    st.write("Select a digit. The app will use a GAN to generate five unique images.")
    col1, col2 = st.columns([1, 3])

    with col1:
        selected_digit = st.selectbox("Choose a digit (0-9):", list(range(10)), index=7)
        generate_button = st.button("Generate Images", use_container_width=True)

    if generate_button:
        with st.spinner(f"Generating images for digit {selected_digit}..."):
            num_images = 5
            z = torch.randn(num_images, latent_dim, device=device)
            labels = torch.LongTensor([selected_digit] * num_images).to(device)
            with torch.no_grad():
                generated_imgs = generator(z, labels)
            
            generated_imgs = 0.5 * generated_imgs + 0.5
            grid = make_grid(generated_imgs, nrow=5, normalize=True)
            img_grid = grid.permute(1, 2, 0).cpu().numpy()

            with col2:
                st.subheader(f"Generated Images for Digit: {selected_digit}")
                st.image(img_grid, width=500)
    else:
        with col2:
            st.info("Click 'Generate Images' to see the results.")