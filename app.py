import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os

# --- 1. App Configuration and Model Definition ---

st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✍️",
    layout="wide"
)

# Define the Generator class exactly as in your training script
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

# --- 2. Load the Trained Model ---

@st.cache_resource
def load_model():
    """Loads the pre-trained generator model."""
    if not os.path.exists("generator.pth"):
        st.error("Model file 'generator.pth' not found. Please run train.py first to generate it.")
        return None
        
    device = torch.device('cpu')
    model = Generator()
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

generator = load_model()

# --- 3. Streamlit User Interface ---

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained Conditional GAN model.")

if generator is None:
    st.stop()

st.write("---")

col1, col2 = st.columns([1, 2])

with col1:
    digit_to_generate = st.selectbox(
        label="Choose a digit to generate (0-9):",
        options=list(range(10))
    )
    
    num_samples = 5

    generate_button = st.button("Generate Images", type="primary")

st.write("---")


# --- 4. Image Generation and Display Logic ---

if generate_button:
    st.subheader(f"Generated images of digit {digit_to_generate}")

    with st.spinner(f"Generating {num_samples} images of '{digit_to_generate}'..."):
        device = torch.device('cpu')
        
        z = torch.randn(num_samples, latent_dim, device=device)
        labels = torch.LongTensor([digit_to_generate] * num_samples).to(device)
        
        with torch.no_grad():
            generated_imgs = generator(z, labels)
            
        generated_imgs = 0.5 * generated_imgs + 0.5 
        
        cols = st.columns(num_samples)
        for i in range(num_samples):
            with cols[i]:
                # Squeeze the channel dimension out (from 1,28,28 to 28,28)
                img_np = generated_imgs[i].squeeze().cpu().numpy()
                
                # THE FIX IS HERE: Replaced 'use_column_width' with 'use_container_width'
                st.image(img_np, caption=f"Sample {i+1}", use_container_width=True)