import streamlit as st
import torch
from flux import Flux2Generator

@st.cache_resource
def get_generator(model_name: str, device: str):
    return Flux2Generator(model_name, device)

def main():
    st.title("Flux Image Generation Demo")
    
    model_name = st.sidebar.text_input("Model Name", "black-forest-labs/FLUX.2-dev")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if st.sidebar.button("Load Model"):
        st.session_state.generator = get_generator(model_name, device)
        st.success(f"Loaded {model_name} on {device}")

    if "generator" not in st.session_state:
        st.info("Please load the model first.")
        return

    prompt = st.text_area("Prompt", "a photo of a forest with mist swirling around the tree trunks. The word \"FLUX\" is painted over it in big, red brush strokes with visible texture")
    
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider("Width", 128, 2048, 1024, step=16)
            height = st.slider("Height", 128, 2048, 1024, step=16)
        with col2:
            num_steps = st.slider("Number of steps", 1, 100, 50)
            guidance = st.slider("Guidance", 1.0, 10.0, 4.0)
        
        seed = st.number_input("Seed (-1 for random)", value=-1, step=1)

    if st.button("Generate"):
        generator = st.session_state.generator
        
        if seed == -1:
            seed_val = None
        else:
            seed_val = int(seed)
            
        with st.spinner("Generating image..."):
            image = generator.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                seed=seed_val
            )
            
        st.image(image, caption=prompt)

if __name__ == "__main__":
    main()
