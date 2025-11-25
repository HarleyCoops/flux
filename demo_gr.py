import torch
import gradio as gr
from flux import Flux2Generator

def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    generator = Flux2Generator(model_name, device)

    with gr.Blocks() as demo:
        gr.Markdown(f"# Flux Image Generation Demo - Model: {model_name}")
        
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="a photo of a forest with mist swirling around the tree trunks. The word \"FLUX\" is painted over it in big, red brush strokes with visible texture")
                
                with gr.Accordion("Advanced Options", open=False):
                    width = gr.Slider(128, 2048, 1024, step=16, label="Width")
                    height = gr.Slider(128, 2048, 1024, step=16, label="Height")
                    num_steps = gr.Slider(1, 100, 50, step=1, label="Number of steps")
                    guidance = gr.Slider(1.0, 10.0, 4.0, step=0.1, label="Guidance")
                    seed = gr.Number(-1, label="Seed (-1 for random)", precision=0)
                
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                seed_output = gr.Number(label="Used Seed")
                
        def generate(width, height, num_steps, guidance, seed, prompt):
            if seed == -1:
                seed = None
            else:
                seed = int(seed)
                
            print(f"Generating '{prompt}' with seed {seed}")
            image = generator.generate(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                seed=seed
            )
            return image, seed

        generate_btn.click(
            fn=generate,
            inputs=[width, height, num_steps, guidance, seed, prompt],
            outputs=[output_image, seed_output],
        )

    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="black-forest-labs/FLUX.2-dev", help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    args = parser.parse_args()

    demo = create_demo(args.name, args.device)
    demo.launch(share=args.share)
