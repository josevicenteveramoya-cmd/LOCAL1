import os
import cv2
import gradio as gr
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def process_image(image, model_name, outscale, fp32):
    if image is None:
        return None
    
    # Selección del modelo según la opción escogida
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRNet_x4plus.pth'
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth'

    model_path = os.path.join('weights', f'{model_name}.pth')
    if not os.path.isfile(model_path):
        os.makedirs('weights', exist_ok=True)
        torch.hub.download_url_to_file(file_url, model_path)

    # Determinar si usar GPU (CUDA) o CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=not fp32,
        gpu_id=0 if torch.cuda.is_available() else None
    )

    # Gradio pasa las imágenes en formato RGB, cv2 trabaja en BGR
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
        output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        return output_rgb
    except Exception as error:
        print('Error durante el procesamiento:', error)
        return None

# Crear la Interfaz Gráfica con Gradio
with gr.Blocks(title="Real-ESRGAN WebUI") as demo:
    gr.Markdown("# 🚀 Real-ESRGAN WebUI - Mejora y Reescalado de Imágenes")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Imagen Original", type="numpy")
            model_dropdown = gr.Dropdown(
                choices=['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B', 'realesr-animevideov3'],
                value='RealESRGAN_x4plus',
                label="Modelo de IA"
            )
            outscale_slider = gr.Slider(minimum=1, maximum=8, value=4, step=1, label="Factor de Reescalado (X)")
            fp32_checkbox = gr.Checkbox(label="Usar precisión FP32 (Actívalo si estás usando CPU)", value=True)
            btn = gr.Button("Reescalar e Incrementar Calidad", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Resultado Mejorado")

    btn.click(
        fn=process_image,
        inputs=[input_image, model_dropdown, outscale_slider, fp32_checkbox],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)