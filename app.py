import gradio as gr
import torch
from model import BigramLanguageModel

# Cargar el modelo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint = torch.load('quijote_gpt.pth', map_location=device)

# Extraer configuración y vocabulario
config = checkpoint['config']
vocab = checkpoint['vocab']
stoi = vocab['stoi']
itos = vocab['itos']

# Funciones encode/decode
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Crear modelo y cargar pesos
model = BigramLanguageModel(
    vocab_size=vocab['vocab_size'],
    n_embd=config['n_embd'],
    block_size=config['block_size'],
    n_head=config['n_head'],
    n_layer=config['n_layer'],
    dropout=config['dropout']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Función de generación
def generate_text(prompt, max_length, temperature):
    if not prompt:
        prompt = "En un lugar de"
    
    # Codificar el prompt
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    # Generar
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_length, temperature=temperature)
    
    # Decodificar
    output = decode(generated[0].tolist())
    return output

# Interfaz Gradio
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Escribe el inicio de tu texto",
            placeholder="En un lugar de la Mancha...",
            lines=3
        ),
        gr.Slider(
            minimum=50,
            maximum=2000,
            value=500,
            step=50,
            label="Longitud de la continuación (caracteres)"
        ),
        gr.Slider(
            minimum=0.5,
            maximum=1.5,
            value=1.0,
            step=0.1,
            label="Temperatura (creatividad)",
            info="Más bajo = más coherente, Más alto = más creativo"
        )
    ],
    outputs=gr.Textbox(label="Texto Generado", lines=15),
    title="🎭 Generador de Texto al Estilo Quijote",
    description="Escribe el inicio de una historia y el modelo continuará al estilo cervantino",
    examples=[
        ["En un lugar de la Mancha", 400, 1.0],
        ["Don Quijote, caballero de", 300, 0.8],
        ["Sancho Panza dijo:", 350, 1.2],
    ],
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
