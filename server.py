"""
server.py - Streamlit server for GPT-OSS text generation
A simple web interface to interact with your trained model
"""
import streamlit as st
import torch
import os
from pathlib import Path

from inference import load_model_and_generate, load_gptoss20b_and_generate
from architecture.gptoss import Transformer, ModelConfig
from architecture.tokenizer import get_tokenizer


# Page configuration
st.set_page_config(
    page_title="GPT-OSS Text Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_sapphire_model(checkpoint_path, device):
    """Load and cache the Sapphire model"""
    try:
        if not os.path.exists(checkpoint_path):
            return None, f"Checkpoint not found: {checkpoint_path}"
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Extract config
        vocab_size = 201088
        if isinstance(checkpoint, dict) and "config" in checkpoint:
            config_dict = checkpoint["config"]
            model_size = config_dict.get("model_size", "toy")
        else:
            model_size = "toy"
        
        # Build config
        from train import build_config
        cfg = build_config(model_size, vocab_size)
        
        # Create and load model
        model = Transformer(cfg, device=device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            iter_num = checkpoint.get("iter_num", "unknown")
            val_loss = checkpoint.get("best_val_loss", "unknown")
        else:
            model.load_state_dict(checkpoint)
            iter_num = "unknown"
            val_loss = "unknown"
        
        model.eval()
        
        return model, {"iter": iter_num, "val_loss": val_loss, "model_size": model_size}
    
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def generate_from_model(model, prompt, max_tokens, temperature, top_k, device):
    """Generate text from the loaded model"""
    from inference import generate_text
    
    try:
        with torch.no_grad():
            generated = generate_text(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
        return generated
    except Exception as e:
        return f"Error generating text: {str(e)}"


def main():
    # Minimal header with icon
    st.markdown(
        """
        <div style='text-align: right; padding: 10px;'>
            <span style='font-size: 24px;'>💎</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_type = st.selectbox(
            "Select Model",
            ["Sapphire (Custom)", "GPT-OSS 20B"],
            index=0
        )
        
        # Checkpoint/weights selection
        if model_type == "Sapphire (Custom)":
            st.subheader("Model Checkpoint")
            # Find available checkpoints
            checkpoint_dir = "model"
            available_checkpoints = []
            if os.path.exists(checkpoint_dir):
                for file in os.listdir(checkpoint_dir):
                    if file.endswith(".pt"):
                        available_checkpoints.append(os.path.join(checkpoint_dir, file))
            
            if available_checkpoints:
                checkpoint_path = st.selectbox(
                    "Checkpoint",
                    available_checkpoints,
                    index=0 if "gptoss_best.pt" not in str(available_checkpoints) else 
                          [i for i, x in enumerate(available_checkpoints) if "best" in x][0]
                )
            else:
                checkpoint_path = st.text_input(
                    "Checkpoint path",
                    value="model/gptoss_best.pt"
                )
        else:
            st.subheader("Weights Directory")
            weights_dir = st.text_input(
                "Path to weights",
                value="architecture/open-gpt-oss/weights"
            )
        
        # Device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        st.text_input("Device", value=device, disabled=True)
        
        st.markdown("---")
        
        # Generation parameters
        st.subheader("Generation Parameters")
        
        max_tokens = st.slider(
            "Max tokens",
            min_value=10,
            max_value=500,
            value=20,
            step=10
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.8,
            step=0.1
        )
        
        top_k = st.slider(
            "Top-k",
            min_value=1,
            max_value=500,
            value=200,
            step=10
        )
    
    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load model once and cache in session state
    if "loaded_model" not in st.session_state or "loaded_model_type" not in st.session_state:
        st.session_state.loaded_model = None
        st.session_state.loaded_model_type = None
        st.session_state.loaded_model_path = None
    
    # Check if we need to reload the model (model type or path changed)
    current_model_key = f"{model_type}_{checkpoint_path if model_type == 'Sapphire (Custom)' else weights_dir}"
    if st.session_state.loaded_model_path != current_model_key:
        with st.spinner("Loading model..."):
            if model_type == "Sapphire (Custom)":
                model, model_info = load_sapphire_model(checkpoint_path, device)
                if model is None:
                    st.error(f"❌ {model_info}")
                    st.stop()
                st.session_state.loaded_model = model
                st.session_state.loaded_model_type = "sapphire"
                st.session_state.loaded_model_path = current_model_key
            else:
                # For GPT-OSS 20B, we'll load on demand since it's called via function
                st.session_state.loaded_model = None
                st.session_state.loaded_model_type = "gptoss20b"
                st.session_state.loaded_model_path = current_model_key
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input at the bottom
    if prompt := st.chat_input("Enter your prompt..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                try:
                    if model_type == "Sapphire (Custom)":
                        # Use cached model
                        generated = generate_from_model(
                            model=st.session_state.loaded_model,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            device=device
                        )
                    else:
                        # GPT-OSS 20B
                        generated = load_gptoss20b_and_generate(
                            weights_dir=weights_dir,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            device=device
                        )
                    
                    st.markdown(generated)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": generated})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
