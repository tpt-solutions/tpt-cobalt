# Hugging Face Integration Bridge for TPT
from __future__ import annotations

from typing import Any


def is_hf_available() -> bool:
    """Check if Hugging Face transformers is installed."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def load_model(model_name: str, device: str = "tpt:0", **kwargs) -> Any:
    """
    Load a Hugging Face model with TPT backend.
    
    Args:
        model_name: Hugging Face model name or path.
        device: Target device string (e.g., "tpt:0").
        **kwargs: Additional arguments passed to the model loader.
    
    Returns:
        The loaded model.
    """
    if not is_hf_available():
        raise RuntimeError("Hugging Face transformers not installed.")
    
    import transformers

    from tptr.pytorch import register_backend
    
    register_backend()
    
    model = transformers.AutoModel.from_pretrained(model_name, **kwargs)
    
    if device.startswith("tpt"):
        import torch
        idx = int(device.split(":")[-1]) if ":" in device else 0
        model = model.to(f"cuda:{idx}" if torch.cuda.is_available() else "cpu")
    
    return model


def load_tokenizer(model_name: str, **kwargs) -> Any:
    """
    Load a Hugging Face tokenizer.
    
    Args:
        model_name: Model name or path.
        **kwargs: Additional arguments.
    
    Returns:
        The loaded tokenizer.
    """
    if not is_hf_available():
        raise RuntimeError("Hugging Face transformers not installed.")
    
    import transformers
    return transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)


def run_inference(model, tokenizer, text: str, max_length: int = 128) -> dict[str, Any]:
    """
    Run inference on a model with TPT backend.
    
    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        text: Input text.
        max_length: Maximum sequence length.
    
    Returns:
        Dictionary with inference results.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    
    import torch
    with torch.no_grad():
        outputs = model(**inputs)
    
    return {
        "logits": outputs.logits if hasattr(outputs, "logits") else outputs.last_hidden_state,
        "inputs": inputs,
    }


class TptHFModel:
    """
    Wrapper for Hugging Face models running on TPT.
    
    Example:
        bridge = TptHFModel("bert-base-uncased")
        result = bridge.predict("Hello world")
    """

    def __init__(self, model_name: str, device: str = "tpt:0", **kwargs):
        self.model = load_model(model_name, device, **kwargs)
        self.tokenizer = load_tokenizer(model_name, **kwargs)
        self.device = device

    def predict(self, text: str, **kwargs) -> dict[str, Any]:
        return run_inference(self.model, self.tokenizer, text, **kwargs)

    def embed(self, text: str) -> Any:
        """Get embeddings for text."""
        result = run_inference(self.model, self.tokenizer, text)
        return result["logits"]

    def __repr__(self):
        return f"TptHFModel(model={self.model.config._name_or_path}, device={self.device})"