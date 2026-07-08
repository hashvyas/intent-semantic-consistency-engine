import os
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class SemanticGateResult:
    label: str
    label_id: int
    confidence: float

_PREDICTOR_CACHE: Dict[tuple, 'SemanticGatePredictor'] = {}

def resolve_model_path(model_path: str) -> str:
    """Resolve model path against absolute path and the b3_semantic_gate directory."""
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    # Try resolving relative to this script's directory (b3_semantic_gate)
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    return os.path.abspath(model_path)

class SemanticGatePredictor:
    def __init__(self, model_path: str, max_length: int = 256, device: Optional[str] = None):
        self.raw_path = model_path
        self.model_path = resolve_model_path(model_path)
        self.max_length = max_length
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path} (resolved from {model_path})")
            
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device).eval()
        
        # Fetch label map from config or defaults
        self.id2label = getattr(self.model.config, "id2label", {0: "BENIGN", 1: "MALICIOUS"})

    def predict(self, texts: List[str], batch_size: int = 32) -> List[SemanticGateResult]:
        """Perform batched inference on a list of input texts.

        Parameters
        ----------
        texts : List[str]
            Texts to classify.
        batch_size : int, optional
            Batch size for inference, by default 32.

        Returns
        -------
        List[SemanticGateResult]
            Structured classification results containing label, label_id, and confidence.
        """
        results: List[SemanticGateResult] = []
        if not texts:
            return results
            
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(
                    batch, 
                    max_length=self.max_length, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                out = self.model(**enc)
                probs = torch.softmax(out.logits, dim=1).cpu().numpy()
                
                preds = probs.argmax(axis=1)
                confs = probs.max(axis=1)
                
                for pred, conf in zip(preds, confs):
                    label_name = self.id2label.get(int(pred), f"LABEL_{pred}")
                    results.append(SemanticGateResult(
                        label=label_name,
                        label_id=int(pred),
                        confidence=float(conf)
                    ))
        return results

def get_predictor(model_path: str, max_length: int = 256, device: Optional[str] = None) -> SemanticGatePredictor:
    """Get or create cached SemanticGatePredictor instance for the given configuration."""
    resolved_path = resolve_model_path(model_path)
    cache_key = (resolved_path, max_length, str(device))
    if cache_key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[cache_key] = SemanticGatePredictor(resolved_path, max_length, device)
    return _PREDICTOR_CACHE[cache_key]
