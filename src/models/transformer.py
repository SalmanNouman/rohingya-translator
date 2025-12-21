import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from typing import Optional, Tuple, Dict, Any, List
import gc
import os
import json

class RohingyaTranslator(nn.Module):
    def __init__(
        self,
        config: dict,
        base_model_name: str = "facebook/nllb-200-distilled-600M"
    ):
        super().__init__()
        self.config = config
        self.max_length = config.get('max_length', 128)
        self.src_lang = config.get('src_lang', 'eng_Latn')
        self.tgt_lang = config.get('tgt_lang', 'ben_Beng')  # Using Bengali as a proxy for Rohingya in NLLB
        self.base_model_name = base_model_name
        
        # Clear CUDA cache before model initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Initialize the tokenizer and model with memory optimizations
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            model_max_length=self.max_length
        )
        
        # Add Rohingya special token if not present (NLLB has many, but we might want custom)
        special_tokens = config.get('special_tokens', ['<roh>'])
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Load model with memory optimizations
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Resize token embeddings to account for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing for memory efficiency during training
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Set source and target language
        self.tokenizer.src_lang = self.src_lang

    def save_pretrained(self, save_directory: str):
        """Save the model, tokenizer, and configuration to a directory."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the model
        self.model.save_pretrained(save_directory)
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(save_directory)
        
        # Save the configuration
        config_dict = {
            'max_length': self.max_length,
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang,
            'base_model_name': self.base_model_name,
            **self.config
        }
        
        config_path = os.path.join(save_directory, 'rohingya_translator_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str, **kwargs):
        """Load a model from a pretrained directory."""
        # Load configuration
        config_path = os.path.join(pretrained_path, 'rohingya_translator_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(config=config, base_model_name=config['base_model_name'])
        
        # Load model and tokenizer from the saved state
        instance.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_path)
        instance.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        
        # Set language codes
        instance.src_lang = config.get('src_lang', 'eng_Latn')
        instance.tgt_lang = config.get('tgt_lang', 'ben_Beng')
        instance.tokenizer.src_lang = instance.src_lang
        
        return instance
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate translations."""
        # Get target language ID for NLLB
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=forced_bos_token_id,
            **kwargs
        )
    
    def translate(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> List[str]:
        """Translate a list of texts."""
        # Prepare the inputs
        self.tokenizer.src_lang = self.src_lang
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length
        )
        
        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate translations
        translated_tokens = self.generate(
            **inputs,
            max_length=max_length or self.max_length,
            num_beams=num_beams,
            **kwargs
        )
        
        # Decode the generated tokens
        translations = self.tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )
        
        return translations
