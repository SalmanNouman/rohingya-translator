import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, MBart50Tokenizer, MBartForConditionalGeneration
from typing import Optional, Tuple, Dict, Any
import gc
import os
import json

class RohingyaTranslator(nn.Module):
    def __init__(
        self,
        config: dict,
        base_model_name: str = "facebook/mbart-large-50"
    ):
        super().__init__()
        self.config = config
        self.max_length = config.get('max_length', 128)
        self.src_lang = config.get('src_lang', 'en_XX')
        self.tgt_lang = 'ar_AR'  # Using Arabic as a proxy for Rohingya since it's not in mBART-50's vocabulary
        self.base_model_name = base_model_name
        
        # Clear CUDA cache before model initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Initialize the tokenizer and model with memory optimizations
        self.tokenizer = MBart50Tokenizer.from_pretrained(
            base_model_name,
            model_max_length=self.max_length,
            padding_side="right",
            truncation_side="right"
        )
        
        # Add Rohingya special token and adjust vocab
        special_tokens = config.get('special_tokens', ['<roh>'])
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Load model with memory optimizations
        self.model = MBartForConditionalGeneration.from_pretrained(
            base_model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Resize token embeddings to account for new special tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Set source and target language
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

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
        instance.model = MBartForConditionalGeneration.from_pretrained(pretrained_path)
        instance.tokenizer = MBart50Tokenizer.from_pretrained(pretrained_path)
        
        # Set language codes
        instance.tokenizer.src_lang = config['src_lang']
        instance.tokenizer.tgt_lang = config['tgt_lang']
        
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
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang],
            **kwargs
        )
    
    def translate(
        self,
        texts: list[str],
        max_length: Optional[int] = None,
        num_beams: int = 5,
        **kwargs
    ) -> list[str]:
        """Translate a list of texts."""
        # Prepare the inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length or self.max_length
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
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
