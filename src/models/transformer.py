import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer, MBart50Tokenizer, MBartForConditionalGeneration
from typing import Optional, Tuple

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
        self.tgt_lang = config.get('tgt_lang', 'roh_XX')
        
        # Initialize the tokenizer and model
        self.tokenizer = MBart50Tokenizer.from_pretrained(base_model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(base_model_name)
        
        # Set source and target language
        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Tensor of input token ids
            attention_mask: Mask to avoid attention on padding tokens
            decoder_input_ids: Decoder input ids for teacher forcing
            labels: Target sequence labels
            
        Returns:
            Model outputs (loss, logits, etc.)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )
        return outputs
    
    def translate(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Translate a text from source language to Rohingya.
        
        Args:
            text: Input text in source language
            max_length: Maximum length of generated translation
            
        Returns:
            Translated text in Rohingya
        """
        if max_length is None:
            max_length = self.max_length
            
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
            src_lang=self.src_lang
        )
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=4,
                length_penalty=0.6,
                early_stopping=True,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang]
            )
        
        # Decode the generated tokens
        translated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return translated_text
