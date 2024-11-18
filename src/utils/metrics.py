from typing import List
import sacrebleu
import torch
from transformers import PreTrainedTokenizer

def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for translation quality evaluation.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations
        
    Returns:
        BLEU score
    """
    # Prepare references in required format (list of list of references)
    refs = [[ref] for ref in references]
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(predictions, refs)
    return bleu.score

def decode_predictions(
    outputs: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    skip_special_tokens: bool = True
) -> List[str]:
    """
    Decode model outputs to text.
    
    Args:
        outputs: Model output tensor
        tokenizer: Tokenizer for decoding
        skip_special_tokens: Whether to skip special tokens in output
        
    Returns:
        List of decoded texts
    """
    predictions = []
    for output in outputs:
        pred = tokenizer.decode(
            output,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=True
        )
        predictions.append(pred)
    return predictions
