import inspect
from transformers import PreTrainedTokenizerFast, MBart50TokenizerFast
from src.data.huggingface_dataset import RohingyaHFDataset

def test_tokenizer_type_hint_is_generic():
    """Ensure the tokenizer type hint is generic, not tied to mBART."""
    init_signature = inspect.signature(RohingyaHFDataset.__init__)
    tokenizer_param = init_signature.parameters['tokenizer']
    
    # Check if the annotation is generic
    # It should effectively be PreTrainedTokenizerFast or similar
    # We want to ensure it is NOT explicitly MBart50TokenizerFast
    
    annotation = tokenizer_param.annotation
    
    # If the annotation is a string (forward reference), resolve it or check string value
    if isinstance(annotation, str):
        assert "MBart50TokenizerFast" not in annotation, "Type hint is still MBart50TokenizerFast"
    else:
        assert annotation != MBart50TokenizerFast, "Type hint is still MBart50TokenizerFast"
        assert issubclass(annotation, PreTrainedTokenizerFast) or annotation == PreTrainedTokenizerFast, \
            f"Expected PreTrainedTokenizerFast, got {annotation}"

def test_preprocess_returns_lists_not_tensors():
    """
    This test is for the NEXT task, but placing it here to fail now.
    Preprocess function should return lists/dicts, not PyTorch tensors.
    """
    # We can mock the tokenizer and dataset
    class MockTokenizer:
        def __call__(self, texts, **kwargs):
            # Simulate returning tensors if return_tensors='pt' is passed
            if kwargs.get('return_tensors') == 'pt':
                return {'input_ids': 'fake_tensor'}
            return {'input_ids': [1, 2, 3]}

    dataset = RohingyaHFDataset.__new__(RohingyaHFDataset)
    dataset.tokenizer = MockTokenizer()
    dataset.max_length = 128
    
    # Mock romanizer as it is now required by preprocess_function
    class MockRomanizer:
        def romanize(self, text):
            return text
            
    dataset.romanizer = MockRomanizer()
    
    examples = {
        'translation': [
            {'en': 'hello', 'roh': 'hola'},
            {'en': 'world', 'roh': 'mundo'}
        ]
    }
    
    output = dataset.preprocess_function(examples)
    
    assert isinstance(output['input_ids'], list), "input_ids should be a list"
    assert isinstance(output['labels'], list), "labels should be a list"
    assert output['input_ids'] != 'fake_tensor', "Should not return tensors"
