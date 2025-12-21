import pandas
import transformers
import pytest
from packaging import version

def test_pandas_security_update():
    """Ensure pandas is updated to a version newer than the vulnerable 2.3.3"""
    current_version = version.parse(pandas.__version__)
    vulnerable_version = version.parse("2.3.3")
    assert current_version > vulnerable_version, f"Pandas version {pandas.__version__} is vulnerable. Upgrade required."

def test_transformers_security_update():
    """Ensure transformers is updated to a version newer than the vulnerable 4.57.3"""
    current_version = version.parse(transformers.__version__)
    vulnerable_version = version.parse("4.57.3")
    assert current_version > vulnerable_version, f"Transformers version {transformers.__version__} is vulnerable. Upgrade required."
