#!/usr/bin/env python3
"""Quick LM Studio probe at higher range."""
import sys
sys.path.insert(0, '.')
from src.api_client_v2 import probe_model_tokens

r = probe_model_tokens(
    'lm_studio',
    'A researcher tests whether plants grow better under white or colored light. Design the experiment briefly.',
    max_range=1024
)
print(f"LM Studio: saturation={r.saturation_tokens}, recommended_max={r.recommended_max}")
