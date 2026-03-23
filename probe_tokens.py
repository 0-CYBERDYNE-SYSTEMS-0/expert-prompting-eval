#!/usr/bin/env python3
"""Run token saturation probe for both models."""
import sys
sys.path.insert(0, '.')
from src.api_client_v2 import run_probe_sequence, set_openai_key

set_openai_key('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')
result = run_probe_sequence()
print("\n=== PROBE RESULTS ===")
for model, data in result.items():
    print(f"{model}: saturation={data['saturation']}, recommended_max={data['recommended_max']}")
