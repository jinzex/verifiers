import base64
import io

import numpy as np
from verifiers.v1.clients.train import _normalize_native_routed_experts


def test_normalize_native_routed_experts():
    array = np.arange(48, dtype=np.uint8).reshape(3, 2, 8)
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)

    payload = _normalize_native_routed_experts(base64.b64encode(buffer.getvalue()).decode(), start=5)

    assert payload["shape"] == [3, 2, 8]
    assert payload["dtype"] == "uint8"
    assert payload["start"] == 5
    assert base64.b64decode(payload["data"]) == array.tobytes()
