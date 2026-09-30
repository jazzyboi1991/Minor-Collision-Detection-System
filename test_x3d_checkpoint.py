"""Regression test for the deployed X3D checkpoint/model contract."""

import unittest

import torch

from app.prediction_job import _get_model


class X3DCheckpointTest(unittest.TestCase):
    def test_service_checkpoint_loads_strictly_and_produces_binary_logits(self):
        """Catch X3D head changes that are incompatible with the deployed weights."""
        model = _get_model()
        activations = []
        hook = model.inception5b.register_forward_hook(
            lambda _module, _inputs, output: activations.append(output)
        )

        try:
            with torch.inference_mode():
                logits = model(torch.zeros(1, 3, 30, 224, 224))
        finally:
            hook.remove()

        self.assertEqual((1, 2), tuple(logits.shape))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertEqual((1, 2048, 1, 7, 7), tuple(activations[0].shape))


if __name__ == "__main__":
    unittest.main()
