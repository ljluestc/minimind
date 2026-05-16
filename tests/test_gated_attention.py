import unittest
import torch

from model.model_minimind import MiniMindConfig, Attention


class TestGatedAttention(unittest.TestCase):
    def _build_attention(self, attn_gate=False, attn_gate_type="elementwise", flash_attn=False):
        config = MiniMindConfig(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            dropout=0.0,
            flash_attn=flash_attn,
            attn_gate=attn_gate,
            attn_gate_type=attn_gate_type,
            max_position_embeddings=128
        )
        attn = Attention(config).eval()
        return attn

    def test_gate_parameter_shape_elementwise(self):
        attn = self._build_attention(attn_gate=True, attn_gate_type="elementwise")
        self.assertIsNotNone(attn.attn_gate_param)
        self.assertEqual(tuple(attn.attn_gate_param.shape), (attn.n_local_heads, attn.head_dim))

    def test_gate_parameter_shape_headwise(self):
        attn = self._build_attention(attn_gate=True, attn_gate_type="headwise")
        self.assertIsNotNone(attn.attn_gate_param)
        self.assertEqual(tuple(attn.attn_gate_param.shape), (attn.n_local_heads,))

    def test_no_gate_parameter_when_disabled(self):
        attn = self._build_attention(attn_gate=False)
        self.assertIsNone(attn.attn_gate_param)

    def test_kv_cache_with_gate(self):
        torch.manual_seed(0)
        attn = self._build_attention(attn_gate=True, attn_gate_type="elementwise", flash_attn=False)
        x1 = torch.randn(2, 3, 32)
        pos1 = (torch.ones(3, 8), torch.zeros(3, 8))
        y1, past = attn(x1, pos1, use_cache=True)
        self.assertEqual(tuple(y1.shape), (2, 3, 32))
        self.assertIsNotNone(past)
        self.assertEqual(tuple(past[0].shape), (2, 3, attn.n_local_kv_heads, attn.head_dim))
        x2 = torch.randn(2, 1, 32)
        pos2 = (torch.ones(1, 8), torch.zeros(1, 8))
        y2, past2 = attn(x2, pos2, past_key_value=past, use_cache=True)
        self.assertEqual(tuple(y2.shape), (2, 1, 32))
        self.assertEqual(tuple(past2[0].shape), (2, 4, attn.n_local_kv_heads, attn.head_dim))

    def test_flash_attention_path_with_gate(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            self.skipTest("scaled_dot_product_attention is not available in this torch build")
        torch.manual_seed(0)
        attn = self._build_attention(attn_gate=True, attn_gate_type="headwise", flash_attn=True)
        x = torch.randn(2, 4, 32)
        pos = (torch.ones(4, 8), torch.zeros(4, 8))
        y, _ = attn(x, pos, use_cache=False, attention_mask=torch.ones(2, 4))
        self.assertEqual(tuple(y.shape), (2, 4, 32))


if __name__ == "__main__":
    unittest.main()
