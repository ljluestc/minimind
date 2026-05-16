import os
import sys
import unittest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.model_minimind import MiniMindConfig, Attention, precompute_freqs_cis


class TestGatedAttention(unittest.TestCase):
    def _build_inputs(self, config, batch_size=2, seq_len=6):
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=seq_len + 4,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        position_embeddings = (freqs_cos[:seq_len], freqs_sin[:seq_len])
        return x, position_embeddings

    def test_gate_parameter_shapes(self):
        head_cfg = MiniMindConfig(attn_gate=True, attn_gate_type="headwise")
        ele_cfg = MiniMindConfig(attn_gate=True, attn_gate_type="elementwise")
        head_attn = Attention(head_cfg)
        ele_attn = Attention(ele_cfg)
        self.assertEqual(tuple(head_attn.attn_gate_param.shape), (head_cfg.num_attention_heads,))
        self.assertEqual(tuple(ele_attn.attn_gate_param.shape), (ele_cfg.num_attention_heads, ele_cfg.head_dim))

    def test_headwise_gate_controls_output_magnitude(self):
        cfg = MiniMindConfig(attn_gate=True, attn_gate_type="headwise")
        attn = Attention(cfg)
        x, pos = self._build_inputs(cfg)
        with torch.no_grad():
            attn.attn_gate_param.fill_(12.0)
            out_large, _ = attn(x, pos)
            attn.attn_gate_param.fill_(-20.0)
            out_small, _ = attn(x, pos)
        self.assertGreater(out_large.norm().item(), out_small.norm().item() * 20)

    def test_elementwise_gate_controls_output_magnitude(self):
        cfg = MiniMindConfig(attn_gate=True, attn_gate_type="elementwise")
        attn = Attention(cfg)
        x, pos = self._build_inputs(cfg)
        with torch.no_grad():
            attn.attn_gate_param.fill_(8.0)
            out_large, _ = attn(x, pos)
            attn.attn_gate_param.fill_(-20.0)
            out_small, _ = attn(x, pos)
        self.assertGreater(out_large.norm().item(), out_small.norm().item() * 20)

    def test_gate_keeps_kv_cache_path_working(self):
        cfg = MiniMindConfig(attn_gate=True, attn_gate_type="elementwise")
        attn = Attention(cfg)
        x1, pos1 = self._build_inputs(cfg, seq_len=4)
        x2, pos2 = self._build_inputs(cfg, seq_len=2)
        out1, past = attn(x1, pos1, use_cache=True)
        out2, past2 = attn(x2, pos2, past_key_value=past, use_cache=True)
        self.assertEqual(out1.shape[1], 4)
        self.assertEqual(out2.shape[1], 2)
        self.assertEqual(past2[0].shape[1], 6)


if __name__ == "__main__":
    unittest.main()
