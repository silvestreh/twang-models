"""Makes the two ONNX graphs Twang's Convert to MIDI runs (crates/transcribe) from YourMT3+'s PyTorch
checkpoint, and checks them against PyTorch.

    python -I scripts/a2m-export/export.py --space <checkout of the YourMT3 Space> --out <folder>

The Space is https://huggingface.co/spaces/mimbres/YourMT3 (Apache-2.0 code, Copyright 2024 The YourMT3
Authors; commit 5e66c1e), cloned with its one checkpoint:

    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/spaces/mimbres/YourMT3 space
    curl -L -o space/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt \\
      https://huggingface.co/spaces/mimbres/YourMT3/resolve/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt
    # 561,544,628 bytes, SHA-256 ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25; no account or token

It needs torch, torchaudio, lightning, transformers 4.45.1, einops, deprecated, onnx, onnxruntime, mido,
mir_eval, librosa and smart-open (wandb, which the Space imports for logging only, is stood in for). Made
with Python 3.13, torch 2.15 and onnxruntime 1.32; the files' sizes and SHA-256 are pinned in
crates/transcribe/src/store.rs (`FILES`), so another torch may give other bytes: pin those then.

The model is "YPTF.MoE+Multi (noPS)", the Space's default: a Perceiver-TF encoder with mixture-of-experts
layers and a 13-channel T5 decoder (47 M parameters), in float32.

encoder.onnx: spec (B, 110, 1024) f32, the log spectrogram Twang computes (torch's complex STFT has no ONNX
              export) -> cross_k_i, cross_v_i for the 8 decoder layers, each (B*13, 6, 110, 64)
decoder.onnx: input_ids (N, 1) i64, self_k_i/self_v_i (N, 6, P, 64) with P >= 0, cross_k_i/cross_v_i (N, 6, 110, 64)
              -> logits (N, 1, 596), new_self_k_i/new_self_v_i (N, 6, P+1, 64); a row is one channel of one segment
"""
import argparse, os, pathlib, sys, types
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--space", required=True, help="the checkout of the YourMT3 Space, with the checkpoint")
parser.add_argument("--out", required=True, help="where encoder.onnx and decoder.onnx go")
cli = parser.parse_args()
SPACE = str(pathlib.Path(cli.space).resolve())
out = pathlib.Path(cli.out).resolve(); out.mkdir(parents=True, exist_ok=True)
SPEC_OUTSIDE = True

CKPT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
# As the Space's app.py for "YPTF.MoE+Multi (noPS)", in float32.
ARGS = [CKPT, '-p', '2024', '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
        '-sqr', '1', '-ff', 'moe', '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
        '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', '32']
os.chdir(SPACE)  # the Space finds its checkpoints from its own folder
sys.path.insert(0, os.path.join(SPACE, "amt/src")); sys.path.insert(0, SPACE)
try:
    import wandb  # noqa: F401
except ImportError:  # only logging: not needed to run the model
    sys.modules["wandb"] = types.ModuleType("wandb"); sys.modules["wandb"].Table = lambda *a, **k: None
from model_helper import load_model_checkpoint
model = load_model_checkpoint(args=ARGS, device="cpu").float().eval()
import torch, torch.nn as nn
L = len(model.decoder.decoder.block); C = model.decoder.num_channels  # 8 layers, 13 channels

class Encoder(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        m = s.m
        if not SPEC_OUTSIDE:
            x = m.spectrogram(x)
        x = m.pre_encoder(x)
        h = m.encoder(inputs_embeds=x)["last_hidden_state"]
        h = m.pre_decoder(h)                              # (B, C, T, D)
        b, c, t, d = h.shape
        h = h.reshape(b * c, t, d)
        outs = []
        for blk in m.decoder.decoder.block:
            a = blk.layer[1].EncDecAttention
            shape = lambda z: z.view(b * c, -1, a.n_heads, a.key_value_proj_dim).transpose(1, 2)
            outs += [shape(a.k(h)), shape(a.v(h))]
        return tuple(outs)

class Decoder(nn.Module):
    """One step on N independent rows (a row is one channel of one segment)."""
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, ids, *kv):
        m = s.m
        self_kv, cross_kv = kv[:2 * L], kv[2 * L:]
        past = tuple((self_kv[2 * i], self_kv[2 * i + 1], cross_kv[2 * i], cross_kv[2 * i + 1]) for i in range(L))
        n = ids.shape[0]
        t = cross_kv[0].shape[2]
        # Only its shape is read: the cross-attention keys and values are given.
        enc = torch.zeros(n, t, 1, dtype=cross_kv[0].dtype)
        o = m.decoder.decoder(inputs_embeds=m.embed_tokens(ids), encoder_hidden_states=enc, past_key_values=past, use_cache=True, return_dict=True)
        logits = m.lm_head(o.last_hidden_state)               # (N, 1, V)
        new = []
        for p in o.past_key_values:
            new += [p[0], p[1]]
        return (logits, *new)

enc, dec = Encoder(model).eval(), Decoder(model).eval()
torch.manual_seed(0)

# The mixture-of-experts block routes with Python control flow on the data (`tolist`, `continue`), which a
# trace would freeze. For the export it is replaced by the same sum in tensor operations (NonZero, Gather,
# ScatterND): each expert on the tokens routed to it, weighted by the top-2 routing weights.
import torch.nn.functional as F
from model.ff_layer import MixtralSparseMoeBlock
def dense_moe(self, hs):
    b, n, d = hs.shape
    x = hs.reshape(-1, d)
    logits = self.gate(x)
    w = F.softmax(logits, dim=1, dtype=torch.float)
    top, idx = torch.topk(w, self.top_k, dim=-1)
    top = top / top.sum(dim=-1, keepdim=True)
    dense = torch.zeros_like(w).scatter(1, idx, top)          # (tokens, experts): 0 where not picked
    out = torch.zeros_like(x)
    for e in range(self.num_experts):
        rows = torch.nonzero(dense[:, e] > 0).squeeze(1)       # the tokens routed to this expert
        y = self.experts[e](x.index_select(0, rows)) * dense[:, e].index_select(0, rows).unsqueeze(1)
        out = out.index_add(0, rows, y)
    return out.reshape(b, n, d), logits
sparse_moe = MixtralSparseMoeBlock.forward
def use_dense(on): MixtralSparseMoeBlock.forward = dense_moe if on else sparse_moe
# The reference for the parity check below, with the original block.
ref_audio = torch.randn(3, 1, 32767) * 0.05
_t = torch.arange(32767) / 16000
ref_audio[0, 0] += 0.3 * torch.sin(2 * np.pi * 440 * _t) * torch.exp(-3 * _t)
ref_audio[1, 0] += 0.2 * torch.sign(torch.sin(2 * np.pi * 110 * _t))
with torch.no_grad():
    ref_cross = Encoder(model).eval()(model.spectrogram(ref_audio) if SPEC_OUTSIDE else ref_audio)
use_dense(True)
B = 2
audio = torch.randn(B, 1, 32767) * 0.1
x_in = model.spectrogram(audio) if SPEC_OUTSIDE else audio
with torch.no_grad():
    cross = enc(x_in)
print("cross kv", cross[0].shape, "spec", tuple(model.spectrogram(audio).shape))
T = cross[0].shape[2]
cross_names = [f"cross_{n}_{i}" for i in range(L) for n in "kv"]
self_names = [f"self_{n}_{i}" for i in range(L) for n in "kv"]
new_names = [f"new_self_{n}_{i}" for i in range(L) for n in "kv"]
in_name = "spec" if SPEC_OUTSIDE else "audio"
torch.onnx.export(enc, (x_in,), str(out / "encoder.onnx"), input_names=[in_name], output_names=cross_names,
                  dynamic_axes={in_name: {0: "batch"}, **{n: {0: "bc"} for n in cross_names}}, opset_version=17, dynamo=False)
# Trace the step with a 3-token past.
ids = torch.randint(0, 500, (B * C, 1))
past = tuple(torch.randn(B * C, 6, 3, 64) for _ in range(2 * L))
dyn = {"input_ids": {0: "bc"}, "logits": {0: "bc"}}
dyn.update({n: {0: "bc", 2: "past"} for n in self_names}); dyn.update({n: {0: "bc"} for n in cross_names}); dyn.update({n: {0: "bc", 2: "now"} for n in new_names})
torch.onnx.export(dec, (ids, *past, *cross), str(out / "decoder.onnx"), input_names=["input_ids", *self_names, *cross_names],
                  output_names=["logits", *new_names], dynamic_axes=dyn, opset_version=17, dynamo=False)
for f in ("encoder.onnx", "decoder.onnx"):
    print(f, (out / f).stat().st_size)

# Parity: ONNX Runtime against PyTorch, on another batch size and a real greedy decode.
import onnxruntime as ort
so = ort.SessionOptions()
es = ort.InferenceSession(str(out / "encoder.onnx"), so, providers=["CPUExecutionProvider"])
ds = ort.InferenceSession(str(out / "decoder.onnx"), so, providers=["CPUExecutionProvider"])
use_dense(False)
x_in = model.spectrogram(ref_audio) if SPEC_OUTSIDE else ref_audio
got_cross = es.run(None, {in_name: x_in.numpy()})
print("encoder max |diff|", max(float(np.abs(a.numpy() - b).max()) for a, b in zip(ref_cross, got_cross)), "max |value|", max(float(a.abs().max()) for a in ref_cross))
# Greedy decode both ways for 40 steps.
bc = 3 * C
self_t = [torch.zeros(bc, 6, 0, 64) for _ in range(2 * L)]; self_o = [z.numpy() for z in self_t]
ids_t = torch.zeros(bc, 1, dtype=torch.long); ids_o = ids_t.numpy()
worst, same = 0.0, True
for step in range(40):
    with torch.no_grad():
        r = dec(ids_t, *self_t, *ref_cross)
    feed = {"input_ids": ids_o, **dict(zip(self_names, self_o)), **dict(zip(cross_names, got_cross))}
    o = ds.run(None, feed)
    worst = max(worst, float(np.abs(r[0].numpy() - o[0]).max()))
    ids_t, self_t = r[0].argmax(-1), list(r[1:])
    ids_o, self_o = o[0].argmax(-1), o[1:]
    same &= bool((ids_t.numpy() == ids_o).all())
print("decoder logits max |diff| over 40 steps", worst, "tokens identical", same)

import hashlib
for f in ("encoder.onnx", "decoder.onnx"):
    print(f, (out / f).stat().st_size, hashlib.sha256((out / f).read_bytes()).hexdigest())
