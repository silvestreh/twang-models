# Twang models

The one place Twang downloads its machine-learning models from: one GitHub release of this
repository holds every model, its files as release assets (never committed).
Twang pins every file's size and SHA-256 and checks them after downloading.

Samples (the Sound Library) are not here: they download from their original sources.

| Model | Used for | Files | Size | Original source | Licence |
|---|---|---|---|---|---|
| YourMT3+ | Convert Harmony / Melody / Drums to MIDI | `encoder.onnx`, `decoder.onnx` | 190 MB | YourMT3+ "YPTF.MoE+Multi (noPS)", exported to ONNX by `export/yourmt3_to_onnx.py` | Apache-2.0 (see below) |
| Demucs | Separate stems | four `*.th` checkpoints | 336 MB | Demucs `htdemucs_ft`, Meta, `dl.fbaipublicfiles.com/demucs/hybrid_transformer` | MIT |
| CLAP | Auto Tag, similar sounds | `larger_clap_music_and_speech-audio.safetensors` | 275 MB | `laion/larger_clap_music_and_speech` at revision `8a2acdbf`: the first 275,119,376 bytes of `model.safetensors` (header and audio tower) | Apache-2.0 |

All are in release `v1`: `https://github.com/silvestreh/twang-models/releases/download/v1/<file>`.
`MANIFEST.json` lists every file with its size and SHA-256.

## Credits and licences

- **YourMT3+**: Sungkyun Chang, Emmanouil Benetos, Holger Kirchhoff, Simon Dixon, "YourMT3+:
  Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset
  Stem Augmentation", 2024. Code and checkpoint from the authors' Hugging Face Space
  (`mimbres/YourMT3`), whose files carry Apache-2.0 headers and which is tagged apache-2.0. The
  authors' GitHub repository (`mimbres/YourMT3`) carries a GPL-3.0 licence file; the weights have
  no separate licence. The ONNX graphs here are a format conversion of checkpoint
  `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt`
  (SHA-256 `ae38e415…fc25`), unchanged in their weights.
- **Demucs** (Hybrid Transformer Demucs, `htdemucs_ft`): Simon Rouard, Francisco Massa, Alexandre
  Défossez, Meta. MIT licence, `github.com/facebookresearch/demucs`. Files mirrored unchanged.
- **CLAP** (`larger_clap_music_and_speech`): LAION. Apache-2.0,
  `huggingface.co/laion/larger_clap_music_and_speech`. The leading part of the file, unchanged.

## Adding or updating a model

1. Never replace an asset under an existing release. Make the next release (`v2`, …) with every
   model's files: the unchanged ones uploaded again, the new or updated ones beside them.
2. Update `MANIFEST.json` and the table above.
3. Point Twang's stores at `https://github.com/silvestreh/twang-models/releases/download/v<N>/<file>`
   with the sizes and SHA-256 pinned.
