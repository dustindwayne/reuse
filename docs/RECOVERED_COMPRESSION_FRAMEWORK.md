---
artifact: recovered_compression_framework
format: ai_ingestible_markdown
lossless_target: true
backend_codec: zlib
evidence_policy:
  confirmed: exact setup/result recovered from source conversation
  prototype_unverified: source reports a result but omits full reversible container overhead and/or independent artifact reproduction
  rejected: measured result did not beat matched zlib baseline
---

# Compression framework recovery

## 0. Source map

| source_id | title | role |
|---|---|---|
| `69cd2120-5cd0-8326-8779-beba116ff7e3` | Subdivision Lattice Compression | lane/subdivision/triplet/grid/anchor-walk experiments |
| `69ebb5f8-1218-83ea-8f0b-0dbf8f7929fc` | 120B Model Feasibility | adaptive fixed-width folding before zlib |
| `6ac7bc09-d054-83ea-98f2-5cdb80cbcda3` | Triplet Compression Results | later recovery pointer; benchmark bodies were references, not primary text |

`verification_scope = transcript_recovery`; no benchmark binaries/results artifacts were independently rerun here.

## 1. Canonical mechanism

```yaml
input: byte_stream B[0..n)
unit: chunk
candidates:
  - identity
  - fold(width=w)
  - experimental_triplet_demux
  - experimental_3_1_3_anchor_walk
backend: zlib(candidate_payload)
selection: argmin(total_encoded_bytes)
total_encoded_bytes: payload + mode + width + boundaries + inverse_transform_metadata
fallback: identity_plus_zlib
decode: zlib_decode -> inverse_selected_transform -> exact_original_bytes
```

### 1.1 Fold / lane transpose

For width `w`, reinterpret a chunk as consecutive `w`-byte records and concatenate equal byte positions:

```text
input:  a0 a1 ... a(w-1) | b0 b1 ... b(w-1) | ...
fold_w: a0 b0 ... | a1 b1 ... | ... | a(w-1) b(w-1) ...
lane_r = B[i] where i mod w = r
```

Properties: reversible; no byte values changed; exposes per-field/per-numeric-byte statistics to zlib.

Terminology equivalence:

```yaml
subdivision_lattice_depth_d: 2^d residue lanes
fold4: width=4; equivalent to radix-2 subdivision depth=2
fold8: width=8; equivalent to radix-2 subdivision depth=3
adaptive_folding: try multiple widths per chunk; retain only a size winner
```

Important: fixed whole-file SLC lane splitting and adaptive chunk folding use the same transpose primitive but are different selection policies.

## 2. Confirmed adaptive-fold benchmarks

Status for this section: `confirmed_in_source_conversation`; data was synthetic/model-like unless explicitly stated otherwise.

### 2.1 Fold8 initial simulation

```yaml
chunk_size: 16384
candidate_widths: [8, 16, 24, 32]
gate: high_entropy_chunks
backend: zlib
selection: smallest_candidate
winning_width: 8_on_most_winning_chunks
```

| input | raw_B | zlib_B | adaptive_fold_plus_zlib_B | delta_B | gain |
|---|---:|---:|---:|---:|---:|
| synthetic model weights | 989,615 | 939,091 | 918,466 | -20,625 | 2.196% |
| downloaded JPG | 1,784 | 1,713 | 1,714 | +1 | -0.058% |
| local ZIP | 3,544 | 2,135 | 2,136 | +1 | -0.047% |

Inference: fold8 aligned with structure in the synthetic model-like layout; already-compressed inputs correctly showed no gain. `fold_plus_byte_distance` did not win.

### 2.2 Fold4 expanded simulation

```yaml
chunk_size: 32768
candidate_widths: [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
entropy_gate_bits_per_byte: 7.15
external_model_download: failed
inputs: generated_model_like_binaries
```

| input | raw_B | zlib_B | adaptive_fold_plus_zlib_B | delta_B | gain | winner |
|---|---:|---:|---:|---:|---:|---|
| float32 transformer-like | 786,432 | 731,200 | 651,937 | -79,263 | 10.84% | fold4 on every gated chunk |
| mixed tensor-pack-like | 99,216 | 83,447 | 83,447 | 0 | 0% | identity |
| Q4 block-like | 240,000 | 207,534 | 207,534 | 0 | 0% | identity |

Inference: width matching the physical scalar width separated byte roles; fold4 grouped corresponding bytes of float32 values.

### 2.3 Broader retest with light mode metadata

```yaml
candidate_widths: [2, 4, 8, 16, 32, 64, 128]
comparison: baseline_zlib_vs_folded_plus_zlib
metadata_accounted:
  identity_chunk: 1_byte
  folded_chunk: 2_bytes_mode_plus_width
gate_revision: try_cheaply_then_keep_only_if_smaller
```

| generated data class | best gain | winning width | evidence read |
|---|---:|---:|---|
| fp32 normal | 4.77% | fold4 | positive |
| fp32 uniform | 5.18% | fold4 | positive |
| fp16 normal | 7.17% | fold2 | positive |
| bf16-like | 10.66% | fold2 | positive; average entropy reported near 6.2 bits/B |
| int8 quantized | 0.03% | mixed/noisy | negligible |
| Q4 block-like | 0.34% | mixed | weak |
| Q8 block-like | 0.18% | mixed | weak |
| random bytes | 0% | identity | correct rejection |

Confirmed conclusion:

```text
numeric/model-shaped bytes
-> test widths matching scalar layout
-> transpose byte lanes
-> zlib
-> retain fold only when total encoded size wins
```

Do not claim: real GGUF/safetensors validation; inference-memory reduction; universal high-entropy compression.

## 3. Subdivision-lattice and triplet results

### 3.1 Fixed SLC lane split: rejected on hostile media

```yaml
transform: lane_r = B[i] where i mod 2^depth = r
backend: zlib_per_lane
```

| input | zlib_B | depth1/fold2_B | depth2/fold4_B | depth3/fold8_B |
|---|---:|---:|---:|---:|
| JPG, raw 494,072 B | 493,777 | 493,837 | 493,907 | 494,049 |
| MP4, raw 9,840,497 B | 9,224,852 | 9,227,436 | 9,231,264 | 9,238,263 |

Status: `rejected_for_these_inputs`. Fixed global splitting lost cross-lane matches and paid multiple-stream overhead.

### 3.2 Lane split then triplet repeat/delta/literal: rejected

```yaml
triplet_codes: [repeat_previous, per_byte_delta_in_-2_to_2, literal]
pipeline: residue_lanes -> triplet_codes -> zlib_per_lane
```

| input | zlib_B | best tested triplet pipeline_B | result |
|---|---:|---:|---|
| JPG | 493,777 | 561,531 at depth1 | +67,754 B |
| MP4 | 9,224,852 | 10,548,770 at depth1 | +1,323,918 B |

Status: `rejected`; control tags/literals manufactured more metadata than redundancy.

### 3.3 Corrected triplet-position demux then recursive split: rejected

```text
Tj = (B[3j], B[3j+1], B[3j+2])
L0 = B[0],B[3],B[6],...
L1 = B[1],B[4],B[7],...
L2 = B[2],B[5],B[8],...
then split each Lk by every-other residue; zlib resulting streams
```

| JPG method | bytes | delta_vs_zlib_B |
|---|---:|---:|
| zlib baseline | 493,777 | 0 |
| triplet demux depth0 | 493,919 | +142 |
| triplet demux depth1 | 494,002 | +225 |
| triplet demux depth2 | 494,220 | +443 |
| triplet demux depth3 | 494,440 | +663 |

Status: `rejected_for_tested_JPG`.

## 4. Unverified positive triplet-grid result

This is the likely source of the recollection “slightly better than zlib.”

```yaml
input: JPG_494072_B
layout_period: 7
layout: [3_bytes_to_L1, 1_byte_to_L2, 3_bytes_to_L3]
layer_sizes_B: [211746, 70582, 211744]
transform_per_layer: anchor_walk
backend: zlib
status: prototype_unverified
```

| pipeline | bytes | delta_vs_plain_zlib_B |
|---|---:|---:|
| plain zlib | 493,777 | 0 |
| 3/1/3 concatenate, no walk | 493,947 | +170 |
| 3/1/3 -> anchor-walk each -> zlib each | 493,608 | **-169** |
| 3/1/3 -> anchor-walk each -> concatenate -> one zlib | 493,628 | **-149** |

Reported anchors: `L1=0x49`, `L2=0x4A`, `L3=0x4A`.

Why not confirmed as a codec win:

```yaml
missing_from_reported_total:
  - complete layering-mode metadata
  - chosen-anchor metadata per layer
  - complete reversible anchor-walk descriptors
  - final container framing/checksums
  - independent rerun or retained benchmark artifact
margin: 169_bytes_max
risk: omitted metadata can erase the full gain
```

Therefore:

```text
confirmed claim: prototype transformed payload was 169 B below the conversation's zlib control
unconfirmed claim: a complete reversible triplet-grid codec beats zlib
```

Related but separate prototype: `3/1/3 center-relative delta streams + per-stream anchor-walk + zlib = 493,704 B`, 73 B below baseline; it originated from a misinterpretation of the requested high/low scheme and is not the primary recovered result.

## 5. Recovered framework decision rule

```pseudo
for chunk in input:
    candidates = [{mode: identity, bytes: chunk}]

    for w in [2, 4, 8, 16, 32, 64]:
        if len(chunk) supports w:
            candidates += fold_transpose(chunk, w)

    # Research-only until full overhead is proven:
    candidates += optional_triplet_demux(chunk)
    candidates += optional_3_1_3_anchor_walk(chunk)

    for c in candidates:
        c.payload = zlib(c.bytes)
        c.cost = len(c.payload) + exact_inverse_metadata(c)

    emit min(candidates, key=cost)
```

Selection must use exact final bytes, not entropy alone. Entropy is a search hint only. Random/already-compressed/quantized regions should select identity when transforms do not pay for themselves.

## 6. Evidence boundary

```yaml
confirmed_positive:
  - adaptive fold8 on generated model-like data: 2.196%, 20,625 B
  - adaptive fold4 on generated float32-like data: 10.84%, 79,263 B
  - fold4 broader generated retest: 4.77% to 5.18%
  - fold2 broader generated retest: 7.17% to 10.66%
confirmed_negative:
  - fixed SLC fold2/fold4/fold8 lost on supplied JPG and MP4
  - triplet repeat/delta/literal lost heavily
  - corrected triplet-position demux lost at all tested depths
prototype_only:
  - 3/1/3 lanes + per-layer anchor-walk reported 169 B payload win
not_established:
  - complete triplet-grid container win after all metadata
  - real model-file benchmark win
  - general-purpose superiority to zlib
```
