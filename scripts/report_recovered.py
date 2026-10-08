import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];r=json.loads((root/'results/lossless/recovered_folding.json').read_text());a=json.loads((root/'results/lossless/recovered_archives.json').read_text());f=json.loads((root/'results/lossless/recovered_fileaware.json').read_text())
assert len(r)==35 and len(a)==8 and len(f)==2
lines=['# Recovered transforms: integration and tests','',
'The user-supplied [recovery note](RECOVERED_COMPRESSION_FRAMEWORK.md) contributes a reversible lane transpose, adaptive width selection with exact metadata, and a 3/1/3 permutation. Its transcript-recovered historical results are attributed reports, not independently reproduced artifacts here. The undefined anchor-walk algorithm cannot be reconstructed from the note, so no anchor-walk implementation or reproduction of its reported 169-byte gain is claimed. Rejected repeat/delta/literal pipelines were not added.', '',
'## Implemented','',
'Fold widths 2, 4, 8, 16, 32, 64 and 128 group byte positions by their index modulo width; unequal final lanes reconstruct exactly. Codec 8 stores one width byte followed by zlib output. Codec 9 is a reversible 3/1/3 permutation followed by one zlib stream, without anchor-walk. Layout is fixed by its mode and tail lengths are derived from decompressed length. Both compete with all prior codecs by actual payload bytes. The common outer mode field remains the same size.', '',
'The archive selector now compares complete pooled encoding, pattern-first pooled encoding and file-boundary-aware encoding. The latter retains different per-file layouts before applying an outer zlib stage. It chooses actual complete archive bytes, including all references, names, lengths, checksums and framing. This is a bounded three-path search, not globally optimal representation or semantic boundary inference.', '',
'## Real whole-file payloads','',
'| Input | Prior framework | Added transforms | Selected new mode | Additional reduction |','|---|---:|---:|---|---:|']
for x in r:
 if x['kind']!='synthetic_control' and x['new_framework_bytes']<x['prior_framework_bytes']:
  label='fold'+str(x['selected_width'])+' → zlib' if x['selected_mode']==8 else '3/1/3 → zlib'
  lines.append(f"| {x['name']} | {x['prior_framework_bytes']:,} | {x['new_framework_bytes']:,} | {label} | {1-x['new_framework_bytes']/x['prior_framework_bytes']:.2%} |")
lines+=['','Across the 25 real corpus files, the earlier selector totals 1,708,148 payload bytes; the new selector totals 1,667,959. Common archive framing is excluded from these payload totals and included in the complete-archive comparison below. Only two corpus files receive additional savings; the earlier improvements on other files remain intact.','',
'Five small, public TensorFlow Lite files were retrieved byte-exactly from the pinned TensorFlow repository. Two audio-preprocessing graph files gain 104 bytes combined. The tiny hello-world models and quantized speech model do not improve further. These files are real models/processing graphs, not a benchmark of large learned weight tensors. Source names, blob hashes, sizes and SHA-256 values: [fold_model_sources.json](../results/lossless/fold_model_sources.json). Raw model files are not redistributed.', '',
'## Complete archive selection','',
'| Corpus | Prior complete path | File-boundary-aware path | Selected complete bytes | Improvement |','|---|---:|---:|---:|---:|']
for c in ('canterbury','calgary'):
 prior=next(x['archive_bytes'] for x in a if x['corpus']==c and not x['matching'] and not x['folding']);new=next(x['archive_bytes'] for x in f if x['corpus']==c);best=min([x['archive_bytes'] for x in a if x['corpus']==c and x['folding']]+[new])
 lines.append(f'| {c} | {prior:,} | {new:,} | {best:,} | {1-best/prior:.2%} |')
lines+=['','Folding the single pooled literal stream provided no added archive saving on these corpora. Preserving file boundaries lets Calgary retain numeric alignment and choose different transforms for text and binary files. Canterbury retains its earlier pooled path. Pattern-first matching remains useful for shifted/edited copies but does not win these two corpus archive selections.','',
'## Synthetic numeric controls','',
'These newly generated controls test the recovered mechanism; they are not reproductions of the note’s original datasets.','',
'| Input | Prior framework | New framework | Additional reduction | Width |','|---|---:|---:|---:|---:|']
for x in r:
 if x['kind']=='synthetic_control':lines.append(f"| {x['name']} | {x['prior_framework_bytes']:,} | {x['new_framework_bytes']:,} | {1-x['new_framework_bytes']/x['prior_framework_bytes']:.2%} | {x['selected_width'] or 'none'} |")
lines+=['','All 35 inputs and all archive variants reconstruct exactly. 37 unit tests pass. Trials have one timing repeat; results establish size, not stable throughput or reduced energy. Trying three archive paths and multiple transforms remains costly. Decoder compatibility includes the existing formats and the new FAW1 file-aware wrapper; older readers cannot understand the new modes.','',
'Full measurements: [per-input candidates](../results/lossless/recovered_folding.json), [pooled ablations](../results/lossless/recovered_archives.json), [file-aware archives](../results/lossless/recovered_fileaware.json).','',
'## Run','','```bash','python scripts/container.py pack --auto example.udf archive/language-model/data/train.txt archive/language-model/data/validation.txt','python scripts/container.py unpack example.udf restored','python scripts/download_corpora.py /tmp/corpus-bench','python scripts/download_fold_models.py /tmp/corpus-bench/models','python scripts/benchmark_recovered.py /tmp/corpus-bench','python scripts/benchmark_recovered_archives.py /tmp/corpus-bench','python scripts/benchmark_recovered_fileaware.py /tmp/corpus-bench','python scripts/report_recovered.py','python -m unittest discover -s tests -v','```','']
(root/'docs/RECOVERED_INTEGRATION.md').write_text('\n'.join(lines))
