"""Generate assessment directly from completed measurements."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
r=json.loads((root/'results/lossless/corpora.json').read_text());w=json.loads((root/'results/lossless/corpora_whole.json').read_text())
assert len(r['files'])==len(w['files'])==25 and len(r['archives'])==8 and len(w['matched_archives'])==2
lines=['# Real-corpus evaluation','',
'Tested the complete 11-file Canterbury corpus and standard 14-file Calgary corpus: 25 files, 5,952,406 bytes. These contain real text, source, markup, spreadsheet, executable, image and geophysical data. No samples were truncated. Inputs match the pinned mirror Git blobs and its published SHA1 checksums; SHA-256 values are recorded in [corpus_sources.json](../results/lossless/corpus_sources.json). Raw corpus files are not redistributed in this repository.', '',
'Sources: [official descriptions](https://corpus.canterbury.ac.nz/descriptions/) and [pinned mirror](https://github.com/pfalcon/canterbury-corpus/tree/master). The mirror tree is `c380ef9dc1dc62691757fdfeebdc2f85710e1e46`; the manifest pins each blob rather than depending on the mutable branch.', '',
'## Whole-file transforms followed by zlib','',
'Each complete file compares raw, zlib, dynamic windows, shared grammar (32 merges), grid, and each transform followed by zlib. Size selection counts internal transform metadata. These payload totals omit common archive framing, so they isolate composition rather than claiming complete-archive superiority. One encode/decode timing per whole file; exact byte equality is asserted.', '',
'| Corpus | Direct zlib payload bytes | Selected payload bytes | Reduction | Improved files |','|---|---:|---:|---:|---:|']
for c in ('canterbury','calgary'):
 files=[f for f in w['files'] if f['path'].startswith(c+'/')];a=sum(f['direct_zlib_bytes'] for f in files);b=sum(f['selected_payload_bytes'] for f in files);wins=sum(f['selected_payload_bytes']<f['direct_zlib_bytes'] for f in files)
 lines.append(f'| {c} | {a:,} | {b:,} | {(a-b)/a:.2%} | {wins}/{len(files)} |')
lines+=['','## Complete archives and matched controls','','Framework archives include all metadata, checksums and references. The matched control has the same container, sharing and chunk size but only raw/zlib candidates. ZIP uses per-file DEFLATE. This isolates transforms from sharing and chunking. Archive framework timings have one repeat at each chunk size; ZIP and the 4 KB matched control have three. The 32 KB matched control has one.','',
'| Corpus | Framework 4 KB | Matched 4 KB | Framework 32 KB | Matched 32 KB | ZIP |','|---|---:|---:|---:|---:|---:|']
for c in ('canterbury','calgary'):
 arcs=[x for x in r['archives'] if x['corpus']==c]
 def find(method,size=None):return next(x['bytes'] for x in arcs if x['method']==method and (size is None or x.get('chunk_size')==size))
 matched=next(x['bytes'] for x in w['matched_archives'] if x['corpus']==c)
 lines.append(f"| {c} | {find('framework',4096):,} | {find('matched_zlib',4096):,} | {find('framework',32768):,} | {matched:,} | {find('zip'):,} |")
lines+=['','## Every file, including standard baselines','','Per-file framework and matched-control sizes include a one-file archive. zlib/gzip/bzip2/xz sizes include their own stream headers but no filename/manifest. Per-file times are medians of three serial runs; all compressed bytes are deterministic and all methods pass exact reconstruction. Defaults: zlib level 6, gzip level 9, bzip2 level 9, xz preset 6, ZIP DEFLATE default.','',
'| File | Raw | Framework 4 KB | Matched 4 KB | zlib | gzip | bzip2 | xz | Whole-file selected mode | Whole-file selected bytes |','|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
modes={0:'raw',1:'zlib',2:'dynamic',3:'grammar',4:'grid',5:'dynamic → zlib',6:'grammar → zlib',7:'grid → zlib'}
for f in sorted(r['files'],key=lambda x:(x['corpus'],x['name'])):
 key=f['corpus']+'/'+f['name'];whole=next(x for x in w['files'] if x['path']==key);m=f['methods'];vals=[f['input_bytes']]+[m[k]['bytes'] for k in ('framework_4k','matched_zlib_4k','zlib','gzip','bzip2','xz')]
 lines.append('| '+key+' | '+' | '.join(f'{v:,}' for v in vals)+' | '+modes[whole['selected_mode']]+' | '+f"{whole['selected_payload_bytes']:,}"+' |')
lines+=['','## Compute cost','','| Corpus | Sum of per-file framework encode medians | Sum of matched encode medians | Framework decode medians | Matched decode medians |','|---|---:|---:|---:|---:|']
for c in ('canterbury','calgary'):
 fs=[f for f in r['files'] if f['corpus']==c]
 vals=[sum(f['methods'][method][field] for f in fs) for method,field in [('framework_4k','encode_seconds'),('matched_zlib_4k','encode_seconds'),('framework_4k','decode_seconds'),('matched_zlib_4k','decode_seconds')]]
 lines.append('| '+c+' | '+' | '.join(f'{v:.4f} s' for v in vals)+' |')
lines+=['','## What the evidence establishes','',
'Transform composition can improve direct zlib on real complete files. The current complete archive still loses to ZIP on both corpora. The 4 KB framework is especially constrained by lost compression context and metadata. Larger chunks improve archive size and allow more structural candidates to win. These measurements do not show a general compression breakthrough or lower power consumption: the Python encoder searches multiple candidates and is much slower than direct zlib. Whole-file compression and shared storage have different access/sharing tradeoffs.', '',
'Corpus parameters were chosen before these results, not tuned per file. Standard corpora are small, old benchmarks, not a substitute for modern multimedia, encrypted data, application access or multi-version storage tests. Corpus sizes are storage bytes, not linguistic token counts. The framework preserves exact bytes, not a learned semantic model.', '',
'Process peak RSS in the main runner is a cumulative high-water mark, not per-codec memory. Timings are serial measurements on this host, without confidence intervals; whole-file and archive sensitivity timing has only one repeat. Tests include ordinary natural text and binary data, but no claim of universal improvement follows.', '',
'29 unit tests pass, including exact reconstruction, shared references, malformed streams and the matched-control selector.','',
'Full records: [per-file/archive tests](../results/lossless/corpora.json), [whole-file tests](../results/lossless/corpora_whole.json). Environment and repeated timings are in those files.','',
'## Reproduce','','```bash','python scripts/download_corpora.py /tmp/corpus-bench','python scripts/benchmark_corpora.py --corpus-root /tmp/corpus-bench --output results/lossless/corpora.json','python scripts/benchmark_corpora_whole.py /tmp/corpus-bench results/lossless/corpora_whole.json','python scripts/report_corpora.py','python -m unittest discover -s tests -v','```','',
'The main runner checkpoints after each file. Delete its output to rerun all timings; otherwise it resumes files whose input hashes and algorithm-source hashes match. Dataset download uses public GitHub blob endpoints and verifies pinned SHA-256 values.']
(root/'docs/CORPUS_EVALUATION.md').write_text('\n'.join(lines)+'\n')
