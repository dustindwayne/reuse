# Grid average and exact offsets

Experiment requested by Dustin: lay out data in rows, take a rectangular block spanning adjacent rows, and represent its values by an average plus offsets. Implemented for unsigned bytes without interpreting file extensions.

For the example `01 11 | 02 20` / `00 22 | 33 21` (decimal), the left 2×2 block has integer mean 8 and offsets −7, +3, −8, +14. The right has mean 19 and offsets −17, +1, +14, +2. Both reconstruct exactly. Since the notation could mean hexadecimal, the benchmark includes both interpretations.

The mean is rounded down. Each signed offset is mapped to a nonnegative integer (0, −1, +1, −2, +2 map to 0, 1, 2, 3, 4), then packed using the minimum fixed bit width that fits the block. The stream contains a 24-byte header with original length, grid width and block dimensions. Each average block adds a mode byte, base byte and bit-width byte; raw blocks add a mode byte. The decoder handles partial rows and partial edge blocks without padding the original data.

Two encodings are tested: forced average for every block, and adaptive average/raw selection by actual serialized block size. Four geometries are tried for each larger sample; reported best sizes include all headers and block metadata. This is an offline search, not a learned selector. It does not merge duplicate blocks, use entropy coding, or claim optimal compression. Encoding and decoding are Python implementations; storage reduction alone does not establish speed or energy savings.

## Measured results

| Sample | Original bytes | Best adaptive grid bytes | zlib bytes |
|---|---:|---:|---:|
| User example (decimal or hex) | 8 | 34 | 16 |
| Constant bytes | 65,536 | 216 | 85 |
| Random values 120–135 | 65,536 | 41,176 | 37,878 |
| Smooth generated 256×256 grid | 65,536 | 17,176 | 441 |
| Random bytes | 65,536 | 65,624 | 65,562 |
| Synthetic text | 112,445 | 112,581 | 9,357 |

Exact roundtrip passed for all samples and shapes. Full measurements, timings, hashes and forced-average comparisons: [grid.json](../results/lossless/grid.json). 22 unit tests pass, including new grid tests for arbitrary bytes, extreme values, partial blocks, shapes, raw fallback and malformed streams.

The proposed transform saves space on clustered values. A numeric grid does not create similarity in arbitrary data; file byte order and block geometry determine whether offsets are small. zlib is smaller on every sample here. The tiny example grows because metadata outweighs payload savings. A future container can select raw storage for whole objects when every transformed representation is larger.

```bash
python scripts/benchmark_grid.py
python -m unittest discover -s tests -v
```
