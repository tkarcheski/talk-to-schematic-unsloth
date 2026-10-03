# Vendored browser libraries

| File | Source | Version | License | SHA256 |
|---|---|---|---|---|
| `pdf.min.mjs` | [pdfjs-dist](https://www.npmjs.com/package/pdfjs-dist) `build/pdf.min.mjs` | 6.4.299 | Apache-2.0 (`pdfjs-LICENSE`) | `57456c8e0c81e46be31174b499ef77f2b9f5ee46d04412ba627320a36755d4c2` |
| `pdf.worker.min.mjs` | pdfjs-dist `build/pdf.worker.min.mjs` | 6.4.299 | Apache-2.0 (`pdfjs-LICENSE`) | `9536359f1b8367850d485731ca1d5e45c159a7b7a0912325e539937aa21ceb18` |

The files are unmodified copies from the npm tarball. They are served from the
local server so uploaded PDFs never leave the machine and the page keeps its
`script-src 'self'` policy. `tests/test_web_ui.py` checks these hashes.
Standard font, CMap and WebAssembly image-decoder assets are not vendored, so
PDFs that depend on them may render with substitute fonts or missing raster
images; vector schematic drawings are unaffected.
