# Third-Party Notices

`mbrila` is an independent PyTorch library. Several of its models are
**clean reimplementations** of algorithms first published as separate
research codebases. Copyright protects the expression of code, not the
underlying algorithms or mathematics; `mbrila` does not copy or import
any upstream source. Nonetheless, out of courtesy and to remove any
ambiguity, the upstream projects and their licenses are acknowledged
below. All upstream licenses are MIT and are compatible with `mbrila`'s
MIT license.

If you use the corresponding `mbrila` model, please also cite the
original work (see the Citation section of `README.md`).

---

## DLAG — Delayed Latents Across Groups

The `DLAG` model in `mbrila` reimplements the algorithm from the DLAG
project by Evren Gokcen.

Reference: Gokcen et al., Nature Computational Science (2022),
<https://doi.org/10.1038/s43588-022-00282-5>.

Original implementation (MATLAB):

```
MIT License

Copyright (c) 2022–2024 Evren Gokcen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## (fast-)mDLAG — multi-population DLAG with ARD

The `MDLAG` model in `mbrila` (both the time-domain and the
frequency-domain "fast" engine) reimplements the algorithm from the
fast-mDLAG project by Evren Gokcen.

References: Gokcen et al., NeurIPS 2023,
<https://nips.cc/virtual/2023/poster/70171>; and the frequency-domain
(fast-mDLAG) variant, Gokcen et al., Neural Computation (2025),
<https://doi.org/10.1162/neco.a.22>.

Original implementation (MATLAB):

```
MIT License

Copyright (c) 2024 Evren Gokcen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ADM — Adaptive Delay Model

The `ADM` model reimplements the Adaptive Delay Model. The reference
implementation (ICML 2025) is itself MIT-licensed and was authored by
the same group; it is listed here for completeness.
