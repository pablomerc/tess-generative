# Suggested response — reviewer JcH2, Q1 ("what does attention-based conditioning mean?")

Draft for Pablo to merge into the official response. **Not** an official file.

The reviewer quotes the sentence at `disentanglement_neurips2026.tex:207`:

> "Note that the number of source and instrument neighbors may vary; therefore, we
> combine the variable sequence of embeddings via attention-based conditioning."

---

## Fact-check of the original draft answer

| claim | verdict |
|---|---|
| "We treat the output of an encoder as a token of dimension 64" | **WRONG — contradicts our own paper.** Each encoder maps one image to **4 tokens of 16 dimensions**, i.e. `(B, 4, 16)`. That is stated in §4 (tex:359) and Table 1 (tex:778-780, `Conv2d(512→16, kernel=1)`, cross-attention dim 16), and it is what the code does (`double_train_fm_neighbors.py:105` projection; `:124-127` reshape to `(B, H'W', D)`; `configs/neighbors_efficient.json:30` `cross_attention_dim: 16`). 64 is the *flattened* width 4×16, which is what the **downstream probes** consume (tex:1011) — not the cross-attention token layout. Saying "a token of dimension 64" to a reviewer who has read the appendix suggests we don't know our own architecture. |
| "we concatenate 1 physics-latent token with one instrument-latent token" | **WRONG for the same reason** — off by 4× on both sides. It is 4 physics tokens + 4 tokens *per neighbour*: `(B, (1+k)·4, 16)` (`:392` flatten, `:394` concat). With k=5 that is **24 tokens of dimension 16** = 4 + 20. |
| *combined risk of (a)+(b)* | **This is the sharpest danger.** "1 token of 64" and "4 tokens of 16" have the *same flattened width*, so a reviewer will infer we mentally flattened the token axis — and may then ask whether the reported model is actually the `instrument_flatten_to_one_token` ablation, which really does collapse tokens to one (`:193`, `:356-366`). It exists in the code but is **off** in the baseline config, and applies only to the instrument branch. The draft would accidentally advertise an ablation the paper does not report, and invite a question about which model Table 5 came from. |
| "The decoder uses crossattention … when modifying the activations of the UNet" | **Right idea, minor wording fix.** "Decoder" is fine — it is the paper's own term ("flow-matching generative decoder", tex:754; also tex:240, tex:512). The only issue is that the sentence reads as if the decoder and the UNet were two modules; they are the same `UNet2DConditionModel`. Worth adding *which way* the attention runs: UNet activations are the queries, the encoder tokens the keys/values. |
| "automatically accommodates a variable number of spatial neighbors (variable sequence length)" | **Defensible, but say it precisely.** Two facts in our favour: cross-attention is length-agnostic and permutation-equivariant, and `collate_neighbors` pads to the **per-batch maximum**, not to a global constant (`neighbors.py:162` `max_n = max(t.size(0) for t in neighbor_tensors)`), so the sequence length genuinely does vary *across* batches. The caveat: *within* a batch it is fixed, absent neighbours are **zeroed** rather than dropped (`:388-390`), and **no `encoder_attention_mask` is passed** to the velocity net (verified: no attention-mask argument anywhere in the file). Since diffusers' `to_k`/`to_v` are bias-free and the conditioning gets no norm, a padded token has key ≡ 0 → logit 0 → a nonzero softmax weight, so it inflates the denominator and slightly down-weights the real tokens. Padding is therefore not a semantic no-op. Don't claim masking we don't do. |

Also worth knowing (do **not** volunteer these, but be ready): padded zero stamps are still pushed through `encoder_2` (`:353-354`), so they enter BatchNorm batch statistics in train mode; and the paper's only mention of padding is **commented out** (tex:821, `% Max neighbors per sample & Variable (padded)`) — uncommenting that table row is a cheap revision that documents the behaviour.

Mitigating fact worth having ready: padding is almost never exercised here. Fewer than
**0.05% of HSC and ~0.3% of Legacy anchors** have fewer than five sky neighbours, so at
batch size 64 essentially every batch runs the full K=5, and the zero-padding path is a
rare edge case rather than a routine part of training.

---

## Important framing point

**The paper already answers this, in two places the reviewer appears not to have
connected to the sentence they quote.** §4 "Model Implementation" (tex:359) says:

> "We map each input image to a sequence of 4 spatial tokens of 16 dimensions, i.e. a
> latent representation of shape $(B,4,16)$. To accommodate a variable number of
> conditioning examples, we treat these embeddings as a set of tokens. For a given
> input set, the tokens from all images are concatenated into a single sequence of
> variable length. Then, we use cross-attention layers within our velocity network, a
> `UNet2DConditionModel`."

and Appendix A.1 (tex:754) repeats it with the projection detail. The quoted sentence
sits in §3 (tex:207), which is ~150 lines earlier and carries **no forward reference**
to either. So the fix in the revision is concrete and cheap: add a pointer at that
sentence. Lead the response with that — it shows the description exists rather than
implying an omission, and it gives the reviewer something actionable.

## Suggested response

> Thank you — this is described in §4 ("Model Implementation") and Appendix A.1, but
> the sentence you quote is in §3 and does not point forward to them; we will add that
> cross-reference. Concretely:
>
> **Token layout.** Each encoder maps one 48×48×4 image to a *sequence* of tokens
> rather than a single vector: the ResNet-18 trunk's final feature map is projected by
> a 1×1 convolution to 16 channels over a 2×2 grid, giving **4 spatial tokens of
> dimension 16** per image (Appendix A.1, Table 1).
>
> **Building the conditioning sequence.** For a target observation $x_{j,k}$ we encode
> (i) the same source seen by the other instrument with the physics encoder, giving 4
> tokens, and (ii) each of the $K$ same-instrument neighbours with the instrument
> encoder, giving $4K$ tokens. These are concatenated along the sequence axis into a
> single set of $4(1+K)$ tokens of dimension 16 — with $K=5$, 24 tokens.
>
> **How it conditions the network.** That sequence is supplied as the key/value
> memory of the cross-attention layers of the velocity network $u_\theta$ (a
> `UNet2DConditionModel`); the UNet's spatial activations are the queries. Attention
> is applied at every resolution, in all four down-blocks and all four up-blocks, so
> each location in the noisy image attends to whichever conditioning tokens are
> relevant to it.
>
> **Why this is the mechanism that handles a variable number of neighbours.** Attention
> is agnostic to sequence length and permutation-equivariant over the memory, so the
> *same* parameters accept any number of conditioning views with no architectural
> change and no ordering convention — unlike conditioning schemes that concatenate a
> fixed number of embeddings along the channel or feature axis, which fix the number of
> views at design time. This is what makes the general formulation of Eq. (Sec. 3)
> usable: the physics set $\{z_{\text{phy}}(x_{j,k'})\}_{k'\neq k}$ and the instrument
> set $\{z_{\text{ins}}(x_{j',k})\}_{j'\neq j}$ can each have any cardinality. In the
> present two-survey instantiation the physics set has one element (the other survey)
> and the instrument set has up to five sky neighbours, but the same trained module
> would accept more instruments per source, which is the direction we discuss for
> extending beyond two surveys.
>
> **Implementation detail, for completeness.** Within a batch we pad the neighbour axis
> to a fixed $K$ and zero the embeddings of absent neighbours. In this dataset that
> path is rarely exercised — 100% of HSC anchors and 99.6% of Legacy anchors have at
> least five neighbours within the 3-arcminute radius — but for generality we will note
> in the revision that masking the padded positions in the attention (rather than only
> zeroing their embeddings) is the cleaner treatment when neighbour counts vary widely.
>
> We will add this description, and a figure of the conditioning sequence, to
> Appendix A.1.

---

## Notes on choices in the above

- **The last paragraph is optional.** It concedes a small implementation
  imperfection. Include it if you would rather pre-empt the question than be asked it
  — reviewers generally reward this, and the ≥5-neighbour statistic makes it
  harmless. Drop it if you want the answer shorter, but then be ready for the
  follow-up.
- **"Attention-based conditioning" is doing two jobs** in the paper: variable
  cardinality (what the reviewer asked) and spatially-resolved conditioning (each
  UNet location attends where it needs to). The answer covers both, since the second
  is why the tokens are spatial rather than pooled.
- **"Decoder" is fine** — it is the paper's own term (tex:754, 240, 512) and the
  reviewer uses it too. Just don't phrase it as if the decoder and the UNet were
  separate modules.
- **Lead with the cross-reference, not with a re-explanation.** The content already
  exists at tex:359 and tex:754; the real defect is the missing forward pointer at
  tex:207. Saying so converts a "you didn't explain this" into a one-line revision.
- **Avoid the phrase "spatial neighbours"** (in the original draft). "Spatial" is
  already doing duty for the *token grid* in this paper; using it for sky-adjacent
  neighbours in the same sentence as "spatial tokens" is exactly the collision that
  produced the confusion. Say "sky neighbours" or "same-instrument neighbours".
- The claim that all blocks attend comes from `all_attention: true`
  (`configs/neighbors_efficient.json:41`) and the block-type list at
  `double_train_fm_neighbors.py:250-262`.
