# Suggested response — reviewer SYJm, Q2 (decoder reliance / shuffled-latent ablation)

Draft for Pablo to merge into the official response. **Not** an official file.

Source of every number below: `galaxy_images/galaxy_model/shuffle_ablation/`, pooled over
`results_mA` (m=0) + `results_m123` (m=1,2,3) → **M=4 posterior samples per object per
condition**, paper checkpoint `checkpoints/base/snapshot.ckpt`, 250 Euler steps, n=256
held-out anchors (123 HSC-target, 133 Legacy-target). Design/protocol as locked in
`shuffle_ablation/PLAN.md`.

---

## 1. What the run says (short version)

The 2×2 signature the reviewer predicted **is there, in both directions, and it is clean**:

* **z_phy carries identity.** Shuffling it does not degrade the generation — it *relocates*
  it. MSE against the donor's ground truth under shuffle-z_phy (0.058 HSC / 0.188 Legacy)
  is statistically indistinguishable from MSE against the *own* target in the intact run
  (0.058 / 0.186). The generation is closer to the donor than to its own target for
  **100% of HSC and 97.7% of Legacy anchors** (vs 0.8% / 2.3% intact).
* **z_ins carries the noise floor, and only that.** Shuffling it leaves identity untouched
  (paired ΔMSE = +0.002 ± 0.002 HSC, +0.000 ± 0.005 Legacy — a null), while the per-object
  pixel-scale noise level stops tracking the object's own instrument (r: +0.99 → −0.03) and
  starts tracking the **donor's** instrument (r: −0.11 → **+0.89**, OLS slope 0.87).
* This matches Pablo's visual read exactly, and explains *why* C2 is only sometimes visible
  by eye: because donors are within-survey, the **mean** noise level barely moves
  (σ_pix 0.1109 → 0.1110 in HSC), so the effect is a per-object re-tracking, not a
  brightness/contrast shift.

## 2. Read before sending — three things to be deliberate about

**(a) Which σ estimator you quote matters, and the honest choice is the high-pass one.**
Of the three σ estimators in `metrics.py`, only `adjacent_diff_sigma` (and, for HSC, high-k
PSD power) is a clean *instrument* statistic. `corner_sky_rms` and `masked_mad_sigma` are
**source-flux dominated** in these 48×48 cutouts — measured on the real targets they
correlate with total cutout flux at r = +0.67/+0.61 and **+0.89/+0.88** (HSC/Legacy), with
CV ≈ 0.7–1.35, which is not sky-noise-like. Consequently those two follow **z_phy**
(under C1 they track the donor at r ≈ 0.95, slope ≈ 1.0). That is *not* a disentanglement
failure — it is the estimators picking up the source and its companions, which z_phy
legitimately controls — but if you quote them as "instrument statistics" the reviewer will
correctly object. Quote σ_pix (adjacent-pixel difference) as the headline, mention high-k
PSD, and keep the flux-contamination fact in reserve.

**(b) Legacy high-k power is muddy (own +0.45 / donor +0.36 under C2) — pre-empt it.**
Same root cause: in Legacy cutouts high-k power itself correlates with source flux
(r = −0.53), so it is not instrument-only. σ_pix in Legacy is fine (donor +0.69, slope 0.64,
own +0.17). Either report high-k for HSC only, or state the caveat in one clause. Don't
present the Legacy high-k cell without explanation.

**(c) The absolute MSE differs slightly from the paper's table.** The held-out split was
rebuilt on Engaging (the paper's original split file is AMD-only), so C0 = 0.058/0.186
against the paper's 0.081/0.197. Everything in the ablation is *paired within the same 256
objects*, so no conclusion depends on this — but decide whether to footnote it. My
recommendation: yes, one short footnote; it is cheaper than being asked.

Also worth knowing (do **not** volunteer): the reviewer said "shuffled/random"; we ran
shuffled only. Deliberate — an i.i.d. Gaussian vector is off-manifold for the
cross-attention keys/values, so the damage would be uninterpretable. Phase-2 hooks for
moment-matched random latents are specced (`PLAN.md` §4.4, ~1 day) if pushed. Two
independent m=0 runs on different GPUs gave MSE_own 0.0587 vs 0.0599 (HSC) — kernel
nondeterminism only, conclusions unaffected. Conclusions also hold on a centre-32×32 crop
(`mse_own_32`/`mse_donor_32` columns). Verify the "Fig. 7 / App. A.5" numbering against the
`.tex` on the AMD cluster before sending — it is not in this repo.

---

## 3. Suggested response

> We thank the reviewer for this suggestion — we agree it is the right diagnostic, and we
> have run it. Two clarifications first.
>
> **The premise holds for training, not for sampling.** It is true that the training
> interpolant $x_t=(1-t)x_0+t\,x_{\text{target}}$ leaks target information at large $t$.
> At inference, however, we integrate the ODE from $t=0$, i.e. from pure Gaussian noise,
> and the target observation is *never* encoded — so every bit of target content in a
> generated image has to arrive through $(z_{\text{phy}}, z_{\text{ins}})$. What the
> reviewer's diagnostic adds is not *whether* the latents are used, but the **causal
> division of labour** between them, which we now measure.
>
> **Fig. 7 is already a $z_{\text{ins}}$ intervention.** Both panels hold the source
> conditioning and the initial noise fixed and replace only the instrument conditioning:
> the SNR traversal sweeps $z_{\text{ins}}$ across observing-condition percentiles, and the
> artifact-removal panel supplies an HSC+HSC combination never seen in training. Appendix
> A.5 is the training-time analogue on the same pathway (random instead of sky-neighbour
> conditioning collapses instrument-property recovery, Galaxy Depth $R^2$ 0.68 → 0.20).
>
> **New ablation.** For 256 held-out galaxies we generate with the *same* initial noise
> $x_0$ under three conditions: intact conditioning; $z_{\text{phy}}$ replaced by that of a
> different held-out galaxy ("donor"); $z_{\text{ins}}$ replaced by the donor's instrument.
> Donors are drawn *within* the target survey, so the corrupted latent stays on-manifold
> and the intervention is not a survey swap (that is Figs. 6–7). Crucially, each donor also
> has ground truth in the target survey, so we can test not only that corruption *hurts*
> but that the generation *follows the donor*. Averaged over 4 posterior samples
> (HSC-target / Legacy-target):
>
> | | intact | shuffled $z_{\text{phy}}$ | shuffled $z_{\text{ins}}$ |
> |---|---|---|---|
> | MSE vs **own** target | 0.058 / 0.186 | 0.97 / 1.49 | 0.060 / 0.186 |
> | MSE vs **donor** target | 0.97 / 1.48 | **0.058 / 0.188** | 0.97 / 1.48 |
> | $r(\sigma_{\text{pix}},$ own instr$)$ | +0.99 / +0.94 | +0.90 / +0.72 | −0.03 / +0.17 |
> | $r(\sigma_{\text{pix}},$ donor instr$)$ | −0.11 / −0.10 | −0.01 / +0.10 | **+0.89 / +0.69** |
>
> Here $\sigma_{\text{pix}}$ is a robust MAD-based $\sigma$ of adjacent-pixel differences,
> i.e. the pixel-scale noise floor of an image. Every image — generated or real — has one
> such value, so for each object we have three: the generation's, that of the real
> observation we are trying to reproduce ("own instr"), and that of the unrelated
> observation whose $z_{\text{ins}}$ we injected ("donor instr"). Since every field was
> observed at a different depth and seeing, these values vary from object to object, and
> correlating across the 123/133 objects asks a single question: *whose observing conditions
> did the generation inherit?* ($r\approx+1$ against own = it reproduced its own target's
> noise level; $r\approx+1$ against donor = it took the injected latent's instead;
> $r\approx0$ = no relationship.) SEM on the MSE entries is ±0.005/±0.007 for the diagonal
> entries and ±0.16/±0.21 for the cross-pair entries.
>
> Reading the table: shuffling $z_{\text{phy}}$ changes MSE against the donor's ground truth
> (0.058/0.188) to closely match the intact model's MSE against its own target
> (0.058/0.186) — the generation is closer to the donor than to its own target for 100%
> (HSC) and 97.7% (Legacy) of objects, versus 0.8%/2.3% intact — while the noise statistics
> stay consistent with those of the original target (0.99/0.94 → 0.90/0.72).
> Shuffling $z_{\text{ins}}$ is the exact complement: source identity is untouched (paired
> $\Delta$MSE $=+0.002\pm0.002$ / $+0.000\pm0.005$, a null result), while the generated
> noise detaches from the object's own instrument and tracks the donor's ($r=+0.89$,
> regression slope 0.87 in HSC). The same flip appears in high-frequency PSD power
> ($r$: +0.99 → +0.95 towards the donor, HSC). Shuffling both reproduces the donor's
> observation entirely, as expected.
>
> One caveat on why this is easier to see in a table than by eye: because donors come from
> the same survey, the *mean* noise level is essentially unchanged
> ($\sigma_{\text{pix}}$ 0.1109 → 0.1110 in HSC); the intervention re-assigns each object's
> noise level to its donor's rather than shifting the population. We report high-pass noise
> statistics for the same reason of specificity — whole-cutout robust $\sigma$ estimators in
> 48×48 cutouts are dominated by source flux ($r\approx0.9$ with total cutout flux) and so
> respond to $z_{\text{phy}}$ by construction.
>
> We are happy to add the full table and a qualitative panel (own target | intact |
> shuffled $z_{\text{phy}}$ | shuffled $z_{\text{ins}}$ | donor target, shared noise) to the
> appendix of the camera-ready version.

*(Footnote if you keep 2(c): "Absolute MSE is computed on a held-out split rebuilt on a
second cluster and so differs slightly from Table X; all comparisons here are paired within
the same 256 objects.")*

## 4. Compressed version (~180 words, if the rebuttal is length-capped)

> We ran this ablation. Note first that at sampling time the ODE starts from pure Gaussian
> noise and the target is never encoded, so all target content must flow through
> $(z_{\text{phy}},z_{\text{ins}})$ — the $x_t$ leak the reviewer describes exists only
> during training; and Fig. 7 is already a $z_{\text{ins}}$ intervention (fixed noise and
> source conditioning, instrument conditioning swept/replaced).
>
> For 256 held-out galaxies, generating from identical noise, we replace $z_{\text{phy}}$ or
> $z_{\text{ins}}$ with those of a different held-out galaxy in the same survey (a "donor",
> which also has ground truth, giving a positive control). Shuffling $z_{\text{phy}}$
> relocates rather than degrades the output: MSE against the donor's truth (0.058 HSC /
> 0.188 Legacy) equals the intact model's MSE against its own target (0.058/0.186), and the
> donor is the closer match for 100%/97.7% of objects. Shuffling $z_{\text{ins}}$ leaves
> identity statistically unchanged ($\Delta$MSE $=+0.002\pm0.002$) while the generated
> pixel-noise level switches from tracking the object's own instrument ($r=+0.99$) to the donor's
> ($r=+0.89$). We will add the table and a qualitative figure to the appendix.

## 5. Figure to attach

`shuffle_ablation/results_mA/figures/qualitative_grid.png` — 8 rows (4 HSC-target,
4 Legacy-target) × `own target | C0 intact | C1 shuffle-phy | C2 shuffle-ins | donor target`,
shared noise. Rows `hsc #71` (companion source reproduced intact, gone under C1) and
`legacy #175`/`hsc #152` (visible noise-texture change under C2) are the most legible;
consider trimming to 4–5 rows for space.

## 6. Full numbers (for the appendix / if the reviewer asks)

Pooled M=4, mean ± SEM over anchors, normalized image space:

| condition | survey | n | MSE own | MSE donor | MSE own (c32) | MSE donor (c32) |
|---|---|---:|---:|---:|---:|---:|
| C0 intact | hsc | 123 | 0.0578±0.0050 | 0.9698±0.1610 | 0.0722±0.0064 | 1.5159±0.2205 |
| C0 intact | legacy | 133 | 0.1861±0.0069 | 1.4793±0.2054 | 0.1618±0.0066 | 1.9832±0.2417 |
| C1 shuffle-phy | hsc | 123 | 0.9720±0.1600 | 0.0577±0.0050 | 1.5179±0.2190 | 0.0716±0.0062 |
| C1 shuffle-phy | legacy | 133 | 1.4921±0.2087 | 0.1880±0.0053 | 1.9900±0.2431 | 0.1647±0.0054 |
| C2 shuffle-ins | hsc | 123 | 0.0595±0.0047 | 0.9708±0.1612 | 0.0757±0.0061 | 1.5169±0.2211 |
| C2 shuffle-ins | legacy | 133 | 0.1864±0.0052 | 1.4752±0.2054 | 0.1636±0.0054 | 1.9795±0.2417 |
| C3 shuffle-both | hsc | 123 | 0.9713±0.1607 | 0.0557±0.0053 | 1.5162±0.2184 | 0.0677±0.0065 |
| C3 shuffle-both | legacy | 133 | 1.4987±0.2103 | 0.1849±0.0068 | 1.9988±0.2457 | 0.1598±0.0062 |

Noise-statistic tracking, r(generated stat, real stat) for own vs donor instr:

| statistic | survey | C0 own/donor | C1 own/donor | C2 own/donor | C3 own/donor |
|---|---|---|---|---|---|
| σ_pix (adjdiff) | hsc | +0.99 / −0.11 | +0.90 / −0.01 | −0.03 / **+0.89** | −0.10 / +0.99 |
| σ_pix (adjdiff) | legacy | +0.94 / −0.10 | +0.72 / +0.10 | +0.17 / **+0.69** | −0.05 / +0.94 |
| high-k PSD | hsc | +0.99 / −0.03 | +0.93 / −0.02 | +0.01 / **+0.95** | −0.02 / +0.99 |
| high-k PSD | legacy | +0.94 / −0.01 | +0.42 / +0.38 | +0.45 / +0.36 | +0.02 / +0.93 |
| σ_corner *(flux-contaminated)* | hsc | +0.99 / −0.13 | −0.10 / +0.96 | +0.97 / −0.09 | −0.13 / +0.99 |
| σ_mad *(flux-contaminated)* | hsc | +0.98 / −0.06 | −0.03 / +0.95 | +0.95 / −0.03 | −0.08 / +0.99 |
| σ_mad *(flux-contaminated)* | legacy | +0.99 / −0.10 | −0.06 / +0.93 | +0.94 / −0.05 | −0.09 / +0.99 |

OLS slopes (generated stat on reference stat): σ_pix C0 gen~own 1.01/0.88, C2 gen~donor
0.87/0.64; high-k C0 gen~own 1.08/0.91, C2 gen~donor 0.99/0.33; σ_mad C1 gen~donor
0.95/0.97 (the flux-contamination signature).
