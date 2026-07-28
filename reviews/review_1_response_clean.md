# Reviewer 1 (SYJm) — response

We thank the reviewer for their comments and highlight the timeliness for the development of structured foundation models in astronomy that capture the instrument characteristics. Below we address their concerns and questions.

---

## Weaknesses

### W1. Flow matching objective.

We acknowledge that the conditional distribution could be learned with any generative model, and this is not a fundamental aspect of our proposed architecture. We chose flow matching for its faster convergence, both during training and inference, and improved reconstructions in earlier experiments where we tried both VAEs and diffusion models.

We have now run an ablation where we fix our neural network architecture but change the objective from flow matching to score-based diffusion, and find the following.

**No significant difference in downstream task performance** after training both models for 75k training steps. $R^2$ scores on physics properties predicted from the physics latent space:

| Target | FM | DDPM | Δ |
|---|---|---|---|
| desi_z | 0.804 | 0.817 | +0.013 |
| provabgs_logmstar | 0.763 | 0.763 | +0.001 |
| ssfr | 0.542 | 0.531 | −0.011 |
| legacy_shape_e1 | 0.924 | 0.932 | +0.008 |
| legacy_shape_e2 | 0.917 | 0.920 | +0.002 |
| hsc_shape_e1_i | 0.924 | 0.919 | −0.006 |
| hsc_shape_e2_i | 0.942 | 0.940 | −0.002 |

**The image generation quality shows slower convergence.** After 75k training steps, the MSE on counterfactual generation on a held-out validation set is

| Sampler steps | FM (Euler) | Best DDPM (η=1) |
|---|---|---|---|
| 25 | 0.044 / 0.154 | 0.195 / 0.355 |
| 50 | 0.047 / 0.160 | 0.143 / 0.288 |
| 100 | 0.048 / 0.164 | 0.119 / 0.255 |
| 250 | 0.049 / 0.166 | 0.108 / 0.244 |

> **TODO (Pablo):** add the n=1000 row. Also state once what the two numbers in each cell are (HSC / Legacy?) — a reader can't tell from the table alone.

The MSE is still decreasing with further training steps, so it might have not fully converged; on the other hand, the loss for the FM model seems to decrease more smoothly and converge well before 75k steps. Additionally, we find that the reconstructions require a higher number of sampling steps for diffusion reconstruction quality to converge (as shown in the table above, FM seems to have converged at only 25 steps while the DDPM keeps improving up to 250).

In summary:

- the choice of generative model is not the main point of the paper;
- we didn't do exhaustive parameter tuning for the diffusion model;
- it requires more steps for sampling;
- extra parameters (eta).

---

## Questions

### Q1. Attention-based conditioning.

**Token layout.** Each encoder maps one 48×48×4 image to a sequence of tokens rather than a single vector: the ResNet-18 trunk's final feature map is projected by a 1×1 convolution to 16 channels over a 2×2 grid, giving 4 spatial tokens of dimension 16 per image (Appendix A.1, Table 1).

**Building the conditioning sequence.** For a target observation $x_{j,k}$ we encode (i) the same source seen by the other instrument with the physics encoder, giving 4 tokens, and (ii) each of the $K$ same-instrument neighbours with the instrument encoder, giving $4K$ tokens. These are concatenated along the sequence axis into a single set of $4(1+K)$ tokens of dimension 16. For example, with $K=5$, 24 tokens.

**How it conditions the network.** That sequence is used as the key/value tokens of the cross-attention layers of the velocity network $u_\theta$ (a `UNet2DConditionModel` following the implementation from `diffusers`); the UNet's spatial activations are the queries. Attention is applied at every resolution, in all four down-blocks and all four up-blocks, so each location in the noisy image attends to whichever conditioning tokens are relevant to it.

**Why this is the mechanism that handles a variable number of neighbours.** Attention is agnostic to sequence length and permutation-equivariant over the sequence, so the same parameters accept any number of conditioning views with no architectural change and no ordering convention, unlike other conditioning schemes that concatenate a fixed number of embeddings along the channel or feature axis, which fix the number of views. This is what makes the general formulation of Eq. (Sec. 3) usable: the physics set $\{z_{\text{phy}}(x_{j,k'})\}_{k'\neq k}$ and the instrument set $\{z_{\text{ins}}(x_{j',k})\}_{j'\neq j}$ can each have any cardinality. In the present two-survey instantiation the physics set has one element (the other survey) and the instrument set can have many sky neighbours, which we set to a maximum of 5 in our training run, but the same trained module would accept more instruments per source, which is the direction we discuss for extending beyond two surveys.

This is briefly described in Section 4 (in the "Model Implementation" paragraph) and Appendix A.1, but the sentence you quote is in Section 3 and does not point forward to them; we will add that cross-reference, and a more thorough explanation in the appendix. We thank the reviewer for pointing out this omission, and we will add this comprehensive description to Appendix A.1.

### Q2. Decoder reliance ablation.

We thank the reviewer for this suggestion, we agree it is the right diagnostic, and we have run it.

**New ablation.** For 256 held-out galaxies we generate samples from the same initial noise realization $x_0$ under three conditions: "intact" conditioning (running the model in the regular way described in the paper); $z_{\text{phy}}$ replaced by that of a different held-out galaxy ("donor"); $z_{\text{ins}}$ replaced by the donor's instrument conditioning. Donors are drawn within the target survey, so the intervention is not a survey swap (that is Figs. 6–7). Crucially, each donor also has ground truth in the target survey, so we can test not only how the corruption hurts against the target but how the generation follows the donor. Averaged over 4 posterior samples and tested on both surveys (HSC-target / Legacy-target):

| | intact | shuffled $z_{\text{phy}}$ | shuffled $z_{\text{ins}}$ |
|---|---|---|---|
| MSE vs **own** target | 0.058 / 0.186 | 0.97 / 1.49 | 0.060 / 0.186 |
| MSE vs **donor** target | 0.97 / 1.48 | **0.058 / 0.188** | 0.97 / 1.48 |
| $r(\sigma_{\text{pix}},$ own instr$)$ | +0.99 / +0.94 | +0.90 / +0.72 | −0.03 / +0.17 |
| $r(\sigma_{\text{pix}},$ donor instr$)$ | −0.11 / −0.10 | −0.01 / +0.10 | **+0.89 / +0.69** |

Here $\sigma_{\text{pix}}$ is a robust MAD-based $\sigma$ of adjacent-pixel differences, i.e. the pixel-scale noise floor of an image. Every image — generated or real — has one such value, so for each object we have three: the generation's, that of the real observation we are trying to reproduce ("own instr"), and that of the unrelated observation whose $z_{\text{ins}}$ we injected ("donor instr"). Since every field was observed at a different depth and seeing, these values vary from object to object, and correlating across the 123/133 objects asks a single question: whose observing conditions did the generation inherit? ($r\approx+1$ against own = it reproduced its own target's noise level; $r\approx+1$ against donor = it took the injected latent's instead; $r\approx0$ = no relationship.)

**Reading the table:** shuffling $z_{\text{phy}}$ changes MSE against the donor's ground truth (0.058 / 0.188) to closely match the intact model's MSE against its own target (0.058 / 0.186), while the noise statistics stay consistent with those of the original target (correlation goes from 0.99 / 0.94 to 0.90 / 0.72). Shuffling $z_{\text{ins}}$ is the exact complement: source identity is untouched (paired $\Delta$MSE $=+0.002\pm0.002$ / $+0.000\pm0.005$, a null result), while the generated noise detaches from the object's own instrument and tracks the donor's ($r=-0.03$ / $+0.17$ for the target's own instrument, and $r=+0.89$ / $+0.69$ on the donor's instrument).

A final note: Fig. 7 is already a $z_{\text{ins}}$ intervention. Both panels hold the source conditioning and the initial noise fixed and replace only the instrument conditioning: the SNR traversal sweeps $z_{\text{ins}}$ across observing-condition percentiles, and the artifact-removal panel supplies an HSC+HSC combination never seen in training. Appendix A.5 is the training-time analogue on the same pathway (random instead of sky-neighbour conditioning collapses instrument-property recovery, Galaxy Depth $R^2$ 0.68 → 0.20).
