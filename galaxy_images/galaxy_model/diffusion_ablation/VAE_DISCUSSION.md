# The "conditional VAE" option — what the reviewer means, and why we should rebut it in text

Reviewer SYJm, W1: *"the conditional distribution modeled in the paper could also be learned using a
diffusion objective or a conditional VAE training framework."*

The diffusion half is a fair ask and we're ablating it ([`PLAN.md`](PLAN.md)). The cVAE half is a
categorically different proposal, and the honest answer is that it conflicts with the paper's core
design constraint. This note unpacks it.

---

## 1. What a "conditional VAE training framework" would mean here

Our model learns `p(x_target | c)` where the condition is
`c = ({z_phy(other-instrument views)}, {z_ins(other-source neighbors)})`. A conditional VAE
(Sohn et al. 2015) for the same conditional would add:

- a **recognition network** `q_φ(w | x_target, c)` — a *third* encoder that sees the **target**
  image and outputs mean/variance of a stochastic latent `w` (reparametrization trick);
- a **one-shot decoder** `p_θ(x_target | w, c)` — a single forward pass (no iterative refinement)
  with, in practice, a Gaussian likelihood, i.e. an MSE reconstruction term;
- the **ELBO**: `E_q[log p(x|w,c)] − β·KL(q(w|x,c) ‖ p(w))`, with prior `p(w) = N(0,I)` (or a
  learned conditional prior `p(w|c)`).

At test time you sample `w ~ p(w)` and decode once. `w` is what carries the stochasticity —
"which exact noise realization / appearance did this observation have" — that our FM model carries
through the noising process x₀.

## 2. Direct answer: would adding a KL loss on your latents make it a VAE?

**No.** It conflates two latents with different roles:

- `z_phy` / `z_ins` are deterministic encodings of **other images** (the conditioning views). In
  cVAE terms they are part of the *condition* `c`, not the VAE latent. A VAE's defining latent is a
  stochastic code of the **target**, inferred by a posterior network with reparametrized sampling
  and a KL term — and our encoders never see the target, by construction.
- Adding KL/noise on `z_phy`/`z_ins` while keeping the FM decoder gives "flow matching with
  stochastically regularized conditioning" — still flow matching, just with an information
  bottleneck on the conditioning pathway. (That's actually an interesting knob — it's essentially
  the instrument-bottleneck-capacity study the paper defers to future work, tex line 416 — but it
  is not what the reviewer is describing and wouldn't answer W1.)

## 3. What a faithful cVAE version would cost

Same dual encoders (as `c`), plus: a new target-encoder `q_φ(w|x_target, c)`, a single-pass decoder
(the same UNet could be reused with a fixed timestep and `w` injected as extra tokens or a class
embedding), the ELBO with a β weight — and realistically a **β sweep (≥3 full retrains)**, because
β directly controls the failure modes below and there is no principled default for this data. So
it's not one matched run; it's a small research project with its own hyperparameter search.

## 4. Why we expect it to underperform (your intuition is right, with specific reasons)

**(a) The recognition network is a target-information side channel that undermines the
counterfactual constraint.** The paper's central design rule (Fig. 1 caption, tex line 148): *the
anchor is never fed into the encoders; it is only the target of the loss* — that's what forces all
information to flow through the counterfactual conditioning. A cVAE *by definition* feeds the
target into `q_φ`. The β dial then has two bad ends:

- **β large → posterior collapse.** With strong conditioning, KL pressure drives `q(w|x,c) → p(w)`;
  `w` is ignored and the model degenerates into deterministic regression `c → E[x|c]` — a blurry
  posterior mean with no noise model, exactly the thing the paper's generative results can't live
  without.
- **β small → the model cheats.** `w` encodes the target; the decoder can reconstruct from `w` and
  lean less on `z_phy`/`z_ins`, weakening the training signal that makes the representations good
  (this is precisely the leakage worry the reviewer *themselves* raises in Q2 about large-t x_t —
  except in the cVAE it's a designed-in channel, not a transient of the noising process).

There is no guarantee of a good middle, and finding it = tuning a hand-engineered loss balance,
which contradicts the paper's stated design principle ("…rather than relying on hand-engineered
loss functions", tex line 160).

**(b) Gaussian-likelihood one-shot decoders are oversmoothers.** `p(x|w,c) = N(μ, σ²I)` means
posterior samples are "blurry mean + i.i.d. noise". But half of what our decoder must model is the
**instrument noise texture** — spatially correlated noise, PSF structure, artifacts. The paper
*measures* exactly this: the power-spectral-density and pixel-lag autocorrelation eval (appendix
Fig. 8) and the pixel-wise posterior calibration Z-scores (Fig. 9). A cVAE would fail these
visibly — the high-frequency end of the PSD gets averaged away. Iterative denoisers (FM/diffusion)
are precisely the model family that fixed this failure mode in the image-generation literature.

**(c) The nuance worth conceding:** as a pure *representation learner*, a cVAE might be fine — the
downstream R² probes only touch the encoders, and the reviewer is right that the conditional could
be "learned" that way in the weak sense. But the paper's contributions are not just embeddings:
counterfactual generation, posterior sampling and calibration, zero-shot pipeline transfer, the
data-driven noise model (Fig. 6, 7, 9) all depend on the decoder being a *good conditional
generative model* with a controllable stochastic channel (the fixed-x₀ SNR traversal literally
requires a deterministic map from a fixable noise variable). That is what rules the cVAE out as the
instantiation, not taste.

## 5. Recommendation

**Do not implement a cVAE for the rebuttal.** Answer it in text:

1. FM and diffusion are the same objective family (and we ablate diffusion empirically — see
   PLAN.md);
2. a cVAE is structurally different: it requires a recognition network on the target, which
   breaks the counterfactual constraint that defines the method, introduces the posterior-collapse
   vs. leakage β dilemma, and its Gaussian one-shot decoder is a known oversmoother that would fail
   our PSD/calibration evals — the very capabilities (noise modeling, posterior sampling) the paper
   is about;
3. offer it as future work / camera-ready appendix only if the AC pushes.

### Draft response snippet (for `reviews/response_1_SYJm.md`, W1, VAE part)

> Regarding a conditional VAE: we agree the conditional distribution could in principle be
> parameterized this way, but it conflicts with the central design constraint of our method. A cVAE
> requires a recognition network q(w | x_target, c) that observes the target during training,
> whereas our framework deliberately never exposes the anchor to any encoder — all information must
> flow through the counterfactual conditioning. Introducing w creates a direct target-information
> side channel: with a strong KL weight the latent collapses and the decoder degenerates to a
> (blurry) posterior-mean regressor with no stochastic instrument model; with a weak KL weight the
> decoder can bypass the conditioning latents, weakening the disentanglement training signal — a
> designed-in version of the leakage concern the reviewer raises in Q2. Moreover, one-shot decoders
> with Gaussian likelihoods systematically oversmooth high-frequency structure, while modeling
> instrument noise statistics is a core requirement of our application (cf. the power-spectrum and
> posterior-calibration analyses in Appendix Figs. 8–9). Iterative denoising objectives
> (flow matching / diffusion, which we now compare directly — see above) avoid both issues while
> keeping the target outside all encoders. We will add this justification to Section 3.
