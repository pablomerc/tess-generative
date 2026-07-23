*Reviewer*:

Q1. Can author explain more on Line 111-112 "Note that the number of source and instrument neighbors may vary; therefore, we combine the variable sequence of embeddings via attention-based conditioning.", what "attention-based conditioning" mean?

*Author response*:






*Reviewer*:

Q2. Decoder reliance ablation. I think it would be useful to include an additional ablation testing whether the decoder actually uses the two conditioning latents. In the flow-matching objective, the decoder receives $(x_t = (1 - t)x_0 + tx_{target})$, so for large (t), the input already contains substantial information about the target galaxy. A simple diagnostic would be to replace either $(z_{phy})$ or $(z_{ins})$ with a shuffled/random latent and examine how the generated output changes. If the disentanglement is effective, corrupting $(z_{phy})$ should mainly affect galaxy morphology/source identity, while corrupting $(z_{ins})$ should mainly affect instrument-specific statistics.


*Author response*:



*Reviewer*:
W1. The paper does not sufficiently justify the choice of the flow-matching objective. In my understanding, the conditional distribution modeled in the paper could also be learned using a diffusion objective or a conditional VAE training framework.
