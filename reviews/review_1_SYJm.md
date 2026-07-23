# Official Review of Submission20256 by Reviewer SYJm

**Official Review** by Reviewer SYJm | **26 Jun 2026 at 03:29 (modified: 23 Jul 2026 at 11:27)** | **Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer SYJm**

**Revisions**

**Summary:**
This paper try to learn representations from multi-instrument observations that separate the intrinsic physical signal of an object from instrument-specific artifacts. Their method is a dual-encoder + conditional flow-matching decoder framework. One encoder encode the source galaxy from different sursey, the other one encode the nearby galaxy observed by the target instrument, then train a decoder in flow-matching loss to reconstruct source galaxy in target instrument. The experimental results show that the learned physics and instrument latent spaces capture different factors of variation, enabling downstream tasks such as parameter inference, outlier detection, instrument-independent similarity search, counterfactual cross-instrument generation, and data-driven denoising.

**Contribution Type:** General: Most submissions will fall into this type.

**Strengths And Weaknesses:**

**Strengths**

1. The topic studied by the authors is timely and important for astronomy.
2. The paper is well written and easy to follow.
3. The paper provides comprehensive analysis and empirical results to support its claims.

**Weaknesses**

1. The paper does not sufficiently justify the choice of the flow-matching objective. In my understanding, the conditional distribution modeled in the paper could also be learned using a diffusion objective or a conditional VAE training framework.

**Quality:** 3: good
**Clarity:** 4: excellent
**Significance:** 4: excellent
**Originality:** 3: good

**Questions:**

1. Can author explain more on Line 111-112 "Note that the number of source and instrument neighbors may vary; therefore, we combine the variable sequence of embeddings via attention-based conditioning.", what "attention-based conditioning" mean?
2. Decoder reliance ablation. I think it would be useful to include an additional ablation testing whether the decoder actually uses the two conditioning latents. In the flow-matching objective, the decoder receives $(x_t = (1 - t)x_0 + tx_{target})$, so for large (t), the input already contains substantial information about the target galaxy. A simple diagnostic would be to replace either $(z_{phy})$ or $(z_{ins})$ with a shuffled/random latent and examine how the generated output changes. If the disentanglement is effective, corrupting $(z_{phy})$ should mainly affect galaxy morphology/source identity, while corrupting $(z_{ins})$ should mainly affect instrument-specific statistics.

**Limitations:**
yes

**Rating:** 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
**Ethical Concerns:** NO or VERY MINOR ethics concerns only
**Paper Formatting Concerns:** no concerns

**Code Of Conduct Acknowledgement:** Yes
**Responsible Reviewing Acknowledgement:** Yes
