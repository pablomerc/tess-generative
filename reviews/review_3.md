# Official Review of Submission20256 by Reviewer myQM

**Official Review** by Reviewer myQM | 02 Jun 2026 at 08:47 (modified: 23 Jul 2026 at 08:27)
Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer myQM | Revisions

**Summary:**
The paper proposes a causal generative model for disentangling shared signal components from sensor-specific artifacts in multi-sensor data. The method employs a CVAE-based architecture with a shared encoder and decoder for learning common latent representations across sensors, complemented by independent per-sensor encoders and decoders for sensor-specific noise components. The approach is evaluated on synthetic benchmarks with controlled signal-to-noise ratios.

**Contribution Type:** General: Most submissions will fall into this type.

**Strengths And Weaknesses:**
The paper addresses a practically important problem—disentangling true physical signals from sensor-specific measurement artifacts. The causal perspective motivating the approach is conceptually appealing: shared causal factors (the true signal) versus sensor-specific factors (artifacts) provide a clear structural inductive bias. The presentation significantly undermines the paper's otherwise solid technical contribution. In terms of originality, it is moderate—the causal framing and consistency regularizer are reasonable but not clearly differentiated from existing approaches in representation learning.

**Quality:** 3: good
**Clarity:** 2: not good
**Significance:** 2: not good
**Originality:** 2: not good

**Questions:**
* How does the proposed causal CVAE architecture and consistency regularizer specifically differ from established shared-private representation learning and multi-view disentanglement methods?
* What is the theoretical or empirical justification for Pearson correlation-based weighting in the consistency regularizer?
* Were latent representations on real gravitational wave data qualitatively validated?
* How does the method handle variable/missing sensors?

**Limitations:**
* Performance degradation with weakly correlated sensors: What happens when sensors share little signal information or have very low inter-sensor correlation? The method's effectiveness likely degrades in such regimes, and the paper does not discuss this operational boundary.
* Failure modes: Under what conditions would the method fail to properly separate signal from artifacts? For example, if artifacts are correlated across sensors (e.g., environmental noise affecting multiple detectors), the causal assumption breaks down—this scenario is not discussed.

**Rating:** 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

**Ethical Concerns:** NO or VERY MINOR ethics concerns only
**Paper Formatting Concerns:**
* 

**Code Of Conduct Acknowledgement:** Yes
**Responsible Reviewing Acknowledgement:** Yes
