# Official Review of Submission20256 by Reviewer JcH2

**Official Review** by Reviewer JcH2 | **Date:** 17 Jun 2026 at 15:52 (modified: 23 Jul 2026 at 11:27) | **Readers:** Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer JcH2

---

**Summary:**
The author(s) present a novel dual-encoder architecture for separating source and instrumental signals in astrophysical images, specifically visual wavelength images of galaxies. By generating a dataset of three tuples(target: source-instrument, source path: source-instrument2, instrument path: source2-instrument), the authors train their encoders to differentiate between instrumental latent spaces and source emission latent spaces by training encoders on source and instrument path and a counterfactual loss flowing from a reconstruction of the target image. The contributions include a use case, a careful study of the new architecture and training method against existing work, and convincing visual and quantitative demonstrations of the method.

**Contribution Type:** General: Most submissions will fall into this type.

**Strengths And Weaknesses:**
The universe is a big place, and time on large glass is extremely limited. The ability to remove "lucky searching" and even surveys from large glass and shift to high return followup on rare phenomena is important, and this paper delivers one toolset to most effectively use existing and ongoing lower quality survey missions to inform the targeting of more capable but lower throughput instruments. While the Audenaert paper already introduced a paired encoder architecture for physics / instrument separation, this work's introduction of a generative objective and real survey data is important.

General comments / questions for the authors. I put these here instead of "questions" because while as a reader I want to know the answers, answering them wouldn't tend to boost my score (i.e. I could wait for the followup paper)
* do you have an idea of "random guessing" for R^2 in Fig 4? I'm wondering what the comment in L222-223 about the untrained ResNet outperforming your encoder -- I don't see this anywhere where your encoder *should* outperform base performance. Is this a comment about the untrained encoder performing better than your instrument encoder on physical properties? If so, is that not an inductive bias for measuring shape (ellipticity) that the instrument encoder might suppress?
* Is there any way to use this to find rare phenomena in HSC that do not appear in Legacy (higher resolution / depth required, things which only appear at higher z but require spatial resolution, etc)? If there was a way to tease this out, and nothing was found, you might be able to use this to suggest future survey design (say, Legacy could maintain X% of rare events found in higher resolution surveys / followups with a cadence allwing Y% more coverage in the same time), or suggest that it actually should have surveyed slower because a whole class of phenomena was entirely missed.

Strengths:
* The writing is clear and crisp; the new method and experiments which tease out the performance and potential pitfalls are well framed, for example the use of a random initialized baseline encoder.
* The "query by HSC observation" Figure 5 and discussion is extremely motivating for how one might surface similar objects that could use deeper / higher resolution followups from faster surveys.
* The use of the generative decoder to probe upsampling, processing pipelines, etc at first made me extremely nervous (by how well it works), and I was relieved when the authors carefully point out that this cannot be used in place of higher resolution followups, but is a powerful technique to design followup studies.
* the surfaced "outliers" speak for themselves -- there is a clear and obvious distinction between "physically relevant" physical outliers and AION-1's instrument / noise outlier dominated surface. That Figure (3) could be improved by also showcasing what the "average" source looks like (so non astronomers can see how "outlier-ish" the physics encoder surfaced galaxies look even by eye to non astronomers).
* Data driven noise model is a broadly useful tool that falls out of the dual decoder framework and a powerful way to correct astronomical images in other domains.

Weaknesses:
* The claim of "generative beats contrastive" isn't directly tested. While the results presented do speak for themselves, having a quantitative study on how this beats contrastive (even an ablation against your own encoder design + contrastive head). This would set up a strong comparison to results in Tbl 5, and also for an outlier detection comparison, and demonstrate alongside the current work the "collapse to lowest resolution" issue described in the introduction.
* The parameters studied here are morphological and geometric (shape, redshift, mass, age, etc). These are used to derive and inform the underlying physics / modeling of galaxy formation and evolution, but calling these "physics" (i.e. the "physics encoder") rather than "source morphology" or "scene geometry" or something did throw me a bit until I settled in to the parameter space of interest (Fig 3).
* (minor) Without knowledge of Legacy and HSC, a reader has to wait until S4.0 to know that these instruments cover the same spectral bandpasses (if not with exactly the same filter curves). Might be worth noting that this work is all in the visual wavelengths, as it also clarifies a bit what you mean when you talk about "physics" -- at first I was wondering how you could do this given that different bandpasses can showcase different physics.
* (minor) In S4.0, L138-139, authors discuss "trade-off between coverage and signal-to-noise". I read SNR here to mean "photometric depth"? Is that right, or is angular resolution the tradeoff (or both?).
* (minor) Fig 2 (UMAP) is qualitative, the quantitative view is Tbl 5, I would put a reference to that in the caption to reinforce that you have a quantitative cut of this too (the vis is powerful, though, I'm not arguing against it).
* (minor) S4.1... how "common" does a rare astropysical event / source need to be to be trained into the encoder's latent space? Can you quantify this? Does the architecture have a "resolution" for rare phenomena in terms of training data, etc? Asked differently: how big of a source instrument catalog is needed for robust performance, and how much overlap with the target instrument, as is seen with Legacy / HSC?

**Quality:** 3: good
**Clarity:** 4: excellent
**Significance:** 3: good
**Originality:** 3: good

**Questions:**
1. Your main differentiation from the closest prior work (the Audenart contrastive dual encoder) is swapping its contrastive objective for a generative one, but the claim of improvement is only asserted, not quantified. Could you add an ablation (same encoders, triplets, and data, swapping only the loss for a contrastive one) to compare on the diagnostics you already have (R^2 asymmetry, outlier detection, cross-survey retrieval)? (This is the study that would easily bump my score up.)
2. A quantification of "how rare of phenomena can be detected by the physics encoder" would strengthen the discussion and motivating usecase for followup observations on better instruments, and also motivate seed survey size requirements (for sampling rare phenomena)
3. A panel of "normal" galaxies in Fig 3 would provide non astronomer readers immediate visual grounding and support that those do in fact appear to be outliers, is that something you could surface from the physics encoder?
4. Could you provide any discussion on (or list as a limitation) the amount of data needed to train the architecture? For example, how big of a dataset on each instrument before we could expect to fit a set of reasonably performant physics / instrument encoders?

**Limitations:**
Limitations could also include the fact that these surveys use the same spectral range, and that expanding to different spectral ranges is unstudied, and that the amount of source data required needs to be ablated for application to smaller survey / datasets ("when does this technique become applicable for a new survey?"

**Rating:** 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
**Confidence:** 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
**Ethical Concerns:** NO or VERY MINOR ethics concerns only
**Paper Formatting Concerns:** na

**Code Of Conduct Acknowledgement:** Yes
**Responsible Reviewing Acknowledgement:** Yes
