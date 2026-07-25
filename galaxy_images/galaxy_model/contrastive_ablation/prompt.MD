We have this project where the codebase lives here
/home/pablomer/orcd/pool/tess-generative

most of the experiments are here
/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model

There is a markdown file here
/home/pablomer/orcd/pool/tess-generative/CLAUDE.md

The paper latex project is here
/home/pablomer/orcd/pool/tess-gen-disentanglement_neurips2026_paper

We submitted to neurips and we got some reviews. Now I want to work on addressing this one
/home/pablomer/orcd/pool/tess-generative/reviews/review_2_JcH2.md

They mention that we should do an ablation where we compare our objective against contrastive learning.

I tried doing it in
/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_baseline
But now if we add new things lets do it here /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_ablation to know what was added new after the reviews

but im not convinced by the results

Can you help me with the following things

-did we implement the ablation correctly?

-Let's maybe focus on comparing the "spatial neighbors cv" version with ours, is this the most similar to how "ours" was trained?

-Is the latent space size of the instrument matched (4x16, 64, etc) between Ours and the contrastive one?

-Was the downstream task set excluded during the training of the contrastive baseline?

-How were the neighbors used for positive and or negative supervision in each space? Is there something where the negative supervision is giving the contrastive model and unfair advantage and that we could tune/didnt try already?

Note that the thing that bothers me a little is that contrastive seems to get as good of a result as Ours in the things where we expect it to score high, while getting an R2 of ~0 on the ones which we expect it to score low which seems to good to be true

Please do some thorough analysis and report back to me, and give me an implementation of plan of any additional experiments needed
