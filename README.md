# Light distillation for Incremental Graph Convolution Collaborative Filtering

Recommender systems presently utilize vast amounts of data and play a pivotal role in enhancing user experiences.
Graph Convolution Networks (GCNs) have surfaced as highly efficient models within the realm of recommender systems due
to their ability to capture extensive relational information. The continuously expanding volume of data may render the training
of GCNs excessively costly. To tackle this problem, incrementally training GCNs as new data blocks come in has become a vital
research direction. Knowledge distillation techniques have been explored as a general paradigm to train GCNs incrementally
and alleviate the catastrophic forgetting problem that typically occurs in incremental settings. However, we argue that current
methods based on knowledge distillation introduce additional parameters and have a high model complexity, which results in
unrealistic training time consumption in an incremental setting and thus difficult to actually deploy in the real world. In this work,
we propose a light preference-driven distillation method to distill the preference score of a user for an item directly from historical
interactions, which reduces the training time consumption in the incremental setting significantly without noticeable loss in
performance. The experimental result on two general datasets shows that the proposed method can save training time from 1.5x
to 9.5x compared to the existing methods and improves Recall@20 by 5.41% and 10.64 from the fine-tune method.
