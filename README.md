# Stable Region Enhanced Online Learning Method for Intermediate Verification Latency and Concept Drift

---

## Abstract

In non-stationary data streams, alongside the challenges posed by concept drift, Intermediate Verification Latency (IVL)
further complicates timely model adaptation. IVL refers to the finite delay between the arrival of data features and 
their corresponding labels. This paper addresses IVL from a novel perspective by focusing on stable regions within the 
feature space, where data remain unaffected by drift. We propose Centroid-based Drift Index (CDI), an unsupervised 
metric that quantifies drift to identify these stable regions. Building on this, we introduce Data Utilization informed 
by STable regions (DUST), a framework that efficiently utilizes both temporarily unlabeled and delayed labeled data by 
distinguishing stable region data through micro-clusters. Comprehensive experiments on synthetic and real-world datasets
validate the effectiveness of CDI and DUST, demonstrating their superior performance in handling IVL and improving 
model adaptation to evolving data streams. The experimental results demonstrate that DUST offers a robust solution for 
addressing IVL.

## Datasets

All datasets used in this project are available [here](https://drive.google.com/drive/folders/1p9q2DrP52P4dI09y7h_N_ifOrMcXUPvP?usp=sharing).


## Reference

```bibtex
@ARTICLE{DUST-2025,
  author={Zhong, Zixin and Song, Liyan and Tang, Fengzhen and Yuan, Bo},
  title={Stable Region Enhanced Online Learning Method for Intermediate Verification Latency and Concept Drift}, 
  year={2025},
}
```