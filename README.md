# create-and-test-generative-model-metric
Recently, new research in the field of image generation has been increasingly appearing. Progress in this area depends on the ability to correctly evaluate new model architectures. One of the most popular metrics for evaluating such models is Fréchet Inception Distance (FID). Several publications show that this metric contains a few inaccuracies, which result in an incorrect perception of the generation results. Due to existing problems, there is a risk of discarding new architectures at the time of their creation and design. The study examines and confirms the existing problems of metrics for evaluating generated images and suggests ways to solve them by creating a framework for constructing alternative metrics. Using the framework, several alternative metrics were created and tested. Based on the results of the comparative analysis, the best metric is selected, which is free from the shortcomings of FID.

## Setup

Install dependencies 

```bash
cd create_and_test_generative_model_metric
pip install -r requirements.txt
```

## Usage Streamlit

```bash
streamlit run front.py
```

## Usage CLI:

```bash
python main.py encoder, class_type, formula, sample_size, effect, power, real_image_path, generated_image_path, batch_size
```

## References:
The work was inspired by [Rethinking FID: Towards a Better Evaluation Metric for Image Generation](https://arxiv.org/abs/2401.09603).

Metrics formula concept usage in [google-research](https://github.com/google-research/google-research/tree/master/cmmd) and in [torchmetrics](https://github.com/Lightning-AI/torchmetrics/blob/27a301a572cf6bd459ba3f5aed7d71f424adbbd1/src/torchmetrics/image/fid.py#L182)
