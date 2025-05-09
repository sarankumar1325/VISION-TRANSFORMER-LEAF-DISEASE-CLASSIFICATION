# Fine-Tuning Vision Transformers for Bean Disease Classification

![Image](https://github.com/user-attachments/assets/da6c2988-4c96-4724-a8fa-a1d3f55977f8)


This repository contains code and documentation for fine-tuning a Vision Transformer (ViT) model to classify bean leaf images into three categories: Angular Leaf Spot, Bean Rust, and Healthy. The project is based on the notebook `image_classification_explained.ipynb` and leverages the Hugging Face transformers and datasets libraries.

## Project Overview

The goal of this project is to demonstrate how to fine-tune a pre-trained Vision Transformer (ViT) model (`google/vit-base-patch16-224-in21k`) for image classification on the beans dataset. The dataset contains images of bean leaves labeled with three classes:

- **Angular Leaf Spot**: Irregular brown patches.
- **Bean Rust**: Circular brown spots with a white-ish yellow ring.
- **Healthy**: No visible disease symptoms.

The notebook walks through the process of data preparation, model configuration, training, and evaluation, using the Hugging Face ecosystem.

## Prerequisites

To run the code in this repository, you need the following dependencies:

- Python 3.11
- Required Python packages (install via pip):
```
pip install transformers datasets evaluate torch numpy pandas pillow tensorboard
```

## Dataset

The dataset used is the beans dataset, available through the Hugging Face datasets library. It includes:

- Train, validation, and test splits with images and corresponding labels.
- Labels: 
  - 0: Angular Leaf Spot
  - 1: Bean Rust
  - 2: Healthy

You can load the dataset using:
```python
from datasets import load_dataset
ds = load_dataset('beans')
```

## Model

The model used is `google/vit-base-patch16-224-in21k`, a pre-trained Vision Transformer. The model is fine-tuned with a classification head for the three bean disease classes. Key configurations include:

- Input size: 224x224 pixels
- Number of labels: 3
- id2label and label2id mappings for human-readable labels.

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/bean-disease-classification.git
cd bean-disease-classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare the environment:
   - Ensure you have a GPU-enabled environment for faster training, as the notebook uses `fp16=True` for mixed-precision training.

## Usage

The notebook (`image_classification_explained.ipynb`) provides a step-by-step guide to:

1. **Loading and exploring the dataset**:
   - Visualize examples from each class.
   - Understand the dataset's structure and labels.

2. **Preprocessing images**:
   - Use `ViTFeatureExtractor` to apply transformations (resize, normalize, etc.) required by the ViT model.
   - Apply transformations to the dataset using a transform function.

3. **Defining the training pipeline**:
   - Data collator: Stack pixel values and labels into batches.
   - Evaluation metric: Compute accuracy using the evaluate library.
   - Model: Load and configure the pre-trained ViT model.
   - Training arguments: Configure hyperparameters (e.g., learning rate, batch size, epochs).

4. **Training the model**:
   - Use Hugging Face's Trainer API to fine-tune the model.
   - Monitor training with TensorBoard.

5. **Evaluation**:
   - Evaluate the fine-tuned model on the validation set to verify its performance.

To run the notebook:
```bash
jupyter notebook image_classification_explained.ipynb
```

## Key Code Snippets

### Data Transformation
```python
from transformers import ViTFeatureExtractor
model_name_or_path = 'google/vit-base-patch16-224-in21k'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['labels']
    return inputs
```

### Training Configuration
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./vit-base-beans-demo-v5",
    per_device_train_batch_size=16,
    eval_strategy="steps",
    num_train_epochs=4,
    fp16=True,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=True,
)
```

### Evaluation Metric
```python
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
```

## Results

The fine-tuned model achieves high accuracy on the validation set, successfully distinguishing between Angular Leaf Spot, Bean Rust, and Healthy bean leaves. Detailed results can be visualized using TensorBoard logs stored in the `output_dir`.

## Notes

- Ensure `remove_unused_columns=False` in TrainingArguments to retain the image column needed for preprocessing.
- The notebook assumes a GPU environment for fp16 training. If running on CPU, set `fp16=False`.
- The `ViTFeatureExtractor` is deprecated in newer versions of transformers. Future updates may require using `ViTImageProcessor` instead.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for bug fixes, improvements, or additional features.


## Acknowledgments

- Hugging Face for providing the transformers, datasets, and evaluate libraries.
- Google Brain for the Vision Transformer (ViT) model.
- The beans dataset creators for making the data publicly available.
