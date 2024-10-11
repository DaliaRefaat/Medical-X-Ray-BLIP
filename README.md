# Medical-X-Ray-BLIP
# MedBLIP - Image Captioning for Chest X-Ray Images

This repository implements **BLIP (Bootstrapping Language-Image Pre-training)** for generating captions from chest X-ray images. BLIP is a powerful vision-language pre-training model capable of handling image-captioning tasks efficiently. This project focuses on medical imaging, specifically applying BLIP to the **Radiology Objects in COntext (ROCO)** dataset. 
<img width="534" alt="Screenshot 2024-10-12 at 1 04 33 AM" src="https://github.com/user-attachments/assets/b2b261e7-63a3-49c1-91c3-d13bfb4f8f48">

 
## Features

- **BLIP Model**: Uses state-of-the-art vision-language pre-training for image captioning.
- **Medical Imaging**: Tailored for radiology, particularly chest X-ray captioning.
- **Preprocessing Pipeline**: Prepares and processes the ROCO dataset for training and evaluation.
- **Model Training**: Easy-to-use notebook format for training, monitoring, and evaluating the model.

## Dataset

The project uses the **ROCO (Radiology Objects in COntext)** dataset, which contains radiology images and their corresponding text descriptions. The dataset can be obtained from [ROCO Dataset](https://www.kaggle.com/datasets/virajbagal/roco-dataset).

## Requirements

To run this project, you need:

- Python 3.8 or higher
- PyTorch
- Hugging Face Transformers
- TQDM (for progress tracking)
- CUDA (optional, for GPU acceleration)

## Installation

Follow these steps to install the required dependencies and set up the project:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MedBLIP-Chest-Xray-Captioning.git
    ```

2. Navigate to the project directory:
    ```bash
    cd MedBLIP-Chest-Xray-Captioning
    ```

3. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Dataset Preparation**:
    - Download and prepare the ROCO dataset.
    - Follow instructions in the Jupyter notebook for dataset loading and preprocessing.

2. **Training**:
    - Use the notebook to begin training the BLIP model on your dataset:
      ```bash
      jupyter notebook MedBLIP-Chest-Xray-Captioning.ipynb
      ```

3. **Real-time Progress**:
    - The training process uses TQDM to display real-time progress and epoch information.

4. **Evaluation**:
    - Once training is complete, evaluate the model’s performance on the test set and generate captions for chest X-ray images.

## Example

Sample code to start training:
```python
# Start training
for epoch in range(5):
    print("Epoch:", epoch)
    for idx, batch in enumerate(train_dataloader):
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)

        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Loss:", loss.item())
