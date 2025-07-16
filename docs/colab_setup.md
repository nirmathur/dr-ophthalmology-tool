# Training on Google Colab

This short guide covers how to clone the repository on Colab, install the dependencies and train the model while keeping the dataset and outputs on Google Drive.

1. **Mount Google Drive**

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Clone the repository**

   Clone the repo inside your Drive so that training artefacts persist after the
   session ends:

   ```bash
   %cd /content/drive/MyDrive
   git clone https://github.com/your-username/dr-ophthalmology-tool.git
   %cd dr-ophthalmology-tool
   ```

3. **Install requirements**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set the dataset directory**

   Either edit `src/config.py` so that `DATA_DIR` points to your Drive location, e.g.

   ```python
   DATA_DIR = "/content/drive/MyDrive/aptos2019-blindness-detection"
   ```

   or pass the path via `--data-dir` when calling the CLI.

5. **Start training**

   ```bash
   python -m src.cli train --data-dir /content/drive/MyDrive/aptos2019-blindness-detection --epochs 30
   ```

   Because the repo is inside your Drive, all logs and checkpoints saved under
   `model_output/` and `logs/` will be stored there automatically.

