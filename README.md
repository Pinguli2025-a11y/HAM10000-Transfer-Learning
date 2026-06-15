# Skin Lesion Classification – Deep Learning Project

This repository contains the complete implementation for the **Introduction to Deep Learning with Python** group project.  
We compare two families of models on the **HAM10000** (ISIC 2018 Task 3) dataset:

- **Custom CNNs** (Baseline & Advanced) – built from scratch.
- **ResNet‑18** (transfer learning) – with three experimental configurations (Exp‑A, Exp‑B, Exp‑C).

All models use the same **fixed data split** (seed 42) to ensure fair comparison. The code is written in **PyTorch** and runs on **Apple Silicon (MPS)** or CPU.

## 📁 File Structure

```
project_root/
├── data/
│   ├── ham10000/                      # Original HAM10000 dataset
│   │   ├── HAM10000_metadata.csv
│   │   ├── HAM10000_images_part_1/
│   │   └── HAM10000_images_part_2/
│   ├── test_official/                 # Official ISIC 2018 test set
│   │   ├── ISIC2018_Task3_Test_GroundTruth.csv
│   │   └── ISIC2018_Task3_Test_Images/
│   └── splits/                        # Pre‑generated train/val/test CSV files
│       ├── train_seed42.csv
│       ├── val_seed42.csv
│       └── test_school.csv
├── CNN Model.ipynb                     # Custom CNN pipeline (Baseline & Advanced)
├── Resnet18 Model.ipynb                # ResNet‑18 pipeline (Exp‑A, Exp‑B, Exp‑C, Grad‑CAM)
├── best_resnet18_Exp_B.pth             # Best performing model (optional)
├── gradcam_output/                     # Generated Grad‑CAM images (after running analysis)
└── README.md                           # This file
```

> **Note:** The actual paths to the datasets are hardcoded in the notebooks.  
> Before running, update the path variables (e.g., `TRAIN_IMAGE_DIR`, `TEST_IMAGE_DIR`) to match your local setup.

---

## 📦 Required Python Packages

Create a conda environment (recommended) and install the following:

```bash
conda create -n dl_project python=3.10 -y
conda activate dl_project

pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn scikit-learn tqdm pillow
pip install grad-cam     # for Grad‑CAM visualization
```

### Optional (if you want to run inside the notebooks)
```bash
conda install ipykernel
python -m ipykernel install --user --name dl_project --display-name "dl_project"
```

---

## 🧪 Data Preparation (Fixed Split)

Both notebooks contain a **data cleaning section** that:

- Reads the HAM10000 metadata and the official test set ground truth.
- Performs a **stratified 80/20 train/validation split** (random seed = 42).
- Creates `train_seed42.csv`, `val_seed42.csv`, and `test_school.csv` inside `data/splits/` (or `data/splits_local/` for ResNet).
- Verifies that all image paths exist.

**You do not need to run the data cleaning separately** – it is already embedded as the first cells of each notebook.

> ⚠️ Make sure to adjust the dataset paths inside the first cell of each notebook to point to your local storage.

---

## 🚀 How to Train the Models
Please remember use the same train, validation, and test CSV files for all experiments.
### 1️⃣ Custom CNN Models (Baseline & Advanced)

Open `CNN Model.ipynb` and run the cells sequentially.

- **Baseline CNN**  
  No data augmentation, no class weights, no regularisation.  
  Trained for **15 epochs** with Adam (lr=1e-3), batch size 64, input size 64×64.  
  → Results are saved as `metrics_curves_fig1.png`, `confusion_matrix_fig2.png`, and the model weights `baseline_cnn_model.pth`.

- **Advanced CNN**  
  Adds data augmentation (random flip, rotation), class‑weighted loss, BatchNorm, Dropout, and higher resolution (128×128).  
  → The best model (by validation macro‑F1) is saved as `best_model.pth`.

### 2️⃣ ResNet‑18 Transfer Learning Experiments

Open `Resnet18 Model.ipynb` and run the cells for the desired configuration.  
The notebook contains three separate cells for **Exp‑A**, **Exp‑B**, and **Exp‑C**. Each cell is self‑contained – you can run one at a time.

| Experiment | Data Augmentation | Class Weights | Regularisation | Saved model file |
|----------------|----------------------------|-----------------------|--------------------|----------------------|
| Exp‑A      | ❌ No             | ❌ No          | ❌ No             | `best_resnet18_Exp_A.pth` |
| Exp‑B      | ✅ Yes (flip, rotation, colour jitter) | ❌ No | ❌ No | `best_resnet18_Exp_B.pth` |
| Exp‑C      | ✅ Yes            | ✅ Yes (inverse‑frequency) | ❌ No | `best_resnet18_Exp_C.pth` |

**Training hyperparameters (all ResNet‑18 experiments):**

- Epochs: 15
- Batch size: 32
- Learning rate: 1e-4 (Adam)
- Input size: 224×224
- Pre‑trained weights: `ResNet18_Weights.IMAGENET1K_V1`
- Full fine‑tuning (all layers updated)

After training, the notebook automatically:
- Plots training/validation loss, accuracy, and macro‑F1 curves.
- Shows the confusion matrix on the test set.
- Saves the results as `training_curves_Exp_X.png`, `confusion_matrix_Exp_X.png`, and `results_Exp_X.txt`.

---

## 📊 How to Evaluate a Model

Evaluation is performed **inside the training notebook** – after the training loop, the cell runs inference on the test set (`test_school.csv`) and prints:

- Accuracy, macro‑F1, macro‑precision, macro‑recall
- Per‑class classification report (precision, recall, F1‑score)
- Confusion matrix (displayed and saved as `.png`)

To evaluate a **pre‑trained model**, you can:

1. Load the saved weights (e.g., `best_resnet18_Exp_B.pth`).
2. Run only the “Test Set Evaluation” section of the notebook.

Example code snippet:

```python
model.load_state_dict(torch.load("best_resnet18_Exp_B.pth"))
test_loss, test_acc, test_f1, test_preds, test_labels = validate(model, test_loader, criterion, DEVICE)
```

---

## 🖼️ Model Interpretation – Grad‑CAM & Error Analysis

The last cell of `Resnet18 Model.ipynb` contains a complete **Grad‑CAM and success/failure analysis** for **Exp‑B** (the best performing model).

It does the following:

- Runs inference on the test set and records predictions & confidence scores.
- Prints the confusion matrix and identifies the most confusing class pair.
- Automatically selects **3 correct** and **3 incorrect** samples with the highest confidence.
- For each selected sample, generates a **side‑by‑side figure**:
  - **Left:** original resized image (224×224)
  - **Right:** Grad‑CAM heatmap overlay showing the regions the model focused on.
- Saves all figures in the `gradcam_output/` folder.

To run Grad‑CAM, make sure `grad-cam` is installed:

```bash
pip install grad-cam
```

Then execute the last cell of the notebook. The output folder will contain images named like:

- `{image_id}_correct_compare.png`
- `{image_id}_wrong_compare.png`

These can be directly used in the project report.

---

## 🔁 How to Reproduce the Reported Results

1. **Clone or download** this repository and ensure the dataset folders are placed correctly (update paths in the notebooks if necessary).
2. **Create the conda environment** and install all dependencies.
3. **Open `Resnet18 Model.ipynb`** and run the **data cleaning cell** once. This will create the fixed splits (train/val/test) inside `data/splits_local/`.
4. **Run the training cells** for Exp‑A, Exp‑B, and Exp‑C **one after another**. Each run takes about 1‑1.5 h on an M1 Mac (MPS).  
   - The results (curves, confusion matrices, text files) will be saved automatically.
   - The best model weights will be stored as `.pth` files.
5. **Evaluate the models** – already done inside each training cell.
6. **Run the Grad‑CAM cell** for Exp‑B to obtain the interpretation figures.

Because the random seed is fixed to **42** and the data split is deterministic, all numerical results should be **exactly reproducible** on the same hardware and software environment.

---

## 📝 Notes

- All code uses **MPS** acceleration on Apple Silicon. If you are on a different platform, change `DEVICE = torch.device("mps")` to `"cuda"` or `"cpu"`.
- The normalisation means and standard deviations for ResNet‑18 are the standard ImageNet values; for the custom CNNs they are computed from the training set.

