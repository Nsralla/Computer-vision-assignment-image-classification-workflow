# Computer-Vision Assignment — 2025

End-to-end computer-vision workflow that covers data loading, image preprocessing, model training, and performance evaluation.  
The polished write-up lives in **docs/assignment-report.pdf**, while fully-execut­able code and experiments are inside **notebooks/** and **src/**.

---

## Repository layout
.
├── data/ # Raw / interim / processed image data

│ ├── raw/

│ ├── interim/

│ └── processed/

├── docs/

│ └── assignment-report.pdf # Comprehensive PDF report

├── notebooks/

│ └── cv_assignment.ipynb # Jupyter notebook with code & visuals

├── src/ # Re-usable modules

│ ├── init.py

│ ├── datamodule.py # PyTorch-Lightning DataModule

│ ├── model.py # CNN / ViT architecture

│ └── train.py # CLI entry-point (lightning CLI)

├── tests/ # Minimal unit tests (pytest)

├── requirements.txt # Exact package versions

└── README.md # You are here







---

## Quick start

1. **Clone**
2. 
   ```bash
   git clone https://github.com/<your-username>/computer-vision-assignment-2025.git
   cd computer-vision-assignment-2025
    ```  
2.Prepare environment

  ``` bash
  python -m venv .venv
  source .venv/bin/activate           # Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```
3.Download / organise data

Place raw images in data/raw/ following the folder structure described in the report (e.g. class_name/imagename.jpg).


4.Run the notebook

jupyter notebook notebooks/cv_assignment.ipynb or execute the full training pipeline from the command line:
  ``` bash
  python src/train.py --config configs/resnet18.yaml
  ```
5.Read the report

Open docs/assignment-report.pdf for methodology, results, and discussion.


### Features
1- Clear project structure suitable for iterative experimentation.

2- Reproducible environment — strict requirements.txt plus optional conda lockfile.

3- Modular code — reusable DataModule and model definitions.

4- Notebook + script parity — explore interactively or run headless.

5- Report-ready visuals — confusion matrices, ROC/PR curves, Grad-CAM heat-maps.

### How it works (high-level)
1-Data ingestion

Images are loaded with torchvision.datasets.ImageFolder, split train/val/test, and augmented (random crop, flip, colour jitter).

2- Modelling

Baselines: ResNet-18 and MobileNet-V3.
Training orchestrated by PyTorch-Lightning with early stopping and mixed-precision.

2- Evaluation

Accuracy, precision/recall, F1, per-class confusion matrices.
Grad-CAM used to interpret model focus areas.

4- Reporting

Key findings exported as figures and embedded in the PDF.
