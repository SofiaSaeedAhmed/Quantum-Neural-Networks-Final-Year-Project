# FYP
## Imports

**Required Libraries**

To run the project, make sure to install the necessary dependencies:

```bash
# PyTorch and Torchvision
pip install torch torchvision

# Quantum Libraries
pip install pennylane
pip install pennylane-qiskit

# Matplotlib for Visualization
pip install matplotlib
```

## Code files 
* _PQC1.py_: File containing PQC1 code  
* _PQC2.py_: File containing PQC2 code  
* _PQC3.py_: File containing PQC3 code  

**MNIST experiments:**    
* _MNIST 1a (PQC1)_: Without dropout, Adam optimiser  
* _MNIST 1 (PQC1)_: With dropout, Adam optimiser  
* _MNIST 1b (PQC1)_: With dropout, SGD optimiser

* _MNIST 2a (PQC2)_: Without dropout, Adam optimiser  
* _MNIST 2 (PQC2)_: With dropout, Adam optimiser  
* _MNIST 2b (PQC2)_: With dropout, SGD optimiser
  
* _MNIST 3a (PQC3)_: Without dropout, Adam optimiser  
* _MNIST 3 (PQC3)_: With dropout, Adam optimiser  
* _MNIST 3b (PQC3)_: With dropout, SGD optimiser  

**CIFAR-10 code files:**    
* _cifarconv2_: Model `C32C64FC256FC25Q5E3` — 2 conv layers, 2 FC layers, 5 quantum circuits, 3 entangling layers, LR: 0.001  
* _cifar2_: Model `C32C64C128C256FC256FC128FC20Q5` — 4 conv layers, 3 FC layers, 5 quantum circuits, 3 entangling layers, LR: 0.001  
* _concat4_: Model `C32C64C128C256FC256FC128FC20Q4` — 4 conv layers, 3 FC layers, 4 quantum circuits, 3 entangling layers, LR: 0.001  
* _concat2_: Model `C32C64C128C256FC256FC128FC20Q6` — 4 conv layers, 3 FC layers, 6 quantum circuits, 3 entangling layers, LR: 0.001  
* _concat3_: Model `C32C64C128C256FC256FC128FC20Q10` — 4 conv layers, 3 FC layers, 10 quantum circuits, 3 entangling layers, LR: 0.001  
* _concat4_: Model `C32C64C128C256FC256FC128FC20Q4E3` — 4 conv layers, 3 FC layers, 4 quantum circuits, 3 entangling layers, LR: 0.001  
* _cifar3_: Model `C32C64C128C256FC256FC128FC20Q4E5` — 4 conv layers, 3 FC layers, 4 quantum circuits, 5 entangling layers, LR: 0.001  
* _cifar1_: Model `C32C64C128C256FC256FC128FC20Q4E0` — 4 conv layers, 3 FC layers, 4 quantum circuits, 0 entangling layers, LR: 0.001  
* _cifar4_: Model `C32C64C128C256FC256FC128FC20Q4E10` — 4 conv layers, 3 FC layers, 4 quantum circuits, 10 entangling layers, LR: 0.001  
* _cifar5_: Model `C32C64C128C256FC256FC128FC20Q4E2` — 4 conv layers, 3 FC layers, 4 quantum circuits, 2 entangling layers, LR: 0.001  
* _cifar6_: Model `C32C64C128C256FC256FC128FC20Q4E3S` — 4 conv layers, 3 FC layers, 4 quantum circuits, 3 entangling layers, LR: 0.001  

**Learning rate tuning experiments:**  
* _cifarsoftlr_: Model `C32C64C128C256FC256FC128FC20Q4E3` — 4 quantum circuits, 3 entangling layers, LR: 0.005/0.001  
* _cifarsoftlr2_: Model `C32C64C128C256FC256FC128FC20Q4E3` — 4 quantum circuits, 3 entangling layers, LR: 0.005/0.001  
* _cifarsoftlr3_: Model `C32C64C128C256FC256FC128FC20Q4E3` — 4 quantum circuits, 3 entangling layers, LR: 0.005/0.003  
* _cifarsoftlr4_: Model `C32C64C128C256FC256FC128FC20Q4E3` — 4 quantum circuits, 3 entangling layers, LR: 0.0025/0.001  

**Supervised CIFAR variant:**  
* _cifarsuper_: Model `C32C64C128C256FC256FC128FC20Q4E3` — 4 quantum circuits, 3 entangling layers, LR: 0.005/0.003  

**Simple CIFAR models (no quantum layers):**  
* _Simplecifar1_: Model `C32C64C128C256FC256FC128FC20` — 4 conv layers, 3 FC layers, LR: 0.003  
* _Simplecifar2_: Model `C32C64C128C256FC256FC128FC20` — 4 conv layers, 3 FC layers, LR: 0.005  

**Medical MNIST experiments:**  
* _medicalquantum_: Model `C32C64C128C256FC256FC128FC20Q4E3` — 4 quantum circuits, 3 entangling layers, LR: 0.005/0.003  
* _medicalsimple_: Model `C32C64C128C256FC256FC128FC20Q4E3` — quantum model (no entangling details), LR: 0.005  

**UCF101 experiments (video):**  
* _cnnUCF_: Classical CNN baseline — LR: 0.005  
* _ucfqcnn_: Quantum CNN variant — 4 quantum circuits, 3 entangling layers, LR: 0.005/0.003  
* _lstmCNN_ucf_: Classical LSTM + CNN — LR: 0.0005  
* _QlstmCNN_ucf_: Quantum LSTM + CNN — 4 quantum circuits, 3 entangling layers, LR: 0.0005  

**Dataset preparation files:**  
* _dataset_: UCF101 — frame extraction and preparation  
* _dataset 2_: UCF101 — video shuffling and setup  

**Evaluation scripts:**  
* _evaluateMNIST_: Code to evaluate MNIST models  
* _evaluateLSTM_: Accuracy testing for all LSTM models  
* _Evaluate_medical_: Evaluation for Medical MNIST models  

**Graphing and visualisation:**  
* _graphs_: Generate all result graphs  
* _visualisationCIFAR_: Visualise CIFAR model predictions  
