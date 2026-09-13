# RoadXAI

> AI-assisted road inspection — segmentation, measurement, severity, XAI, 3D visualization and reporting.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-CV-5C3EE8?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3D-3F4F75?logo=plotly&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Training-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-Research-lightgrey)

</p>

---

## What is RoadXAI?

RoadXAI converts road images into an interpretable inspection workflow.

```mermaid
flowchart LR
    A[Road Image] --> B[Preprocess]
    B --> C[U-Net Segmentation]
    C --> D[Defect Mask]

    D --> E[Engineering]
    D --> F[XAI]
    D --> G[3D Visualization]

    E --> H[Inspection Result]
    F --> H
    G --> H

    H --> I[JSON / TXT / PDF]
