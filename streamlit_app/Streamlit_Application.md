# RoadXAI — Streamlit Application

The Streamlit application is the user-facing interface of RoadXAI. It connects image upload, model inference, engineering analysis, explainable AI, interactive 3D visualization, and report generation into one interactive workflow.

---

## 1. Streamlit Application — One View

```mermaid
flowchart TB
    A["RoadXAI Streamlit App"]

    A --> B["Road Inspection"]
    A --> C["Advanced Analysis"]

    B --> D["Image Upload"]
    B --> E["Segmentation Result"]
    B --> F["Inspection Summary"]
    B --> G["Defect Measurements"]

    C --> H["Probability Maps"]
    C --> I["Engineering Analysis"]
    C --> J["Grad-CAM"]
    C --> K["Interactive 3D"]
    C --> L["Road Decision Summary"]
    C --> M["Report Generation"]

    D --> N["Road Image"]
    N --> O["CNN Model"]
    O --> P["Prediction Mask"]
    P --> E
    P --> I
    P --> J
    P --> K
    I --> F
    I --> G
    I --> L
    E --> M
    I --> M
    J --> M
    K --> M
```

---

## 2. End-to-End Application Workflow

```mermaid
flowchart LR
    A["Upload Road Image"] --> B["Load / Validate Image"]
    B --> C["Load Trained Model"]
    C --> D["Preprocess Image"]
    D --> E["CNN Inference"]
    E --> F["Logits"]

    F --> G["Sigmoid"]
    G --> H["Probability Map"]
    H --> I["Threshold"]
    I --> J["Binary Prediction Mask"]

    J --> K["Engineering Analysis"]
    J --> L["Visualization"]
    J --> M["XAI"]

    K --> N["Road Health / Severity / Measurements"]
    L --> O["3D Inspection"]
    M --> P["Grad-CAM"]

    N --> Q["RoadXAI Report"]
    O --> Q
    P --> Q
    J --> Q

    Q --> R["JSON"]
    Q --> S["TXT"]
    Q --> T["PDF"]
```

---

## 3. User Interface Structure

```mermaid
flowchart TB
    A["RoadXAI"] --> B["Sidebar"]
    A --> C["Main Workspace"]

    B --> D["Inference Settings"]
    B --> E["Segmentation Threshold"]
    B --> F["Model Image Size"]

    B --> G["3D Inspection"]
    G --> H["Analysis Mode"]
    G --> I["Visual Depth Scale"]
    G --> J["Defect Markers"]
    G --> K["Defect Boundaries"]

    C --> L["Road Inspection Tab"]
    C --> M["Advanced Analysis Tab"]
```

---

## 4. Road Inspection View

```mermaid
flowchart TB
    A["Road Inspection"]

    A --> B["Original Road Image"]
    A --> C["Numbered Defect Overlay"]
    A --> D["Inspection Summary"]
    A --> E["Road Condition"]
    A --> F["Field Inspection Status"]
    A --> G["Detected Defects"]

    D --> D1["Detected Defects"]
    D --> D2["Affected Area"]
    D --> D3["Health Score"]
    D --> D4["Model Confidence"]

    G --> G1["Area"]
    G --> G2["Length"]
    G --> G3["Width"]
    G --> G4["Centroid"]
    G --> G5["Severity"]
    G --> G6["Explanation"]
```

The main inspection view is designed as the primary screening interface.

---

## 5. Advanced Analysis View

```mermaid
flowchart TB
    A["Advanced Analysis"]

    A --> B["Inference Summary"]
    A --> C["Defect Probability"]
    A --> D["Engineering Analysis"]
    A --> E["Explainable AI"]
    A --> F["Interactive 3D Visualization"]
    A --> G["Road Decision Summary"]
    A --> H["Inspection Report"]

    B --> B1["Defect Area"]
    B --> B2["Confidence"]
    B --> B3["Defect Pixels"]

    C --> C1["Probability Intensity"]
    C --> C2["Probability Heatmap"]
    C --> C3["Probability Overlay"]

    D --> D1["Defect Count"]
    D --> D2["Affected Area"]
    D --> D3["Condition"]
    D --> D4["Priority"]

    E --> E1["Grad-CAM"]
    F --> F1["3D Surface"]
    G --> G1["Condition"]
    G --> G2["Health Score"]
    G --> G3["Maintenance Priority"]
    G --> G4["Field Inspection"]
```

---

## 6. Image Upload and Preprocessing

```mermaid
flowchart LR
    A["Streamlit UploadedFile"] --> B["uploaded_file_to_image()"]
    B --> C["PIL RGB Image"]
    C --> D["image_to_numpy()"]
    C --> E["preprocess_image()"]

    E --> F["Resize"]
    F --> G["RGB → Float32"]
    G --> H["Normalize /255"]
    H --> I["CHW"]
    I --> J["Batch Dimension"]
    J --> K["[1,3,H,W]"]
```

Supported upload formats:

```text
JPG
JPEG
PNG
BMP
WEBP
```

---

## 7. Model Loading

```mermaid
flowchart LR
    A["CRACK500 best.pt"] --> B["load_roadxai_model()"]
    B --> C["build_model(base_channels=16)"]
    C --> D["load_model()"]
    D --> E["CPU Model"]
    E --> F["Streamlit Cached Resource"]
```

The trained checkpoint is expected at:

```text
checkpoints/crack500/best.pt
```

The application caches the loaded model using Streamlit's resource cache so it is not repeatedly loaded for every interaction.

---

## 8. Inference Pipeline

```mermaid
flowchart TB
    A["RGB Image"] --> B["preprocess_image()"]
    B --> C["Input Tensor [1,3,H,W]"]

    C --> D["RoadXAI CNN"]
    D --> E["Segmentation Logits"]

    E --> F["Sigmoid"]
    F --> G["Probability Map"]

    G --> H{"Probability >= Threshold?"}

    H -->|Yes| I["Defect Pixel = 1"]
    H -->|No| J["Background Pixel = 0"]

    I --> K["Prediction Mask"]
    J --> K

    G --> L["Confidence"]
    K --> L
```

The application currently performs batch-size-one inference for an uploaded image.

---

## 9. Prediction Outputs

```mermaid
flowchart TB
    A["Model Output"] --> B["Prediction Mask"]
    A --> C["Probability Map"]
    A --> D["Confidence"]

    B --> E["Defect Localization"]
    C --> F["Probability Visualization"]
    D --> G["Dashboard Metric"]

    B --> H["Engineering Analysis"]
    B --> I["3D Visualization"]
    B --> J["Report Generation"]
```

Confidence is calculated from the predicted probabilities. When defect pixels exist, it is the mean probability over predicted defect pixels; otherwise it is based on the inverse mean probability.

---

## 10. Segmentation Visualization

```mermaid
flowchart LR
    A["Original Image"] --> B["Prediction Mask"]
    B --> C["Resize to Original Image"]
    A --> D["Mask Overlay"]
    C --> D

    D --> E["Numbered Defect Overlay"]
```

Detected connected defect regions are assigned numbers. These numbers correspond to the defect measurements displayed in the Road Inspection view.

---

## 11. Dashboard Metrics

```mermaid
flowchart TB
    A["InferenceOutput"] --> B["build_dashboard_summary()"]

    B --> C["Confidence"]
    B --> D["Defect Percentage"]
    B --> E["Defect Pixels"]
    B --> F["Image Height"]
    B --> G["Image Width"]

    C --> H["Streamlit Metrics"]
    D --> H
    E --> H
```

The main inspection dashboard additionally displays:

| Metric | Meaning |
|---|---|
| Detected Defects | Number of detected defect regions |
| Affected Area | Percentage of analyzed road pixels affected |
| Health Score | Image-based road-health indicator |
| Model Confidence | Model confidence indicator |

---

## 12. Engineering Analysis Integration

```mermaid
flowchart LR
    A["Prediction Mask"] --> B["analyze_road()"]
    C["Road Area"] --> B
    D["No Calibration"] --> B
    E["No Repair Rate"] --> B

    B --> F["Defect Measurements"]
    B --> G["Severity"]
    B --> H["Road Health"]
    B --> I["Repair Cost"]
```

The application calls engineering analysis with:

```text
calibration = None
repair_rate_per_m2 = None
currency = INR
min_area_pixels = 10.0
```

Therefore, the current application reports image/pixel measurements and does not produce physically calibrated dimensions or a repair-cost estimate from the default Streamlit workflow.

---

## 13. Defect Records and Maintenance Priority

```mermaid
flowchart TB
    A["Engineering Result"] --> B["build_defect_records()"]

    B --> C["Area"]
    B --> D["Length"]
    B --> E["Width"]
    B --> F["Centroid"]
    B --> G["Severity"]
    B --> H["Severity Score"]

    G --> I["Priority"]
    H --> I

    I --> J["Critical"]
    I --> K["High"]
    I --> L["Moderate"]
    I --> M["Low"]

    C --> N["Sorted Defect Records"]
    D --> N
    E --> N
    F --> N
    I --> N
```

Defect records are sorted by severity score and then defect area.

---

## 14. Maintenance Priority Logic

```mermaid
flowchart TB
    A["Defect Records + Affected Area"]

    A --> B{"Critical Defect?"}
    B -->|Yes| C["Critical Priority"]
    B -->|No| D{"High Defect?"}

    D -->|Yes| E["High Priority"]
    D -->|No| F{"Affected Area >= 25%?"}

    F -->|Yes| E
    F -->|No| G{"Affected Area >= 10%?"}

    G -->|Yes| H["Moderate Priority"]
    G -->|No| I{"Any Defect?"}

    I -->|Yes| H
    I -->|No| J["Low Priority"]
```

---

## 15. Field Inspection Trigger

```mermaid
flowchart TB
    A["Defect Records"] --> D{"Serious Defect?"}
    B["Road Condition"] --> E{"Poor / Critical?"}
    C["Confidence"] --> F{"Confidence < 0.65?"}

    D --> G["Inspection Required"]
    E --> G
    F --> G

    D -->|No| H["No Serious Defect"]
    E -->|No| I["Condition Not Poor/Critical"]
    F -->|No| J["Confidence ≥ 0.65"]

    H --> K["No Immediate Trigger"]
    I --> K
    J --> K
```

Field inspection is prioritized when there is a high/critical defect, poor/critical road condition, or model confidence below `0.65`.

---

## 16. Probability Visualization

```mermaid
flowchart LR
    A["Probability Map [0,1]"] --> B["Scale to 0–255"]
    B --> C["Grayscale"]
    B --> D["Color Heatmap"]

    D --> E["Blend With Original"]
    A --> E

    C --> F["Probability Intensity"]
    D --> G["Probability Heatmap"]
    E --> H["Probability Overlay"]
```

The Advanced Analysis view displays three probability representations:

1. Probability intensity
2. Probability heatmap
3. Probability over the original image

---

## 17. Grad-CAM Workflow

```mermaid
flowchart TB
    A["Inference Input Tensor"] --> B["prepare_xai_tensor()"]
    B --> C["Fresh CPU Tensor"]

    C --> D["Default Target Layer"]
    C --> E["RoadXAI Model"]

    E --> F["Grad-CAM"]
    D --> F

    F --> G["Raw Heatmap"]
    G --> H["Normalize [0,1]"]
    H --> I["Heatmap"]
    I --> J["Overlay With Image"]

    J --> K["Streamlit XAI View"]
```

The current application generates **Grad-CAM** using the default target layer and target class `0`.

---

## 18. XAI Display

```mermaid
flowchart LR
    A["Grad-CAM Result"] --> B["Attention Heatmap"]
    A --> C["Heatmap Overlay"]

    B --> D["Mean Attention"]
    B --> E["Maximum Attention"]
    B --> F["Pixels >= 0.5"]

    C --> G["Visual Explanation"]
```

The UI also displays the selected target layer.

The explanation represents model attention/contribution behavior; it is not a physical measurement of damage.

---

## 19. Interactive 3D Visualization

```mermaid
flowchart TB
    A["Prediction Mask"] --> B["generate_3d()"]
    C["Original Image"] --> B
    D["Probability Map"] --> B
    E["Engineering Result"] --> B
    F["XAI Heatmap"] --> B

    B --> G["3D Visualization Module"]
    G --> H["Heightmap"]
    G --> I["Mesh"]
    G --> J["Plotly Figure"]

    H --> K["Visual Depth"]
    I --> L["Vertices / Faces"]
    J --> M["Interactive Streamlit Plot"]
```

The application supports:

```text
Inspection
Road Surface
Severity
AI Confidence
AI Attention
```

The visual depth control changes visualization exaggeration only.

---

## 20. 3D Controls

```mermaid
flowchart LR
    A["3D Inspection Sidebar"]

    A --> B["Analysis Mode"]
    A --> C["Visual Depth Scale"]
    A --> D["Show Defect Markers"]
    A --> E["Show Defect Boundaries"]

    B --> F["3D Rendering"]
    C --> F
    D --> F
    E --> F
```

The visualization uses:

```text
max_depth = visual_depth
pixel_size = 1.0
vertical_exaggeration = visual_depth
```

Physical depth is not inferred from the RGB image.

---

## 21. Session-State Architecture

```mermaid
flowchart TB
    A["Uploaded Image"] --> B["Image Key"]

    B --> C["analysis_image_key"]

    C --> D["xai_result"]
    C --> E["visualization_result"]
    C --> F["report"]

    D --> G["xai_image_key"]
    E --> H["visualization_key"]
    F --> I["report_image_key"]

    B --> J["Clear stale results when image changes"]
```

The image key combines the uploaded filename, a SHA-256 digest of the RGB image data, and image shape.

This prevents previous XAI, 3D, and report results from being reused for a different uploaded image.

---

## 22. Streamlit Resource Caching

```mermaid
flowchart LR
    A["Application Rerun"] --> B["load_roadxai_model()"]
    B --> C{"Cached?"}

    C -->|Yes| D["Reuse Model"]
    C -->|No| E["Build + Load Checkpoint"]
    E --> F["Cache Resource"]
    F --> D
```

The model-loading function is decorated with Streamlit's resource cache.

---

## 23. Report Generation Integration

```mermaid
flowchart TB
    A["Current Image"] --> E["generate_report()"]
    B["Inference"] --> E
    C["Engineering Result"] --> E
    D["XAI / 3D Results"] --> E

    E --> F["RoadXAIReport"]

    F --> G["save_json_report()"]
    F --> H["save_text_report()"]
    F --> I["generate_pdf_report()"]

    G --> J["Download JSON"]
    H --> K["Download TXT"]
    I --> L["Download PDF"]

    F --> M["Report Preview"]
```

The report is generated only when the user selects **Generate RoadXAI Report**.

The report files are written to:

```text
reports/
```

using the uploaded image filename as the report base name.

---

## 24. Download Architecture

```mermaid
flowchart LR
    A["Generated Report"]

    A --> B["JSON"]
    A --> C["TXT"]
    A --> D["PDF"]

    B --> E["Download JSON"]
    C --> F["Download TXT"]
    D --> G["Download PDF"]
```

All three report formats are exposed through Streamlit download buttons.

---

## 25. Utility Layer

```mermaid
flowchart TB
    A["Streamlit Utilities"]

    A --> B["Image Conversion"]
    A --> C["Mask Conversion"]
    A --> D["Visualization Helpers"]
    A --> E["Formatting"]
    A --> F["Session State"]
    A --> G["Uploaded File Handling"]
    A --> H["Streamlit Messages"]

    B --> B1["numpy_to_pil"]
    B --> B2["pil_to_numpy"]

    C --> C1["mask_to_uint8"]
    C --> C2["probability_to_uint8"]

    D --> D1["create_side_by_side"]
    D --> D2["normalize_display_image"]

    E --> E1["format_percentage"]
    E --> E2["format_confidence"]
    E --> E3["format_area"]
    E --> E4["format_length"]

    F --> F1["initialize_streamlit_state"]
    F --> F2["reset_streamlit_state"]

    G --> G1["save_uploaded_file"]
    G --> G2["create_download_bytes"]

    H --> H1["display_error"]
    H --> H2["display_success"]
    H --> H3["display_info"]
```

---

## 26. Application Module Responsibilities

| Component | Responsibility |
|---|---|
| `app.py` | Main Streamlit application and end-to-end UI workflow |
| `module.py` | Image handling, preprocessing, inference, UI components, XAI/3D helpers |
| `utils.py` | Display conversion, formatting, state, upload and Streamlit utility functions |
| `__init__.py` | Package metadata |

---

## 27. Public Application Components

```mermaid
flowchart TB
    A["streamlit_app.module"]

    A --> B["InferenceOutput"]
    A --> C["load_image()"]
    A --> D["image_to_numpy()"]
    A --> E["preprocess_image()"]
    A --> F["postprocess_prediction()"]
    A --> G["run_inference()"]
    A --> H["resize_mask_to_image()"]
    A --> I["create_mask_overlay()"]
    A --> J["calculate_defect_percentage()"]
    A --> K["build_dashboard_summary()"]
    A --> L["display_image()"]
    A --> M["display_inference_summary()"]
    A --> N["create_upload_section()"]
    A --> O["create_app_header()"]
    A --> P["create_settings_sidebar()"]
    A --> Q["get_device()"]
    A --> R["load_model_checkpoint()"]
```

---

## 28. Main Application Dependencies

```mermaid
flowchart TB
    A["Streamlit App"]

    A --> B["CNN Model"]
    A --> C["Deployment"]
    A --> D["Engineering Analysis"]
    A --> E["Explainable AI"]
    A --> F["3D Visualization"]
    A --> G["Report Generation"]

    B --> H["Segmentation Model"]
    C --> I["Checkpoint Loading"]
    D --> J["Measurements / Severity / Health"]
    E --> K["Grad-CAM"]
    F --> L["Interactive Plotly Visualization"]
    G --> M["JSON / TXT / PDF"]
```

---

## 29. Application-Level Architecture

```mermaid
flowchart TB
    A["User"]

    A --> B["Streamlit UI"]

    B --> C["Upload Image"]
    B --> D["Inference Settings"]
    B --> E["3D Settings"]

    C --> F["Image Preprocessing"]
    D --> F
    F --> G["RoadXAI Model"]
    G --> H["Prediction"]

    H --> I["Road Inspection"]
    H --> J["Engineering Analysis"]
    H --> K["Probability Visualization"]
    H --> L["Grad-CAM"]
    H --> M["3D Visualization"]

    J --> N["Road Decision"]
    I --> N
    K --> N
    L --> N
    M --> N

    N --> O["Report Generation"]
    O --> P["JSON"]
    O --> Q["TXT"]
    O --> R["PDF"]
```

---

## 30. Research-Paper View

```mermaid
flowchart TB
    A["RoadXAI Interactive Application"]

    A --> B["Image Acquisition"]
    A --> C["AI Inference"]
    A --> D["Engineering Interpretation"]
    A --> E["Explainability"]
    A --> F["Visualization"]
    A --> G["Reporting"]

    B --> H["RGB Road Image"]
    C --> I["Segmentation Mask + Probability"]
    D --> J["Defect Geometry + Road Health"]
    E --> K["Grad-CAM"]
    F --> L["Interactive 3D Representation"]
    G --> M["Inspection Artifacts"]

    H --> N["Human-in-the-loop Road Inspection"]
    I --> N
    J --> N
    K --> N
    L --> N
    M --> N
```

### System role

The Streamlit layer functions as the **interactive integration layer** of RoadXAI. It connects the trained model and downstream analysis modules to a user-facing inspection workflow.

---

## 31. Scientific / Engineering Limitations

```mermaid
flowchart TB
    A["Streamlit Results"]

    A --> B["AI Prediction"]
    A --> C["Pixel Geometry"]
    A --> D["3D Visualization"]
    A --> E["Maintenance Decision"]

    B --> F["AI-assisted screening"]
    C --> G["Requires calibration for physical dimensions"]
    D --> H["Normalized visual depth"]
    E --> I["Requires appropriate field validation"]
```

Important limitations reflected by the application:

- The system is an AI-assisted road-inspection screening and decision-support interface.
- Physical metres and square metres require validated pixel-to-meter calibration.
- The 3D vertical dimension is visualization geometry, not physical pothole depth.
- Automated results should be validated through appropriate field inspection before engineering or maintenance decisions.
- The current Streamlit workflow loads the CRACK500 checkpoint and uses a binary segmentation output.
- The current application runs inference on CPU.
- The current application uses a single uploaded image at a time.

---

## 32. Final Architecture

```mermaid
flowchart TB
    A["RoadXAI Streamlit"]

    A --> B["Input"]
    A --> C["AI"]
    A --> D["Analysis"]
    A --> E["Explainability"]
    A --> F["Visualization"]
    A --> G["Reporting"]

    B --> B1["Road Image"]
    B --> B2["Threshold"]
    B --> B3["Image Size"]

    C --> C1["CNN Segmentation"]
    C1 --> C2["Probability Map"]
    C2 --> C3["Binary Mask"]

    D --> D1["Defect Measurements"]
    D --> D2["Severity"]
    D --> D3["Road Health"]
    D --> D4["Maintenance Priority"]

    E --> E1["Grad-CAM"]

    F --> F1["Probability Views"]
    F --> F2["Interactive 3D"]

    G --> G1["JSON"]
    G --> G2["TXT"]
    G --> G3["PDF"]

    C3 --> D
    C3 --> E
    C3 --> F
    D --> G
    E --> G
    F --> G
```

> **Scope note:** The Streamlit application is the user-facing orchestration layer. It integrates existing RoadXAI components into an interactive inspection workflow; the underlying model, engineering, XAI, 3D, and report-generation logic remains implemented in their respective modules.
