# RoadXAI — Output Gallery and Interpretation

> This document describes the visible outputs of the RoadXAI Streamlit application and one generated inspection report. It focuses on results, visualizations, measurements, reporting, and interpretation rather than re-documenting the system architecture.
>
> **Important:** The screenshots and report are a sample run, not independent validation of model performance. RoadXAI is an image-based screening and decision-support tool; its results should be checked by a qualified person before engineering or maintenance decisions.

## Contents

- [1. User input and inspection controls](#1-user-input-and-inspection-controls)
- [2. Road inspection overview](#2-road-inspection-overview)
- [3. Defect measurements](#3-defect-measurements)
- [4. Segmentation probability visualizations](#4-segmentation-probability-visualizations)
- [5. Explainable AI: Grad-CAM](#5-explainable-ai-grad-cam)
- [6. Interactive 3D inspection](#6-interactive-3d-inspection)
- [7. Report generation and exported report](#7-report-generation-and-exported-report)
- [8. Interpreting the displayed metrics](#8-interpreting-the-displayed-metrics)
- [9. Limitations and validation notes](#9-limitations-and-validation-notes)
- [10. Suggested GitHub placement](#10-suggested-github-placement)

## 1. User input and inspection controls

The interface begins with an image upload area and sidebar controls. The captured screen shows a segmentation threshold of `0.30` and a model image size selection of `256`, with other available sizes of `384`, `512`, `640`, and `768`. These are the settings visible in this screenshot; this document does not assume how every setting affects internal processing without checking the implementation.

![RoadXAI inference settings and upload screen](screenshots/app/01-inference-settings-and-upload.png)

*Figure 1. Inference settings and upload screen.*

The sample road image used in the captured run is shown below. It depicts a highly cracked, dry-looking surface. The image itself is included for context; visual appearance alone does not establish the model's correctness.

![Sample road image used in the captured run](screenshots/app/02-sample-road-image.jpg)

*Figure 2. Sample input image.*

The 3D Inspection sidebar exposes an analysis-mode selector, a visual-depth scale, and toggles for defect markers and defect boundaries.

![3D Inspection controls](screenshots/app/03-3d-inspection-controls.png)

*Figure 3. Interactive inspection controls.*

## 2. Road inspection overview

The inspection view presents the original image beside a version annotated with detected defect locations. The displayed summary reports:

- **Detected defects:** 95
- **Affected area:** 1.05%
- **Road-health score:** 98.95/100
- **Model confidence:** 30.34%

These are the values shown together in this UI view. A separate Advanced Analysis view and the generated PDF report show a defect-area figure of **1.38%**. Because the captured outputs disagree on affected/defect area, the discrepancy should be investigated before one figure is presented as the canonical value.

![Road inspection overview and summary metrics](screenshots/inspection/04-road-inspection-overview.png)

*Figure 4. Original image, defect-marked image, and inspection summary.*

The app also shows a condition label of **EXCELLENT** and advises field inspection before maintenance decisions. This should be described as the application's generated assessment, not as a verified engineering conclusion.

## 3. Defect measurements

Detected regions are presented as individually expandable records. The visible example for Defect 1 includes:

| Field | Displayed value | Interpretation |
|---|---:|---|
| Area | 277.50 px² | Number of pixels assigned to the region, expressed as area in image-pixel units |
| Length | 40.70 px | Estimated region length in pixels |
| Width | 13.77 px | Estimated region width in pixels |
| Centroid X | 182.43 | Horizontal image-coordinate location |
| Centroid Y | 549.97 | Vertical image-coordinate location |
| Severity | LOW | Application-assigned severity category |
| Score | 0.47 in the expanded UI | Application-displayed defect score; its formula is not established by the screenshot alone |

![Defect measurements and severity details](screenshots/inspection/05-defect-measurements-and-severity.png)

*Figure 5. Expandable defect records with area, dimensions, centroid coordinates, and severity.*

The exported PDF lists measurements for all 95 regions, ordered by defect ID. It explicitly says the measurements are in pixels because no validated pixel-to-meter calibration was supplied. Centroids are image coordinates, not geographic coordinates.

## 4. Segmentation probability visualizations

The Advanced Analysis view displays three complementary visualizations:

1. **Probability intensity:** a grayscale view of the model's pixel-level response.
2. **Probability heatmap:** a color map where brighter/hotter regions represent higher displayed defect probability.
3. **Probability over the original image:** the response map overlaid on the input for spatial context.

The same panel reports a defect area of **1.38%**, confidence of **30.34%**, and **8,144 defect pixels**. The displayed note says the segmentation threshold is `0.10` in this view, while the initial sidebar screenshot shows `0.30`. These may be different settings or states captured at different points; the screenshots alone do not establish why they differ.

![Advanced analysis probability views](screenshots/analysis/06-defect-probability-heatmaps.png)

*Figure 6. Probability intensity, heatmap, and overlay.*

A probability visualization is not the same thing as a calibrated probability of real-world damage. Its meaning depends on the model, preprocessing, thresholding, and calibration procedure. Avoid describing heatmap intensity as a verified physical defect severity unless that relationship has been validated.

## 5. Explainable AI: Grad-CAM

The application shows a Grad-CAM attention image and an overlay on the road image. The captured panel reports:

- **Mean attention:** 0.001
- **Maximum attention:** 1.000
- **Pixels above 0.5:** 368

![Grad-CAM attention map and overlay](screenshots/analysis/07-gradcam-attention-and-overlay.png)

*Figure 7. Grad-CAM attention and overlay outputs.*

Grad-CAM is a visualization intended to indicate image regions that influenced a model's target output. It is not, by itself, proof that the model reasoned correctly, nor does it prove that every highlighted region is a real defect. The precise target layer, target score, and model component used for this map should be documented from the implementation if a deeper explanation is required.

## 6. Interactive 3D inspection

The interactive 3D view projects the road image onto a surface and displays detected locations and boundary-like structures in a 3D plotting space. Multiple captured angles show how the display changes with viewpoint.

![3D inspection view — angle 1](screenshots/3d/09-3d-inspection-angle-1.png)

*Figure 8. 3D inspection view, angle 1.*

![3D inspection view — angle 2](screenshots/3d/10-3d-inspection-angle-2.png)

*Figure 9. 3D inspection view, angle 2.*

![3D inspection view — angle 3](screenshots/3d/11-3d-inspection-angle-3.png)

*Figure 10. 3D inspection view, angle 3.*

The report explicitly warns that the 3D vertical dimension is **visualization geometry unless external depth data is available**. Accordingly, the plotted heights should not be described as measured pothole depth or real-world road elevation. The display is useful for spatializing detected image regions, but physical depth claims require validated depth sensing or calibration.

The 3D overlay's legend identifies road surface, damage boundary, and defect locations. The summary box in these captures rounds road health to `99.0/100`, labels the condition Excellent, displays confidence as `30.3%`, and reports 95 detected regions. The rounded health value is consistent with `98.95/100` when rounded to one decimal place.

## 7. Report generation and exported report

The app includes an **Inspection Report** section and a **Generate RoadXAI Report** button. The UI says it can generate JSON, TXT, and PDF versions of the current assessment.

![Inspection report generation controls](screenshots/inspection/08-inspection-report-export.png)

*Figure 11. Report-generation interface.*

A PDF from the captured run is included at [`docs/reports/roadxai-sample-inspection-report.pdf`](reports/roadxai-sample-inspection-report.pdf). It contains seven pages, including:

- Executive summary and image-based inspection caveat
- Defect detection summary
- A table of 95 defect measurements
- Severity assessment for the 95 detected regions
- Overall road-health result and recommendation
- Repair-cost estimate section
- Explainable-AI method label (Grad-CAM)

The report identifies the run as `ROADXAI-20261022062935674840`, generated on 22 October 2026, and references the sample image filename. Keep the included report only if you are comfortable publishing this sample output in the public repository.

## 8. Interpreting the displayed metrics

| Metric | What the captured output says | How to explain it responsibly |
|---|---|---|
| Detected defects | 95 regions | Count of regions produced by the current image-analysis pipeline; not necessarily 95 independently verified physical defects |
| Affected/defect area | 1.05% in inspection summary; 1.38% in Advanced Analysis and PDF | A mismatch exists across captured views; reconcile before quoting a single value |
| Defect pixels | 8,144 in Advanced Analysis | Count displayed by that panel; exact mask resolution and counting method should be confirmed in code |
| Model confidence | 30.34% | Application-reported confidence indicator; its definition and calibration are not evident from screenshots alone |
| Road-health score | 98.95/100; rounded to 99.0/100 in 3D view | Application-generated score, not a validated civil-engineering rating unless its formula has been tested against ground truth |
| Severity | All 95 are shown as Low in the PDF | Current output distribution for this sample, not proof that all surface damage is low-risk in the field |
| Dimensions | px and px² | Image-space measurements; physical units require valid pixel-to-distance calibration |
| Repair area/cost | 0.00 m² and INR 0.00/m², cost INR 0.00 in the PDF | This is not a meaningful real-world estimate without a calibrated physical area and a configured, justified repair rate |
| Grad-CAM | Attention visualization generated | Helps inspect model focus; not a correctness guarantee or causal explanation |

### Why the confidence and health score should not be conflated

The sample output reports **30.34% model confidence** and **98.95/100 road health**. These appear to be separate indicators: one is labelled as confidence and the other as a road-health score. The screenshots and report do not define their formulas, so the document should not infer that one is calculated directly from the other. A high health score alongside many detected regions is worth reviewing against the scoring logic and labelled validation examples.

## 9. Limitations and validation notes

Before treating these outputs as engineering evidence, the following should be checked:

1. **Resolve the area mismatch.** The overview shows 1.05%, while Advanced Analysis and the PDF show 1.38%.
2. **Document the confidence definition.** State exactly what the 30.34% indicator measures and whether it is calibrated against held-out data.
3. **Validate the health-score formula.** Explain how region count, affected area, severity, and confidence influence the final score, if at all.
4. **Validate severity labels.** Compare automated labels with human-reviewed ground truth and describe the thresholding rules.
5. **Do not claim physical dimensions without calibration.** Pixel measurements are not metres; the current PDF explicitly notes no calibration is available.
6. **Do not claim physical depth from the 3D plot.** The report says the vertical dimension is visualization geometry without external depth data.
7. **Treat the zero repair estimate cautiously.** A zero square-metre estimate and zero cost may reflect missing calibration or a default rate, rather than no repair need.
8. **Treat Grad-CAM as an aid, not proof.** Validate the highlighted regions and document which model output is being explained.
9. **Use field inspection for decisions.** The app's own report says results should be confirmed before engineering or maintenance decisions.

These are documentation and validation observations based on the captured UI and sample PDF; they are not a claim that the underlying implementation necessarily contains a bug. Source-code review and controlled tests are needed to determine the cause of each discrepancy.

## 10. Suggested GitHub placement

Place this file at `docs/ROADXAI_OUTPUTS.md` and keep screenshots and the sample PDF in the relative paths shown in the Markdown. GitHub renders the images directly from the repository, so no external image hosting is required.

Suggested structure:

```text
RoadXAI/
├── docs/
│   ├── ROADXAI_OUTPUTS.md
│   ├── screenshots/
│   │   ├── app/
│   │   ├── inspection/
│   │   ├── analysis/
│   │   └── 3d/
│   └── reports/
│       └── roadxai-sample-inspection-report.pdf
└── README.md
```

From the repository README, link to this page with:

```markdown
[Explore the output gallery and interpretation](docs/ROADXAI_OUTPUTS.md)
```

---

*Documentation scope: captured application outputs and one generated sample report. System architecture is intentionally outside the scope of this page.*
