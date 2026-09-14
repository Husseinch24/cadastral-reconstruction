# THESIS PROJECT BRIEFING FOR CLAUDE CODE
# Read this first before doing anything
# Masters Thesis — AI-Based Panoramic Cadastral Map Reconstruction
# Author: Hussein Chalhoub

## PROJECT GOAL
Automatically stitch 13 historical Lebanese cadastral blueprint maps into
a single panoramic image by:
1. Detecting parcel polygons on each map (done - Mask R-CNN)
2. Matching the same parcel across adjacent maps using its number (FAILING)
3. Computing a homography from matched parcels
4. Warping and blending all maps into one panorama (done for manual pairs)

## FOLDER STRUCTURE
```
Thesis/
├── venv_thesis/              <- Python virtual environment (activate first)
├── requirements.txt
├── output/                   <- OLD classical pipeline (Steps 1-6)
│   ├── preprocessed/         <- 13 cleaned maps: map_43_clean.png ... map_55_clean.png
│   ├── homographies/         <- manual homographies from drag-and-place tool
│   └── panorama/             <- 2 panoramas from manual alignment (WORKING)
└── new_pipeline/             <- NEW deep learning pipeline
    ├── src/
    │   ├── utils/data_gen.py         <- synthetic data generator (DONE)
    │   ├── step4_segmentation.py     <- Mask R-CNN training + inference (DONE)
    │   ├── step5_graph.py            <- parcel adjacency graph builder (DONE)
    │   └── step6_gnn_matching.py     <- GNN matching (PARTIAL - see below)
    ├── data/
    │   ├── synthetic/                <- 5700 training images + COCO annotations
    │   ├── predictions/              <- Mask R-CNN detections per map (JSON + overlay PNGs)
    │   ├── graphs/                   <- NetworkX graphs per map (GraphML + JSON)
    │   └── matches/                  <- GNN match results (poor quality)
    └── models/
        ├── segmentation/best_model.pth  <- trained Mask R-CNN weights
        └── gnn/best_gnn.pth             <- trained GNN weights (poor results)
```

## WHAT WORKS
1. Mask R-CNN detects parcels correctly - 11,414 parcels across 13 maps
2. Graph construction correctly captures parcel adjacency - 27,462 edges
3. Classical panorama from manual alignment - see output/panorama/
4. All preprocessing (Steps 1-3) - working

## WHAT FAILED AND WHY
1. SIFT matching - adjacent sheets have NO visual overlap by design
2. Tesseract OCR - <1% on handwritten Arabic-Indic on aged blueprints
3. EasyOCR (Arabic+English) - 0 numbers found, same root cause
4. GNN matching - only 0-3 matches per pair, insufficient training data
   (only 13 maps total, 19 known boundary parcels from CSV)

## THE CSV GROUND TRUTH (critical reference)
This is the ONLY source of truth linking boundary parcels to map pairs:

parcel_number  maps
2580           45, 47
2616           45, 46
2619           45, 46
2749           47, 48
2803           47, 48
2814           47, 48
2893           48, 49
3022           49, 50
3054           49, 50
3068           50, 51
3215           52, 53
3216           52, 54
3217           52, 55
3338           54, 55
3339           54, 55
3345           54, 55
3346           54, 55
3813           50, 51
3866           50, 51

## CURRENT TASK
Find parcel numbers written in handwritten Arabic-Indic numerals inside
parcel regions on the map images. Numbers are:
- 4 digits, all > 2000
- Written horizontally but at various rotation angles (cartographer wrote
  each number along the natural orientation of the parcel)
- NOT mirrored (earlier assumption was wrong)
- Located inside the parcel polygon detected by Mask R-CNN
- Size: approximately 40-80px tall at full map resolution

OCR approach tried: EasyOCR with Arabic - FAILED (not trained on handwriting)

Next approach to try: PaddleOCR
- Handles rotated text natively via its own text detection pipeline
- Better on degraded/handwritten documents than EasyOCR
- Install: pip install paddlepaddle paddleocr

## WHAT TO DO NEXT
1. Try PaddleOCR on parcel regions from map 45 and map 47
2. Check if it can find 2580 in both maps
3. If yes - write full pipeline to find all 19 boundary parcels
4. If no - try custom CRNN trained on digit crops from the maps

## TECHNICAL NOTES
- GPU: RTX 2070 Max-Q, 8GB VRAM, CUDA 13.0 (use cu124 wheels)
- Python 3.11.6, venv at: venv_thesis/
- Activate: .\venv_thesis\Scripts\Activate.ps1
- Run scripts from thesis root folder always
- Windows paths - use Path() not hardcoded strings
- num_workers=0 required on Windows with CUDA (multiprocessing bug)
- Maps are grayscale, 6000-9000px, stored as PNG

## MAPS PRESENT
Maps 43-55 (13 total). All at output/preprocessed/map_N_clean.png
Maps 43, 44 have NO boundary parcels in the CSV (isolated sheets)
Maps 49-51 are partially problematic (map 50 is incomplete from instructor)

## DO NOT TOUCH
- output/panorama/ - these are the final deliverable panoramas
- output/homographies/ - manual alignments that took hours
- new_pipeline/models/ - trained model weights
