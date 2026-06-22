# ProofShield AI v2

ProofShield AI v2 is a local-first multimodal AI-generated image risk estimator. It returns a calibrated probability, risk band, confidence, and model-derived branch explanations. It never labels an image "fake" and is not a provenance, identity, or legal-verification system.

## v2 architecture

```text
image bytes
  |- frozen OpenCLIP ViT-B/32 image encoder -> normalized visual embedding
  |- FFT/noise extractor -> spectral/high-frequency feature vector
  |- metadata summary -> weak auxiliary feature vector
  `- learned MLP fusion -> temperature calibration -> probability
```

The fusion classifier is `Linear(input, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 128) -> ReLU -> Linear(128, 1)`. It trains with BCE-with-logits (or optional focal loss), not a rule score. Frequency and metadata weights are jointly learned with the visual embedding; metadata is auxiliary and cannot independently decide an outcome.

The prior browser/backend heuristic scoring path has been removed from serving. `/analyze` only returns a score from a trained local checkpoint. Without a validated checkpoint, it returns `503` rather than inventing a score.

## Layout

```text
ml/clip_encoder.py  frozen OpenCLIP branch
ml/frequency.py     FFT, spectral entropy, residual/noise statistics
ml/metadata.py      private metadata summary and encoding
ml/fusion.py        MLP, feature normalizer, temperature scaler
ml/dataset.py       manifest dataset and corruption augmentation
ml/train.py         reproducible training
ml/evaluate.py      metrics, ECE, robustness report
ml/inference.py     local model-only inference
backend/api.py      FastAPI POST /analyze
frontend/           browser-local public demo UI
checkpoints/        validated model artifact location
```

## Install

```powershell
cd C:\Users\q1984\Documents\Playground\proofshield-ai
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The pretrained OpenCLIP weights may download once when training/bootstraping the encoder, then remain cached locally. With cached weights and `checkpoints/model.pt`, inference uses no external API and sends no image to any cloud service.

## Dataset manifest

Use [data/manifest.example.csv](data/manifest.example.csv) as a schema. Labels are `0` for real/camera imagery and `1` for AI-generated imagery. Recommended real sources are COCO, ImageNet, and licensed camera-photo collections. Recommended AI data is GenImage plus balanced Stable Diffusion, Midjourney, and DALL-E samples where licences permit. Split by source/generator/prompt family so near duplicates never cross train, validation, and test.

## Train

```powershell
python -m ml.train --manifest data\manifest.csv --epochs 12 --batch-size 16 --output checkpoints\model.pt
```

Training uses JPEG/social recompression, resize, screenshot borders, crop, blur, and sensor-like noise. The OpenCLIP encoder is frozen; only the fusion MLP, normalizer, and temperature scalar are learned. The saved checkpoint includes configuration, normalizer, calibration, validation metrics, manifest, and seed.

## Evaluate and robustness test

```powershell
python -m ml.evaluate --checkpoint checkpoints\model.pt --manifest data\manifest.csv --output reports\evaluation.json
```

The report includes Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision, Expected Calibration Error, and performance drops for JPEG Q35, 50-percent resize, center crop, Gaussian blur, and screenshot-like recompression. Do not deploy without held-out-source and robustness validation.

## Serve locally

```powershell
cd backend
..\.venv\Scripts\uvicorn.exe api:app --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
cd frontend
npm install
npm run dev
```

The local development API is available at `http://127.0.0.1:8000/analyze` when a validated checkpoint is present. GitHub Pages cannot execute the Python model. Its public edition therefore runs a browser-local demo risk signal, does not upload images, and is explicitly not a trained-model result.

For the research API, use the included [Dockerfile](Dockerfile) and
[deployment guide](deploy/README.md) when you have a validated checkpoint.

## API response

```json
{
  "ai_probability": 62.41,
  "risk_level": "Medium",
  "confidence": 0.2482,
  "signals": {
    "vision_score": 0.59,
    "frequency_score": 0.53,
    "metadata_score": 0.48,
    "learned_contributions": {"vision": 0.11, "frequency": 0.03, "metadata": 0.01}
  },
  "explanation": ["..."],
  "metadata": {"format": "JPEG", "width": 2048, "height": 1365}
}
```

Branch scores are conditional learned branch outputs with all other branches set to their standardized baseline. Contributions are probability changes from branch ablation. They explain the trained network's output; they are not independent rule detectors.

## Operational guardrails

- Treat the output as a probability estimate, never proof.
- Non-commercial research, learning, and demonstration use only; see
  [LICENSE-NONCOMMERCIAL.md](LICENSE-NONCOMMERCIAL.md).
- Never use the output as legal evidence or to decide academic integrity, examination
  or coursework cheating, discipline, hiring, credit, insurance, medical, law-
  enforcement, or other high-impact matters.
- Re-calibrate after material data, generator, or preprocessing shifts.
- Version manifests and retain data-source/licence records for each training run.
- Inspect source-level errors before release; do not rely on random-only split metrics.
- Input images remain in memory only and are never logged or persisted by the API.
