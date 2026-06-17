# ProofShield AI

本地优先的 AI 生成图片风险信号工具。

Local-first AI-generated image risk signal app.

作者 / Author: [Marcel330-ait](https://github.com/Marcel330-ait)

## 在线使用 / Use Online

打开 GitHub Pages 链接即可使用，不需要安装后端：

Open the GitHub Pages link to use the app directly. No backend setup is required:

```text
https://marcel330-ait.github.io/proofshield-ai/
```

公开网页版本会直接在浏览器中分析图片，不会上传图片到服务器。

The public web version analyzes images directly in the browser and does not upload images to a server.

## 项目简介 / Overview

ProofShield AI 是一个 MVP Web 应用，用于分析上传图片是否疑似由 AI 生成，并提示潜在欺诈或误导信息风险。

ProofShield AI is an MVP web app for analyzing whether an uploaded image is suspected to be AI-generated and whether it may carry fraud or misinformation risk.

返回内容 / It returns:

- AI 生成概率 / AI-generated probability from 0 to 100
- 风险等级：低、中、高 / Risk level: Low, Medium, or High
- 谨慎结论 / Cautious conclusion
- 检测信号 / Detection signals
- 安全建议 / Safety recommendations
- 隐私友好的元数据摘要 / Privacy-preserving metadata summary

## 产品边界 / Product Boundary

ProofShield AI 不证明图片是真实或伪造的，不判断某个人是否诈骗，也不作法律结论。本工具只提供风险信号，并建议用户通过原始文件、视频证据或可信来源进一步核验。

ProofShield AI does not prove whether an image is real or fake. It does not say a person is scamming and does not make legal conclusions. The app only provides a risk signal and recommends independent verification.

## 隐私模型 / Privacy Model

MVP 采用本地优先、隐私优先的设计：

The MVP is local-first and privacy-preserving:

- 上传图片在开发阶段保留在用户本机 / Uploaded images stay on the user's local machine during development.
- 在线版本直接在浏览器中处理图片 / The online version processes images directly in the browser.
- 本地开发版本也可以调用本地 FastAPI 后端 / The local development version can also call the local FastAPI backend.
- 不调用第三方 API / No third-party APIs are called.
- 不使用云存储 / No cloud storage is used.
- 不永久保存上传图片 / Uploaded images are not saved permanently.
- 后端从内存中处理图片 / The backend processes images from memory.
- 不记录原始图片文件或图片字节 / Raw image files and bytes are not logged.

## 项目结构 / Project Structure

```text
proofshield-ai/
  backend/
    main.py
    detector.py
    metadata_checker.py
    risk_report.py
    requirements.txt
  frontend/
    src/
      App.jsx
      main.jsx
      styles.css
    package.json
    vite.config.js
  README.md
```

## 运行后端 / Run Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

健康检查 / Health check:

```bash
curl http://localhost:8000/health
```

## 运行前端 / Run Frontend

```bash
cd frontend
npm install
npm run dev
```

打开 / Open:

```text
http://localhost:5173
```

## 部署到 GitHub Pages / Deploy to GitHub Pages

本仓库包含 GitHub Actions workflow：`.github/workflows/deploy.yml`。推送到 `main` 后会自动构建 `frontend` 并发布到 GitHub Pages。

This repository includes a GitHub Actions workflow at `.github/workflows/deploy.yml`. After pushing to `main`, it builds `frontend` and deploys it to GitHub Pages automatically.

如果第一次部署后页面没有出现，请在 GitHub 仓库中打开：

If the page does not appear after the first deployment, open the GitHub repository settings:

```text
Settings -> Pages -> Build and deployment -> Source -> GitHub Actions
```

## API 示例 / Example API Response

```json
{
  "ai_probability": 87,
  "risk_level": "High",
  "conclusion": "This image is highly suspected to be AI-generated and may carry fraud or misinformation risk.",
  "signals": [
    "The visual detector found stronger synthetic-image risk patterns.",
    "The image has limited or missing metadata.",
    "No trusted provenance information was detected."
  ],
  "recommendations": [
    "Do not rely on this image alone as proof.",
    "Ask for the original file, video evidence, or trusted third-party records.",
    "Be cautious if the image is used for payment, identity, product, accident, contract, or dispute evidence."
  ],
  "localized": {
    "zh": {
      "risk_level": "高",
      "conclusion": "这张图片高度疑似由 AI 生成，并可能带来欺诈或误导信息风险。"
    },
    "en": {
      "risk_level": "High",
      "conclusion": "This image is highly suspected to be AI-generated and may carry fraud or misinformation risk."
    }
  },
  "metadata": {
    "has_exif": false,
    "exif_keys": [],
    "format": "PNG",
    "width": 1024,
    "height": 1024,
    "mode": "RGB",
    "file_size_bytes": 1200000
  },
  "disclaimer": "This tool provides an AI-generated risk signal only. It does not prove whether an image is real or fake.",
  "disclaimer_localized": {
    "zh": "本工具仅提供 AI 生成风险信号，并不能证明图片是真实或伪造的。",
    "en": "This tool provides an AI-generated risk signal only. It does not prove whether an image is real or fake."
  }
}
```

## 替换占位检测器 / Replacing the Placeholder Detector

当前 `backend/detector.py` 使用确定性的占位逻辑。后续可以把 `detect_ai_generated(image, metadata)` 替换为本地模型，例如：

The current detector in `backend/detector.py` is deterministic placeholder logic. Replace `detect_ai_generated(image, metadata)` with a real local model later, such as:

- CLIP/SigLIP embedding + MLP classifier
- EfficientNet/ConvNeXt binary classifier
- Any local synthetic-image detector

默认保持本地推理。只有在用户明确启用云模式时，才应调用外部推理 API。

Keep inference local by default. Cloud inference should stay disabled unless the user explicitly enables a cloud mode.
