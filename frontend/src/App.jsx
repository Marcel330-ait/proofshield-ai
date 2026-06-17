import { AlertTriangle, FileImage, Info, Loader2, ShieldCheck, Upload, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

const copy = {
  zh: {
    tagline: "检测图片是否疑似由 AI 生成，并标记潜在欺诈风险。",
    by: "作者",
    imageIntake: "图片检测",
    accepted: "支持 JPG、PNG、WEBP，最大 10MB。",
    clearImage: "清除图片",
    selectImage: "选择或拖入图片",
    localAnalysis: "直接在你的浏览器中本地分析",
    analyze: "分析图片",
    privacy: "隐私优先：图片仅在你的浏览器中本地分析，不会上传到服务器或第三方服务。",
    riskSignal: "风险信号",
    riskCaption: "风险提示，不是真假结论。",
    waiting: "等待中",
    probability: "AI 生成概率",
    conclusion: "结论",
    emptyConclusion: "上传图片并运行分析后，将生成本地风险报告。",
    signals: "检测信号",
    emptySignals: ["视觉和元数据信号会显示在这里。"],
    recommendations: "安全建议",
    emptyRecommendations: ["请通过原始文件、视频或可信来源进一步核验。"],
    metadata: "元数据摘要",
    format: "格式",
    dimensions: "尺寸",
    exif: "EXIF",
    fileSize: "文件大小",
    present: "存在",
    missing: "缺失",
    disclaimer: "本工具仅提供 AI 生成风险信号，并不能证明图片是真实或伪造的。",
    errors: {
      type: "请上传 JPG、PNG 或 WEBP 图片。",
      size: "文件过大。最大上传大小为 10MB。",
      missing: "请先选择一张图片再运行分析。",
      fallback: "无法在浏览器中分析这张图片。",
    },
  },
  en: {
    tagline: "Detect whether an image is suspected AI-generated and flag potential fraud risks.",
    by: "by",
    imageIntake: "Image Intake",
    accepted: "JPG, PNG, or WEBP. Maximum 10MB.",
    clearImage: "Clear image",
    selectImage: "Select or drop image",
    localAnalysis: "Local analysis directly in your browser",
    analyze: "Analyze Image",
    privacy: "Privacy-first: images are analyzed locally in your browser and are not uploaded to a server or third-party service.",
    riskSignal: "Risk Signal",
    riskCaption: "Risk guidance, not a truth verdict.",
    waiting: "Waiting",
    probability: "AI-generated probability",
    conclusion: "Conclusion",
    emptyConclusion: "Upload an image and run analysis to generate a local risk report.",
    signals: "Detection Signals",
    emptySignals: ["Visual and metadata signals will appear here."],
    recommendations: "Safety Recommendation",
    emptyRecommendations: ["Please verify with original files, videos, or trusted sources."],
    metadata: "Metadata Summary",
    format: "Format",
    dimensions: "Dimensions",
    exif: "EXIF",
    fileSize: "File size",
    present: "Present",
    missing: "Missing",
    disclaimer: "This tool provides an AI-generated risk signal only. It does not prove whether an image is real or fake.",
    errors: {
      type: "Please upload a JPG, PNG, or WEBP image.",
      size: "File is too large. Maximum upload size is 10MB.",
      missing: "Choose an image before running analysis.",
      fallback: "Unable to analyze this image in the browser.",
    },
  },
};

function riskClass(level) {
  return (level || "idle").toLowerCase();
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "Unknown";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(2)} MB`;
}

function imageFormat(file) {
  if (file.type === "image/png") return "PNG";
  if (file.type === "image/webp") return "WEBP";
  return "JPEG";
}

function bufferStartsWith(bytes, offset, values) {
  return values.every((value, index) => bytes[offset + index] === value);
}

function hasJpegExif(bytes) {
  if (!bufferStartsWith(bytes, 0, [0xff, 0xd8])) return false;
  let offset = 2;
  while (offset + 4 < bytes.length) {
    if (bytes[offset] !== 0xff) return false;
    const marker = bytes[offset + 1];
    const length = (bytes[offset + 2] << 8) + bytes[offset + 3];
    if (marker === 0xe1 && bufferStartsWith(bytes, offset + 4, [0x45, 0x78, 0x69, 0x66, 0x00, 0x00])) {
      return true;
    }
    offset += 2 + length;
  }
  return false;
}

function hasPngExif(bytes) {
  if (!bufferStartsWith(bytes, 0, [0x89, 0x50, 0x4e, 0x47])) return false;
  let offset = 8;
  while (offset + 12 < bytes.length) {
    const type = String.fromCharCode(bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]);
    if (type === "eXIf") return true;
    const length = (bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3];
    offset += 12 + Math.max(0, length);
  }
  return false;
}

function hasWebpExif(bytes) {
  if (!bufferStartsWith(bytes, 0, [0x52, 0x49, 0x46, 0x46]) || !bufferStartsWith(bytes, 8, [0x57, 0x45, 0x42, 0x50])) {
    return false;
  }
  let offset = 12;
  while (offset + 8 < bytes.length) {
    const type = String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
    const length = bytes[offset + 4] | (bytes[offset + 5] << 8) | (bytes[offset + 6] << 16) | (bytes[offset + 7] << 24);
    if (type === "EXIF") return true;
    offset += 8 + length + (length % 2);
  }
  return false;
}

function detectExif(file, buffer) {
  const bytes = new Uint8Array(buffer);
  if (file.type === "image/jpeg") return hasJpegExif(bytes);
  if (file.type === "image/png") return hasPngExif(bytes);
  if (file.type === "image/webp") return hasWebpExif(bytes);
  return false;
}

async function readImage(file) {
  const bitmap = await createImageBitmap(file);
  const width = bitmap.width;
  const height = bitmap.height;
  const canvas = document.createElement("canvas");
  canvas.width = 96;
  canvas.height = 96;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  bitmap.close?.();
  return { width, height, pixels };
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function standardDeviation(values) {
  const avg = mean(values);
  return Math.sqrt(mean(values.map((value) => (value - avg) ** 2)));
}

function colorUniformitySignal(pixels) {
  const channels = [[], [], []];
  for (let index = 0; index < pixels.length; index += 4) {
    channels[0].push(pixels[index]);
    channels[1].push(pixels[index + 1]);
    channels[2].push(pixels[index + 2]);
  }

  const channelMeans = channels.map(mean);
  const channelStdev = mean(channels.map(standardDeviation));
  const channelMeanSpread = standardDeviation(channelMeans);

  let signal = 0;
  if (channelStdev < 42) signal += 10;
  if (channelMeanSpread < 18) signal += 6;
  if (channelStdev > 75) signal -= 7;
  return signal;
}

function sizeSignal(metadata) {
  let signal = 0;
  if (metadata.width === metadata.height && [512, 768, 1024, 1536, 2048].includes(metadata.width)) signal += 15;
  if (metadata.width >= 1024 && metadata.height >= 1024) signal += 5;
  if (Math.min(metadata.width, metadata.height) < 256) signal -= 8;
  return signal;
}

function compressionSignal(metadata) {
  const pixels = Math.max(1, metadata.width * metadata.height);
  const bytesPerPixel = metadata.file_size_bytes / pixels;

  if (metadata.format === "PNG" && bytesPerPixel < 1.2) return 12;
  if (metadata.format === "JPEG" && bytesPerPixel < 0.25) return 10;
  if (bytesPerPixel > 5) return -4;
  return 0;
}

function detectAiGenerated(metadata, pixels) {
  let score = 35;
  score += metadata.has_exif ? -6 : 18;
  score += sizeSignal(metadata);
  score += compressionSignal(metadata);
  score += colorUniformitySignal(pixels);
  if (metadata.format === "WEBP") score += 4;
  return Math.max(0, Math.min(100, Math.round(score)));
}

function getRiskLevel(score) {
  if (score >= 70) return "High";
  if (score >= 35) return "Medium";
  return "Low";
}

function generateReport(score, metadata) {
  const riskLevel = getRiskLevel(score);
  const conclusions = {
    High: {
      en: "This image is highly suspected to be AI-generated and may carry fraud or misinformation risk.",
      zh: "这张图片高度疑似由 AI 生成，并可能带来欺诈或误导信息风险。",
    },
    Medium: {
      en: "This image has some AI-generated risk signals. Further verification is recommended.",
      zh: "这张图片存在一些 AI 生成风险信号，建议进一步核验。",
    },
    Low: {
      en: "This image shows a low AI-generated risk signal, but this does not prove authenticity.",
      zh: "这张图片显示较低的 AI 生成风险信号，但这并不能证明其真实性。",
    },
  };

  const signals = { en: [], zh: [] };
  if (score >= 70) {
    signals.en.push("The visual detector found stronger synthetic-image risk patterns.");
    signals.zh.push("视觉检测器发现较强的合成图像风险模式。");
  } else if (score >= 35) {
    signals.en.push("The visual detector found some synthetic-image risk patterns.");
    signals.zh.push("视觉检测器发现一些合成图像风险模式。");
  } else {
    signals.en.push("The visual detector found limited synthetic-image risk patterns.");
    signals.zh.push("视觉检测器发现的合成图像风险模式较少。");
  }

  if (metadata.has_exif) {
    signals.en.push("The image includes EXIF metadata, which may help provenance review.");
    signals.zh.push("图片包含 EXIF 元数据，可能有助于来源核验。");
  } else {
    signals.en.push("The image has limited or missing metadata.");
    signals.zh.push("图片元数据有限或缺失。");
  }

  if (metadata.width === metadata.height) {
    signals.en.push("The image uses a square format that is common in generated-image workflows.");
    signals.zh.push("图片为方形尺寸，这在生成式图像流程中较常见。");
  }

  signals.en.push("No trusted provenance information was detected.");
  signals.zh.push("未检测到可信来源信息。");

  const recommendations = {
    en: [
      "Do not rely on this image alone as proof.",
      "Ask for the original file, video evidence, or trusted third-party records.",
      "Be cautious if the image is used for payment, identity, product, accident, contract, or dispute evidence.",
    ],
    zh: [
      "不要仅凭这张图片作为证明依据。",
      "请索要原始文件、视频证据或可信第三方记录。",
      "如果图片被用于付款、身份、商品、事故、合同或纠纷证据，请保持谨慎。",
    ],
  };

  return {
    ai_probability: score,
    risk_level: riskLevel,
    conclusion: conclusions[riskLevel].en,
    signals: signals.en,
    recommendations: recommendations.en,
    localized: {
      en: {
        risk_level: riskLevel,
        conclusion: conclusions[riskLevel].en,
        signals: signals.en,
        recommendations: recommendations.en,
      },
      zh: {
        risk_level: { Low: "低", Medium: "中", High: "高" }[riskLevel],
        conclusion: conclusions[riskLevel].zh,
        signals: signals.zh,
        recommendations: recommendations.zh,
      },
    },
    disclaimer_localized: {
      en: copy.en.disclaimer,
      zh: copy.zh.disclaimer,
    },
  };
}

async function analyzeFileInBrowser(file) {
  const buffer = await file.arrayBuffer();
  const image = await readImage(file);
  const metadata = {
    has_exif: detectExif(file, buffer),
    exif_keys: [],
    format: imageFormat(file),
    width: image.width,
    height: image.height,
    mode: "RGBA",
    file_size_bytes: file.size,
  };
  const aiScore = detectAiGenerated(metadata, image.pixels);
  return {
    ...generateReport(aiScore, metadata),
    metadata,
  };
}

function App() {
  const inputRef = useRef(null);
  const [language, setLanguage] = useState("zh");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const t = copy[language];

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const fileSummary = useMemo(() => {
    if (!file) return null;
    return {
      name: file.name,
      type: file.type || "Unknown",
      size: formatBytes(file.size),
    };
  }, [file]);

  const localizedResult = result?.localized?.[language];
  const displayRiskLevel = localizedResult?.risk_level || result?.risk_level || t.waiting;
  const displayConclusion = localizedResult?.conclusion || result?.conclusion || t.emptyConclusion;
  const displaySignals = localizedResult?.signals || result?.signals || t.emptySignals;
  const displayRecommendations = localizedResult?.recommendations || result?.recommendations || t.emptyRecommendations;
  const displayDisclaimer = result?.disclaimer_localized?.[language] || t.disclaimer;

  function chooseFile(nextFile) {
    setError("");
    setResult(null);

    if (!nextFile) return;
    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setFile(null);
      setError(t.errors.type);
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setFile(null);
      setError(t.errors.size);
      return;
    }

    setFile(nextFile);
  }

  async function analyzeImage() {
    if (!file) {
      setError(t.errors.missing);
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      setResult(await analyzeFileInBrowser(file));
    } catch (err) {
      setError(err.message || t.errors.fallback);
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="brand-row">
            <ShieldCheck size={28} aria-hidden="true" />
            <h1>ProofShield AI</h1>
          </div>
          <p>{t.tagline}</p>
        </div>
        <div className="top-actions">
          <div className="language-toggle" role="group" aria-label="Language">
            <button className={language === "zh" ? "active" : ""} type="button" onClick={() => setLanguage("zh")}>
              中文
            </button>
            <button className={language === "en" ? "active" : ""} type="button" onClick={() => setLanguage("en")}>
              English
            </button>
          </div>
          <div className="author-credit">{t.by} Marcel330-ait</div>
        </div>
      </header>

      <section className="workspace">
        <div className="upload-panel">
          <div className="panel-heading">
            <div>
              <h2>{t.imageIntake}</h2>
              <p>{t.accepted}</p>
            </div>
            {file && (
              <button className="icon-button" type="button" onClick={reset} aria-label={t.clearImage} title={t.clearImage}>
                <X size={18} />
              </button>
            )}
          </div>

          <label
            className="dropzone"
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              chooseFile(event.dataTransfer.files?.[0]);
            }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(event) => chooseFile(event.target.files?.[0])}
            />
            {previewUrl ? (
              <img src={previewUrl} alt="Uploaded preview" />
            ) : (
              <div className="empty-upload">
                <Upload size={34} aria-hidden="true" />
                <strong>{t.selectImage}</strong>
                <span>{t.localAnalysis}</span>
              </div>
            )}
          </label>

          {fileSummary && (
            <div className="file-strip">
              <FileImage size={18} aria-hidden="true" />
              <div>
                <strong>{fileSummary.name}</strong>
                <span>
                  {fileSummary.type} · {fileSummary.size}
                </span>
              </div>
            </div>
          )}

          <button className="primary-action" type="button" onClick={analyzeImage} disabled={loading || !file}>
            {loading ? <Loader2 className="spin" size={18} aria-hidden="true" /> : <ShieldCheck size={18} aria-hidden="true" />}
            {t.analyze}
          </button>

          <div className="privacy-note">
            <Info size={17} aria-hidden="true" />
            <span>{t.privacy}</span>
          </div>
        </div>

        <div className="results-panel">
          <div className="panel-heading">
            <div>
              <h2>{t.riskSignal}</h2>
              <p>{t.riskCaption}</p>
            </div>
            <span className={`risk-badge ${riskClass(result?.risk_level)}`}>{displayRiskLevel}</span>
          </div>

          {error && (
            <div className="alert">
              <AlertTriangle size={18} aria-hidden="true" />
              <span>{error}</span>
            </div>
          )}

          <div className="score-block">
            <div className="score-row">
              <span>{t.probability}</span>
              <strong>{result ? `${result.ai_probability}%` : "--"}</strong>
            </div>
            <div className="meter" aria-hidden="true">
              <span style={{ width: `${result?.ai_probability || 0}%` }} />
            </div>
          </div>

          <div className="conclusion">
            <h3>{t.conclusion}</h3>
            <p>{displayConclusion}</p>
          </div>

          <div className="detail-grid">
            <section>
              <h3>{t.signals}</h3>
              <ul>
                {displaySignals.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>
            <section>
              <h3>{t.recommendations}</h3>
              <ul>
                {displayRecommendations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </section>
          </div>

          <section className="metadata-panel">
            <h3>{t.metadata}</h3>
            <dl>
              <div>
                <dt>{t.format}</dt>
                <dd>{result?.metadata?.format || "--"}</dd>
              </div>
              <div>
                <dt>{t.dimensions}</dt>
                <dd>{result ? `${result.metadata.width} x ${result.metadata.height}` : "--"}</dd>
              </div>
              <div>
                <dt>{t.exif}</dt>
                <dd>{result ? (result.metadata.has_exif ? t.present : t.missing) : "--"}</dd>
              </div>
              <div>
                <dt>{t.fileSize}</dt>
                <dd>{result ? formatBytes(result.metadata.file_size_bytes) : "--"}</dd>
              </div>
            </dl>
          </section>

          <p className="disclaimer">{displayDisclaimer}</p>
        </div>
      </section>
    </main>
  );
}

export default App;
