import { AlertTriangle, FileImage, Info, Loader2, ShieldCheck, Upload, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import teichosLogo from "./assets/teichos-ai-safety-logo.png";

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
    localAnalysis: "浏览器本地演示分析，不上传图片",
    analyze: "分析图片",
    privacy: "隐私优先：图片只在当前浏览器分析，不会上传到服务器或第三方服务。",
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
    disclaimer: "当前公开版使用浏览器端启发式演示分析，只提供风险提示，不能证明图片真实或伪造。",
    usageNotice: "仅限非商业的研究、学习与演示用途。不得作为法律、学术诚信、考试或作业作弊、纪律处分、招聘、信贷、保险、医疗、执法或任何其他高风险决策的依据。",
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
    localAnalysis: "Browser-local demo analysis. No upload.",
    analyze: "Analyze Image",
    privacy: "Privacy-first: images are analyzed only in this browser and are never uploaded to a server or third party.",
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
    disclaimer: "The public edition uses a browser-side heuristic demo. It provides a risk signal only and does not prove whether an image is real or fake.",
    usageNotice: "Non-commercial research, learning, and demonstration use only. Do not use as evidence or a basis for legal, academic integrity, examination or coursework cheating, disciplinary, hiring, credit, insurance, medical, law-enforcement, or other high-impact decisions.",
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

function hasJpegExif(bytes) {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return false;
  for (let offset = 2; offset + 9 < bytes.length; ) {
    if (bytes[offset] !== 0xff) return false;
    const marker = bytes[offset + 1];
    const length = (bytes[offset + 2] << 8) + bytes[offset + 3];
    if (marker === 0xe1 && bytes[offset + 4] === 0x45 && bytes[offset + 5] === 0x78) return true;
    if (length < 2) return false;
    offset += 2 + length;
  }
  return false;
}

async function readImageMetrics(file) {
  const bitmap = await createImageBitmap(file);
  const width = bitmap.width;
  const height = bitmap.height;
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const histogram = new Array(16).fill(0);
  let edgeCount = 0;
  let comparisons = 0;
  const gray = new Float32Array(128 * 128);
  for (let index = 0, pixel = 0; index < pixels.length; index += 4, pixel += 1) {
    const value = 0.299 * pixels[index] + 0.587 * pixels[index + 1] + 0.114 * pixels[index + 2];
    gray[pixel] = value;
    histogram[Math.min(15, Math.floor(value / 16))] += 1;
  }
  for (let y = 0; y < 128; y += 1) {
    for (let x = 0; x < 128; x += 1) {
      const current = gray[y * 128 + x];
      if (x < 127) {
        comparisons += 1;
        if (Math.abs(current - gray[y * 128 + x + 1]) > 25) edgeCount += 1;
      }
      if (y < 127) {
        comparisons += 1;
        if (Math.abs(current - gray[(y + 1) * 128 + x]) > 25) edgeCount += 1;
      }
    }
  }
  const entropy = histogram.reduce((sum, count) => {
    if (!count) return sum;
    const probability = count / gray.length;
    return sum - probability * Math.log2(probability);
  }, 0);
  return { width, height, entropy, edgeDensity: edgeCount / Math.max(1, comparisons) };
}

function browserRiskLevel(score) {
  if (score < 40) return "Low";
  if (score < 70) return "Medium";
  return "High";
}

async function analyzeFileInBrowser(file) {
  const [buffer, image] = await Promise.all([file.arrayBuffer(), readImageMetrics(file)]);
  const bytes = new Uint8Array(buffer);
  const metadata = {
    has_exif: file.type === "image/jpeg" && hasJpegExif(bytes),
    format: imageFormat(file),
    width: image.width,
    height: image.height,
    file_size_bytes: file.size,
  };
  const bytesPerPixel = file.size / Math.max(1, image.width * image.height);
  let score = 30;
  if (!metadata.has_exif) score += 7;
  if (image.edgeDensity < 0.04) score += 12;
  if (image.entropy < 3.15) score += 10;
  if (metadata.format === "PNG" && bytesPerPixel < 0.9) score += 5;
  if (image.edgeDensity > 0.12) score -= 6;
  score = Math.max(0, Math.min(100, Math.round(score)));
  const level = browserRiskLevel(score);
  const explanation = [
    "Browser demo: this estimate uses local image texture, compression, and metadata signals.",
    "No image was uploaded, and this result is not a trained-model verdict.",
  ];
  const explanationZh = [
    "浏览器演示版：此估计使用本地图片纹理、压缩与元数据风险信号。",
    "图片未被上传；结果不是经过训练模型的判定。",
  ];
  return {
    ai_probability: score,
    risk_level: level,
    confidence: 0.2,
    explanation,
    recommendations: ["Use original files and trusted provenance records for important decisions."],
    metadata,
    localized: {
      en: { risk_level: level, conclusion: "Browser-local demo risk signal. Verify important claims independently.", signals: explanation, recommendations: ["Use original files and trusted provenance records for important decisions."] },
      zh: { risk_level: { Low: "低", Medium: "中", High: "高" }[level], conclusion: "浏览器本地演示风险提示；重要事项请独立核验。", signals: explanationZh, recommendations: ["请结合原始文件与可信来源独立核验。"] },
    },
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
  const displaySignals = localizedResult?.signals || result?.explanation || t.emptySignals;
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
        <div className="brand-block">
          <div className="brand-row">
            <div className="org-logo" aria-label="Teichos AI Safety">
              <img src={teichosLogo} alt="Teichos AI Safety logo" />
            </div>
            <div>
              <h1>ProofShield AI</h1>
              <span className="org-label">A Teichos AI Safety tool</span>
            </div>
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
          <p className="use-notice">{t.usageNotice}</p>
        </div>
      </section>
    </main>
  );
}

export default App;
