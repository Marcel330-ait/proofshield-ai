import { AlertTriangle, FileImage, Info, Loader2, ShieldCheck, Upload, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

const API_URL = "http://localhost:8000/analyze";
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
    localAnalysis: "通过你自己的本地后端进行分析",
    analyze: "分析图片",
    privacy: "隐私优先：图片会通过你自己的本地后端分析，不会上传到第三方服务器。",
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
      fallback: "无法连接本地后端。",
    },
  },
  en: {
    tagline: "Detect whether an image is suspected AI-generated and flag potential fraud risks.",
    by: "by",
    imageIntake: "Image Intake",
    accepted: "JPG, PNG, or WEBP. Maximum 10MB.",
    clearImage: "Clear image",
    selectImage: "Select or drop image",
    localAnalysis: "Local analysis through your own backend",
    analyze: "Analyze Image",
    privacy: "Privacy-first: images are analyzed locally through your own backend and are not uploaded to third-party servers.",
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
      fallback: "Unable to reach the local backend.",
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

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || t.errors.fallback);
      }
      setResult(payload);
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
