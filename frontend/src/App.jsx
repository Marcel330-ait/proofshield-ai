import { AlertTriangle, FileImage, Info, Loader2, ShieldCheck, Upload, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import teichosLogo from "./assets/teichos-ai-safety-logo.png";

const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const API_BASE_URL =
  window.__PROOFSHIELD_API_BASE_URL__ || import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const copy = {
  zh: {
    tagline: "检测图片是否疑似由 AI 生成，并标记潜在欺诈风险。",
    by: "作者",
    imageIntake: "图片检测",
    accepted: "支持 JPG、PNG、WEBP，最大 10MB。",
    clearImage: "清除图片",
    selectImage: "选择或拖入图片",
    localAnalysis: "由本机 ProofShield ML 服务分析",
    analyze: "分析图片",
    privacy: "隐私优先：图片只会发送到你本机的 ProofShield 服务，不会调用第三方推理 API。",
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
    disclaimer: "本工具只提供经过校准的机器学习风险估计，不能证明图片真实或伪造。",
    usageNotice: "仅限非商业的研究、学习与演示用途。不得作为法律、学术诚信、考试或作业作弊、纪律处分、招聘、信贷、保险、医疗、执法或任何其他高风险决策的依据。",
    errors: {
      type: "请上传 JPG、PNG 或 WEBP 图片。",
      size: "文件过大。最大上传大小为 10MB。",
      missing: "请先选择一张图片再运行分析。",
      fallback: "本机 ProofShield 服务无法分析这张图片。",
      serviceUnavailable: "模型服务尚未连接。请稍后重试，或在本机启动 ProofShield API。",
    },
  },
  en: {
    tagline: "Detect whether an image is suspected AI-generated and flag potential fraud risks.",
    by: "by",
    imageIntake: "Image Intake",
    accepted: "JPG, PNG, or WEBP. Maximum 10MB.",
    clearImage: "Clear image",
    selectImage: "Select or drop image",
    localAnalysis: "Analyzed by your local ProofShield ML service",
    analyze: "Analyze Image",
    privacy: "Privacy-first: this UI sends the image only to your local ProofShield service. No third-party inference API is used.",
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
    disclaimer: "This tool provides a calibrated ML risk estimate only. It does not prove whether an image is real or fake.",
    usageNotice: "Non-commercial research, learning, and demonstration use only. Do not use as evidence or a basis for legal, academic integrity, examination or coursework cheating, disciplinary, hiring, credit, insurance, medical, law-enforcement, or other high-impact decisions.",
    errors: {
      type: "Please upload a JPG, PNG, or WEBP image.",
      size: "File is too large. Maximum upload size is 10MB.",
      missing: "Choose an image before running analysis.",
      fallback: "Unable to analyze this image with the local ProofShield service.",
      serviceUnavailable: "The model service is not connected yet. Try again later or start the local ProofShield API.",
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

    const isPublicPageWithoutApi =
      window.location.hostname.endsWith("github.io") && /127\.0\.0\.1|localhost/.test(API_BASE_URL);
    if (isPublicPageWithoutApi) {
      setError(t.errors.serviceUnavailable);
      setLoading(false);
      return;
    }

    try {
      const formData = new FormData();
      formData.append("file", file, file.name);
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Local analysis failed (${response.status}).`);
      }
      setResult(payload);
    } catch (err) {
      setError(err instanceof TypeError ? t.errors.serviceUnavailable : err.message || t.errors.fallback);
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
