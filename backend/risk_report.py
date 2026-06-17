from __future__ import annotations


def get_risk_level(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 35:
        return "Medium"
    return "Low"


def _conclusion(score: int) -> str:
    level = get_risk_level(score)
    if level == "High":
        return "This image is highly suspected to be AI-generated and may carry fraud or misinformation risk."
    if level == "Medium":
        return "This image has some AI-generated risk signals. Further verification is recommended."
    return "This image shows a low AI-generated risk signal, but this does not prove authenticity."


def _conclusion_zh(score: int) -> str:
    level = get_risk_level(score)
    if level == "High":
        return "这张图片高度疑似由 AI 生成，并可能带来欺诈或误导信息风险。"
    if level == "Medium":
        return "这张图片存在一些 AI 生成风险信号，建议进一步核验。"
    return "这张图片显示较低的 AI 生成风险信号，但这并不能证明其真实性。"


def generate_report(score: int, metadata: dict) -> dict:
    signals: list[str] = []
    signals_zh: list[str] = []

    if score >= 70:
        signals.append("The visual detector found stronger synthetic-image risk patterns.")
        signals_zh.append("视觉检测器发现较强的合成图像风险模式。")
    elif score >= 35:
        signals.append("The visual detector found some synthetic-image risk patterns.")
        signals_zh.append("视觉检测器发现一些合成图像风险模式。")
    else:
        signals.append("The visual detector found limited synthetic-image risk patterns.")
        signals_zh.append("视觉检测器发现的合成图像风险模式较少。")

    if metadata["has_exif"]:
        signals.append("The image includes EXIF metadata, which may help provenance review.")
        signals_zh.append("图片包含 EXIF 元数据，可能有助于来源核验。")
    else:
        signals.append("The image has limited or missing metadata.")
        signals_zh.append("图片元数据有限或缺失。")

    if metadata["width"] == metadata["height"]:
        signals.append("The image uses a square format that is common in generated-image workflows.")
        signals_zh.append("图片为方形尺寸，这在生成式图像流程中较常见。")

    signals.append("No trusted provenance information was detected.")
    signals_zh.append("未检测到可信来源信息。")

    recommendations = [
        "Do not rely on this image alone as proof.",
        "Ask for the original file, video evidence, or trusted third-party records.",
        "Be cautious if the image is used for payment, identity, product, accident, contract, or dispute evidence.",
    ]
    recommendations_zh = [
        "不要仅凭这张图片作为证明依据。",
        "请索要原始文件、视频证据或可信第三方记录。",
        "如果图片被用于付款、身份、商品、事故、合同或纠纷证据，请保持谨慎。",
    ]

    return {
        "risk_level": get_risk_level(score),
        "conclusion": _conclusion(score),
        "signals": signals,
        "recommendations": recommendations,
        "localized": {
            "en": {
                "risk_level": get_risk_level(score),
                "conclusion": _conclusion(score),
                "signals": signals,
                "recommendations": recommendations,
            },
            "zh": {
                "risk_level": {
                    "Low": "低",
                    "Medium": "中",
                    "High": "高",
                }[get_risk_level(score)],
                "conclusion": _conclusion_zh(score),
                "signals": signals_zh,
                "recommendations": recommendations_zh,
            },
        },
    }
