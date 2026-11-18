# 视频违章智能分析系统

面向施工现场的无人机航拍视频分析桌面应用：导入视频 → 生成待复核候选 →
输出标注图、报告与证据包。分析可使用已配置的视觉服务，产物本地归档；候选或未检出均不等于现场结论。

## 功能一览

- **视频分析工作台**：导入单视频，可暂停 / 继续 / 停止；批处理走有明确副作用标记的分类工具
- **九大类违章识别**：安全防护、高处作业、临边防护、孔洞防护、临时用电、起重吊装、消防动火、材料堆放、有限空间
- **成果自动产出**：逐帧标注图（含违章标签）、违章清单、证据包（原始画面 + 标注画面 + 说明）
- **场景代表帧**：按场景与配置选择样本，高空远景也可能保留代表帧；抽样不等于完整覆盖

## Publication scope
This repository contains application source plus a small, fully synthetic sample under `examples/synthetic_test_assets/`. It is hand-authored vector art and metadata for format/loader smoke checks only, not real site evidence or a substitute for the private test suite.
Original fixtures, photographs, videos, datasets, model weights, local configuration, generated output, knowledge bases, and deployment artifacts are intentionally excluded.

## Configuration
运行期配置在 `config.json`（UTF-8 无 BOM），完整默认值见 `core/config.py` 的 `DEFAULT_CONFIG`。
常用项：`parallel_workers`（总并发）、`scene_concurrency`（场景分析并发）、
`event_report_mode`（事件上报口径：`all` 单帧即报 / `confirmed_only` 多帧确认）。

## LLM request constraints
- LLM 推理请求：**不得带 seed 参数**；completion 额度须覆盖"思考 + 正文"
  （下限见 `core/api_client.py` 的 `MIN_COMPLETION_TOKENS`）

## Runtime requirements
- Python 3.13
- Install the dependencies listed in `requirements.txt`.
- Supply model assets and `config.json` locally; do not commit secrets or runtime data.

## Source layout
- `application/`: application orchestration
- `core/`: analysis and business logic
- `ui/`: desktop and web UI
- `main.py`: desktop application entry point
- `web_preview_entry.py`: web preview entry point
