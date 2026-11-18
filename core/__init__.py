# -*- coding: utf-8 -*-
"""核心业务模块包。

模块划分：
- config            配置与应用目录定位
- bbox_utils        边界框几何/文本工具（全项目唯一实现）
- api_client        大模型推理池与解析
- analysis_engine   分析主流程编排
- event_tracker     事件化多帧表决
- yolo_detector     本地YOLO检测先验
- video_processor   视频抽帧
- scene_classifier  场景分段/代表帧
- scene_skills      场景领域知识包
- knowledge_base    法规知识库（RAG）
- annotation_renderer 标注图与报告文本
- evidence_exporter 证据包/整改通知书导出
"""
