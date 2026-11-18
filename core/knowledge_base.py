# -*- coding: utf-8 -*-
"""
知识库管理模块 - PDF解析与检索
支持向量检索 + 关键词检索降级方案
"""
import os
import json
import re
import threading
import numpy as np
from core.config import get_config, get_app_dir
from core.atomic_io import atomic_write_json

# pymupdf（PyMuPDF）惰性导入：其 import 会加载 mupdfcpp64.dll（约 25MB）与
# _mupdf.pyd（约 12.5MB），实测冷启动约 105ms。仅"添加PDF/解析PDF"时才需要，
# 放到 _extract_pdf_chunks 内部导入，启动路径不再付出该开销。

# 索引写盘互斥：导入/删除并发时不互相覆盖条目，也不会写坏 2.6MB 索引文件
_kb_lock = threading.RLock()

# 预设违章场景（粗筛与知识库分类共用）
SCENES = [
    "高处作业与安全防护",
    "孔洞与临边防护",
    "临时用电",
    "起重吊装",
    "有限空间作业",
    "施工机具与设备",
    "消防与动火作业",
    "材料堆放与文明施工",
    "人员资质与作业管理",
]

# 各场景本地关键词（用于添加PDF时自动打标签，免API）
SCENE_KEYWORDS = {
    "高处作业与安全防护": ["安全帽", "安全带", "高处作业", "登高", "脚手架", "爬梯", "坠落", "安全绳", "防坠", "悬空", "吊篮", "作业平台"],
    "孔洞与临边防护": ["孔洞", "盖板", "临边", "防护栏杆", "栏杆", "基坑", "围栏", "围挡", "井口", "沟槽", "安全网", "洞口"],
    "临时用电": ["配电箱", "临时用电", "电缆", "漏电", "接地", "电线", "电源", "开关箱", "用电", "电气"],
    "起重吊装": ["起重", "吊装", "吊车", "吊臂", "起重机", "吊索", "吊具", "警戒区", "吊物", "塔吊"],
    "有限空间作业": ["有限空间", "井下", "隧道", "通风", "气体检测", "缺氧", "硫化氢", "有毒气体", "密闭"],
    "施工机具与设备": ["机具", "机械设备", "电动工具", "切割机", "电焊机", "搅拌机", "工器具"],
    "消防与动火作业": ["动火", "灭火器", "消防", "焊接", "明火", "防火", "易燃"],
    "材料堆放与文明施工": ["材料堆放", "堆放", "文明施工", "场地", "道路", "标识", "警示", "垃圾", "整洁"],
    "人员资质与作业管理": ["资质", "培训", "交底", "特种作业", "证件", "作业票", "审批", "监护", "操作证", "持证"],
}


def official_name(title):
    """文档标题 -> 官方文件名（去扩展名）"""
    if not title:
        return title
    return title[:-4] if title.lower().endswith(".pdf") else title


class KnowledgeBase:
    def __init__(self):
        self.config = get_config()
        self.knowledge_dir = os.path.join(get_app_dir(), self.config.get("knowledge_dir", "knowledge"))
        self.index_file = os.path.join(self.knowledge_dir, "knowledge_index.json")
        self.documents = []  # [{id, title, content, page, embedding, has_embedding}]
        self._ensure_dir()
        self._load_index()

    def _ensure_dir(self):
        if not os.path.exists(self.knowledge_dir):
            os.makedirs(self.knowledge_dir)

    def _load_index(self):
        """加载索引"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.documents = data.get("documents", [])
                    # 兼容旧格式
                    for doc in self.documents:
                        if "has_embedding" not in doc:
                            doc["has_embedding"] = doc.get("embedding") is not None
            except Exception as e:
                print(f"加载知识库索引失败: {e}")

    def _save_index(self):
        """保存索引（原子写 + 加锁）。

        索引重建代价高（重新解析 PDF + 扫描件 OCR 消耗 API 额度），
        因此绝不允许"写一半"；加锁同时防止导入/删除并发时条目互相覆盖。
        """
        try:
            with _kb_lock:
                atomic_write_json(self.index_file, {"documents": self.documents})
        except Exception as e:
            print(f"保存知识库索引失败: {e}")

    def add_pdf(self, pdf_path, title=None, progress_callback=None):
        """
        添加PDF文件到知识库
        自动检测扫描件并使用视觉模型OCR识别
        
        Returns:
            dict: {success, count, total_chunks, error, mode, ocr_pages}
        """
        if not os.path.exists(pdf_path):
            return {"success": False, "count": 0, "total_chunks": 0,
                    "error": "文件不存在", "mode": "", "ocr_pages": 0}

        if title is None:
            title = os.path.basename(pdf_path)

        # 复制一份到「知识库」文件夹，便于集中管理和下次启动自动加载
        try:
            kb_folder = os.path.join(get_app_dir(), "知识库")
            os.makedirs(kb_folder, exist_ok=True)
            src_dir = os.path.dirname(os.path.abspath(pdf_path))
            if src_dir.lower() != kb_folder.lower():
                dst = os.path.join(kb_folder, os.path.basename(pdf_path))
                if not os.path.exists(dst):
                    import shutil
                    shutil.copy2(pdf_path, dst)
        except Exception:
            pass

        # 注意：同名文档的删除必须等到"新内容确认可入库"之后再做（见下方
        # chunks 非空处）——此前在这里先删，一旦 PDF 解析/OCR 失败就直接 return，
        # 旧同名文档的索引块已被永久删除，RAG 会静默丢掉整部法规。

        # 提取文本（含扫描件检测与OCR）
        if progress_callback:
            progress_callback(5, "正在解析PDF...")

        try:
            chunks, ocr_pages, pdf_info = self._extract_pdf_chunks(
                pdf_path, title, progress_callback=progress_callback
            )
        except Exception as e:
            return {"success": False, "count": 0, "total_chunks": 0,
                    "error": f"PDF解析出错：{e}", "mode": "", "ocr_pages": 0}

        if not chunks:
            # 给出针对性的错误提示
            if pdf_info["is_scanned"] and not pdf_info["api_ok"]:
                return {"success": False, "count": 0, "total_chunks": 0,
                        "error": (f"该PDF是扫描件（{pdf_info['total_pages']}页，无可提取文字），"
                                  f"需要通过视觉模型OCR识别，但当前API不可用。\n\n"
                                  f"请先在「设置」中配置正确的API地址和密钥（点测试连接确认），"
                                  f"然后重新添加此PDF。"),
                        "mode": "", "ocr_pages": 0}
            elif pdf_info["is_scanned"]:
                return {"success": False, "count": 0, "total_chunks": 0,
                        "error": (f"该PDF是扫描件，OCR识别未返回有效文字"
                                  f"（共{pdf_info['total_pages']}页，OCR了{ocr_pages}页）。\n"
                                  f"请检查API的视觉模型是否支持图片识别。"),
                        "mode": "", "ocr_pages": ocr_pages}
            else:
                return {"success": False, "count": 0, "total_chunks": 0,
                        "error": (f"未能从PDF中提取到文字内容"
                                  f"（共{pdf_info['total_pages']}页）。"),
                        "mode": "", "ocr_pages": 0}

        total = len(chunks)
        doc_official = (pdf_info.get("doc_title") or "").strip() or official_name(title)

        # 解析已成功、chunks 已到手：此时才移除同名旧条目（重新导入场景，
        # 不删知识库文件夹的源文件）。放在这里可保证解析/OCR 失败时旧数据不丢。
        self.remove_document(title, delete_file=False)

        # 尝试生成embedding（本地 bge 检索模型，批量编码，零 API 开销）
        from core.embedder import embed_many
        vecs = embed_many([c["content"] for c in chunks], is_query=False)
        if not vecs:
            vecs = [None] * len(chunks)

        new_docs = []
        embedding_success = 0
        embedding_failed = 0

        for i, chunk in enumerate(chunks):
            if progress_callback:
                pct = 60 + int((i / max(total, 1)) * 35)
                status = f"生成向量中 ({i+1}/{total})..."
                if embedding_failed > 0:
                    status += f"（{embedding_failed}个失败，将用关键词检索）"
                progress_callback(pct, status)

            doc = {
                "id": f"{title}_{i}",
                "title": title,
                "official_title": doc_official,
                "page": chunk["page"],
                "content": chunk["content"],
                "article": chunk.get("article", ""),
                "scenes": chunk.get("scenes", []),
                "embedding": None,
                "has_embedding": False
            }

            embedding = vecs[i]
            if embedding:
                doc["embedding"] = embedding
                doc["has_embedding"] = True
                embedding_success += 1
            else:
                embedding_failed += 1

            new_docs.append(doc)

        if embedding_success > total * 0.5:
            mode = "embedding"
        else:
            mode = "keyword"

        self.documents.extend(new_docs)
        self._save_index()

        if progress_callback:
            progress_callback(100, f"添加完成（共{total}块）")

        return {
            "success": True,
            "count": len(new_docs),
            "total_chunks": total,
            "embedding_success": embedding_success,
            "embedding_failed": embedding_failed,
            "mode": mode,
            "ocr_pages": ocr_pages,
            "error": None
        }

    def _extract_pdf_chunks(self, pdf_path, title, max_chunk_size=600, overlap=80, progress_callback=None):
        """
        从PDF中提取文本并分块
        自动检测扫描件：文字提取不到时用视觉模型OCR
        
        Returns:
            (chunks, ocr_pages, pdf_info)
        """
        chunks = []
        ocr_pages = 0
        import pymupdf  # 惰性导入：见文件头说明
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        api_ok = True

        # 第一遍：常规文本提取，统计每页情况
        page_texts = {}
        first_page_raw = ""
        total_text_len = 0
        for page_num in range(total_pages):
            page = doc[page_num]
            raw_text = page.get_text()
            if page_num == 0:
                first_page_raw = raw_text
            text = self._clean_text(raw_text)
            page_texts[page_num] = text
            total_text_len += len(text.strip())

        # 判断是否扫描件：平均每页文字 < 30字 视为扫描件/图片型PDF
        avg_text = total_text_len / max(total_pages, 1)
        is_scanned = avg_text < 30

        if is_scanned:
            # 检查API可用性
            from core.api_client import get_ai_client
            ai_client = get_ai_client()
            ok, msg = ai_client.test_connection()
            if not ok:
                api_ok = False
                doc.close()
                return [], 0, {
                    "total_pages": total_pages,
                    "is_scanned": True,
                    "api_ok": False,
                    "api_error": msg
                }

            # OCR模式：逐页渲染成图片让视觉模型识别
            if progress_callback:
                progress_callback(10, f"检测到扫描件，正在OCR识别 ({total_pages}页)...")

            import base64
            for page_num in range(total_pages):
                if progress_callback:
                    pct = 10 + int((page_num / max(total_pages, 1)) * 45)
                    progress_callback(pct, f"OCR识别中 第{page_num+1}/{total_pages}页...")

                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

                ocr_text = ai_client.ocr_image(img_b64)
                if ocr_text and len(ocr_text.strip()) > 5:
                    page_texts[page_num] = ocr_text
                    ocr_pages += 1

        doc.close()

        # 从首页提取正式文档标题（引用时显示《官方文件名》而非PDF文件名）
        # 用未合并行的原始文本；扫描件用OCR结果
        doc_title = self._extract_doc_title(page_texts.get(0, "") if is_scanned else first_page_raw)

        # 第二遍：拼接全文（带页码映射），优先按"第X条"条款切分
        page_spans = []  # [(起始offset, 结束offset, 页码)]
        full_text = ""
        for page_num in range(total_pages):
            text = page_texts.get(page_num, "")
            if not text.strip():
                continue
            start = len(full_text)
            full_text += text + "\n"
            page_spans.append((start, len(full_text), page_num + 1))

        def _page_of(offset):
            for s, e, p in page_spans:
                if s <= offset < e:
                    return p
            return page_spans[-1][2] if page_spans else 1

        chunks = []
        article_matches = list(re.finditer(r'第[一二三四五六七八九十百千零\d]{1,6}条', full_text))

        if len(article_matches) >= 3:
            # 法规类文档：按条款切分，短条合并、长条细分
            segs = []  # [(article, content, offset)]
            for i, m in enumerate(article_matches):
                start = m.start()
                end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(full_text)
                content = full_text[start:end].strip()
                if not content:
                    continue
                if len(content) > max_chunk_size:
                    # 长条款切成多段（条款号相同）
                    # 步长必须防零：max_chunk_size <= 50 时 range(0, N, 0) 会抛 ValueError
                    _piece_len = max(1, max_chunk_size - 50)
                    for j in range(0, len(content), _piece_len):
                        piece = content[j:j + _piece_len].strip()
                        if piece:
                            segs.append((m.group(), piece, start))
                else:
                    segs.append((m.group(), content, start))

            buf_articles, buf_content, buf_offset = [], "", 0

            def _flush_buf():
                """把当前缓冲的条款块写入 chunks（空缓冲则跳过）。"""
                if buf_content:
                    chunks.append({
                        "page": _page_of(buf_offset),
                        "content": buf_content,
                        "article": self._join_articles(buf_articles),
                        "scenes": self._local_scenes(buf_content),
                    })

            for article, content, offset in segs:
                if buf_content and len(buf_content) + len(content) + 1 <= max_chunk_size:
                    buf_content += "\n" + content
                    buf_articles.append(article)
                else:
                    _flush_buf()
                    buf_articles, buf_content, buf_offset = [article], content, offset
            _flush_buf()
        else:
            # 非条款结构文档：按大小滑窗切分
            step = max(max_chunk_size - overlap, 100)
            pos = 0
            while pos < len(full_text):
                piece = full_text[pos:pos + max_chunk_size].strip()
                if piece:
                    chunks.append({
                        "page": _page_of(pos),
                        "content": piece,
                        "article": "",
                        "scenes": self._local_scenes(piece),
                    })
                pos += step

        return chunks, ocr_pages, {
            "total_pages": total_pages,
            "is_scanned": is_scanned,
            "api_ok": api_ok,
            "doc_title": doc_title
        }

    @staticmethod
    def _extract_doc_title(first_page_text):
        """从PDF首页文本提取正式文档标题（如《电力重大事故隐患判定标准及治理监督管理规定》）"""
        head = (first_page_text or "")[:600]
        if not head:
            return None
        kw = ("规定", "办法", "标准", "条例", "规程", "大纲", "规范", "导则", "细则", "制度")
        exclude_words = ("企业标准", "国家标准", "行业标准", "公司标准", "技术标准", "安全法")

        # 1) 书名号标题（长度足够且含法规词，排除正文里引用的其他法规短名）
        for c in re.findall(r'《([^》]{6,50})》', head):
            if any(k in c for k in kw) and len(c) >= 10 and not any(x in c for x in exclude_words):
                m = re.search(r'[（(][^（）()]*版[^（）()]*对比[^（）()]*[）)]', head)
                if m:
                    c += m.group()
                return c

        # 2) 首页前10行中找标题行（排除英文翻译行、标准号行、正文章节行）
        lines = [l.strip() for l in head.split('\n') if l.strip()][:10]
        for i, line in enumerate(lines):
            if not any(k in line for k in kw):
                continue
            if len(line) > 35 or len(line) < 6:
                continue
            if any(x in line for x in exclude_words):
                continue
            if re.search(r'[A-Za-z]{3,}', line):
                continue
            if re.match(r'^第[一二三四五六七八九十百千\d]+[条章节]', line):
                continue
            title = line
            # 标题被PDF分成两行时拼接前一行（纯中文短行）
            if i > 0:
                prev = lines[i - 1]
                if (4 <= len(prev) <= 20 and not re.search(r'[A-Za-z]', prev)
                        and not any(k in prev for k in kw)
                        and not re.search(r'发布|实施|施行|20\d\d', prev)
                        and not re.match(r'^第[一二三四五六七八九十百千\d]+[条章节]', prev)):
                    title = prev + title
            return title[:50]
        return None

    @staticmethod
    def _join_articles(articles):
        """多条合并的条款号列表 -> 紧凑字符串"""
        articles = [a for a in articles if a]
        if not articles:
            return ""
        if len(articles) <= 3:
            return "、".join(articles)
        return "、".join(articles[:3]) + "等"

    @staticmethod
    def _local_scenes(content):
        """按场景关键词给文档块打场景标签（本地免API）"""
        scenes = []
        for scene, keywords in SCENE_KEYWORDS.items():
            if any(kw in content for kw in keywords):
                scenes.append(scene)
        return scenes

    def _clean_text(self, text):
        """清理PDF提取的文本"""
        # 移除多余空格
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        # 合并段落（简单规则：行尾不是句号、分号等标点的可能是换行导致的断句）
        result = []
        for i, line in enumerate(cleaned_lines):
            if i == 0:
                result.append(line)
            else:
                prev = result[-1]
                # 如果前一行以标点结尾，或者当前行以数字/大写开头（新段落），则换行
                if (prev and prev[-1] in '。；：！？、，.])）】》' 
                    or re.match(r'^[（(]?[一二三四五六七八九十\d]+[、.）)]', line)):
                    result.append(line)
                else:
                    # 否则合并（PDF中常见的换行断句）
                    result[-1] = prev + line
        
        return '\n'.join(result)

    @staticmethod
    def _as_result(doc, score, mode):
        """统一结果结构（检索与浏览共用）"""
        return {
            "title": doc["title"],
            "official_title": doc.get("official_title") or official_name(doc["title"]),
            "page": doc.get("page", 0),
            "content": doc.get("content", ""),
            "article": doc.get("article", ""),
            "scenes": doc.get("scenes", []) or [],
            "similarity": float(score),
            "mode": mode,
        }

    def search(self, query, top_k=5, scenes=None, titles=None, min_score=0.0):
        """
        搜索相关知识库内容（自动选择向量检索或关键词检索）

        scenes:  场景过滤（只在这些场景的文档块中检索；无匹配时回退全库）
        titles:  文档过滤（只在指定文档标题的块中检索）
        min_score: 相似度下限，低于该值的结果丢弃
        """
        if not self.documents:
            return []

        candidates = self.documents
        if titles:
            title_set = set(titles)
            filtered = [d for d in candidates if d.get("title") in title_set]
            if filtered:
                candidates = filtered
        if scenes:
            scene_set = set(scenes)
            filtered = [d for d in candidates if d.get("scenes") and scene_set.intersection(d["scenes"])]
            if filtered:
                candidates = filtered

        # 检查是否有可用的embedding
        has_embeddings = any(doc.get("has_embedding") for doc in candidates)

        if has_embeddings:
            results = self._vector_search(query, top_k, candidates)
            if results:
                return [r for r in results if r["similarity"] >= min_score]

        # 降级到关键词检索
        results = self._keyword_search(query, top_k, candidates)
        return [r for r in results if r["similarity"] >= min_score]

    def all_scenes(self):
        """汇总全部场景标签及其文档块数量（用于界面筛选）"""
        counter = {}
        for doc in self.documents:
            for scene in (doc.get("scenes") or []):
                counter[scene] = counter.get(scene, 0) + 1
        return dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))

    def stats(self):
        """知识库统计信息"""
        docs = self.list_documents()
        vec = sum(1 for d in self.documents if d.get("has_embedding"))
        return {
            "chunks": len(self.documents),
            "documents": len(docs),
            "vector_chunks": vec,
            "scenes": len(self.all_scenes()),
            "mode": "vector" if vec > 0 else "keyword",
        }

    def list_chunks(self, title=None, scene=None, keyword=None, offset=0, limit=50):
        """按文档/场景/关键词浏览条款块（不打分，按文档与页码排序）"""
        items = self.documents
        if title:
            items = [d for d in items if d.get("title") == title]
        if scene:
            items = [d for d in items if scene in (d.get("scenes") or [])]
        if keyword:
            kw = keyword.strip()
            if kw:
                items = [d for d in items if kw in (d.get("content") or "")]

        items = sorted(items, key=lambda d: (str(d.get("title", "")), d.get("page", 0)))
        total = len(items)
        page_items = items[offset:offset + limit]
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": [self._as_result(d, 0.0, "browse") for d in page_items],
        }

    def _vector_search(self, query, top_k, candidates):
        """向量相似度检索（本地 bge 模型编码查询）"""
        from core.embedder import embed
        query_embedding = embed(query, is_query=True)
        if not query_embedding:
            return []

        query_vec = np.array(query_embedding)
        q_norm = float(np.linalg.norm(query_vec))
        if not np.isfinite(q_norm) or q_norm <= 0:
            # 零范数/非有限查询向量：余弦分母为 0 会得到 NaN/inf（numpy 只告警不抛异常），
            # 之后 NaN >= min_score 恒为 False，整库结果会被静默丢空 → 直接降级走关键词检索
            print("[kb] 查询向量无效（范数为 0 或非有限），降级为关键词检索")
            return []

        similarities = []
        for doc in candidates:
            if not doc.get("has_embedding") or not doc.get("embedding"):
                continue
            doc_vec = np.array(doc["embedding"])
            if doc_vec.shape != query_vec.shape:
                continue      # 维度不一致（换过 embedding 模型的历史块）→ 跳过，别让 dot 抛异常
            d_norm = float(np.linalg.norm(doc_vec))
            if not np.isfinite(d_norm) or d_norm <= 0:
                continue      # 退化向量：分母为 0 → NaN，跳过
            similarity = float(np.dot(query_vec, doc_vec)) / (q_norm * d_norm)
            if not np.isfinite(similarity):
                continue
            similarities.append((similarity, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [self._as_result(doc, sim, "vector") for sim, doc in similarities[:top_k]]

    def _keyword_search(self, query, top_k, candidates):
        """关键词匹配检索（降级方案）"""
        # 提取查询中的关键词
        keywords = self._extract_keywords(query)

        if not keywords:
            # 没有关键词就返回前几条
            return [self._as_result(doc, 0.5, "keyword") for doc in candidates[:top_k]]

        # 计算每个文档的关键词匹配得分
        scores = []
        for doc in candidates:
            content = doc["content"]
            score = 0
            matched = 0
            for kw in keywords:
                count = content.count(kw)
                if count > 0:
                    score += count * len(kw)  # 长词权重更高
                    matched += 1

            if matched > 0:
                # 匹配度 = 匹配关键词数 / 总关键词数
                match_ratio = matched / len(keywords)
                final_score = match_ratio * 0.6 + min(score / 100, 0.4)
                scores.append((final_score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [self._as_result(doc, score, "keyword") for score, doc in scores[:top_k]]

    def _extract_keywords(self, query):
        """从查询中提取关键词（简单实现）"""
        query = query or ""      # 界面/调用方可能传 None → 后续 len() 会 TypeError
        # 移除常见停用词
        stop_words = set([
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '或',
            '在', '有', '不', '这', '那', '也', '都', '要', '会', '能',
            '可以', '可能', '应该', '必须', '需要', '什么', '怎么', '如何',
            '为什么', '哪些', '哪里', '哪个', '多少', '几',
            '安全', '隐患', '违章', '规定', '标准', '要求', '作业',
        ])
        
        # 简单分词：按标点和空格切分 + 2-4字的连续中文
        words = set()
        
        # 2-4字滑窗提取
        for length in [4, 3, 2]:
            for i in range(len(query) - length + 1):
                word = query[i:i+length]
                if word not in stop_words and len(word.strip()) >= 2:
                    words.add(word)
        
        # 也保留原查询中的关键短语
        phrases = re.split(r'[，。！？、；：\s]+', query)
        for p in phrases:
            if 2 <= len(p) <= 20 and p not in stop_words:
                words.add(p)
        
        return list(words)

    def get_context_text(self, query, top_k=3, scenes=None):
        """获取用于RAG的上下文文本（来源标注为《官方文件名》条款号）"""
        results = self.search(query, top_k, scenes=scenes)
        if not results:
            return ""

        context_parts = []
        for r in results:
            name = r.get('official_title') or official_name(r['title'])
            article = r.get('article', '')
            if article:
                src = f"《{name}》{article}"
            else:
                src = f"《{name}》第{r['page']}页"
            context_parts.append(f"【来源：{src}】\n{r['content']}")

        return "\n\n".join(context_parts)

    def ai_classify_all(self, progress_callback=None, batch_size=20, skip_classified=True):
        """AI按预设场景分类文档块（增量模式：默认跳过已分类块，只处理新增PDF的块）
        返回本次实际分类的块数；某批API失败时该批不标记，下次自动重试
        """
        from core.api_client import get_ai_client
        ai_client = get_ai_client()

        if skip_classified:
            docs = [d for d in self.documents if not d.get("ai_classified")]
        else:
            docs = self.documents

        total = len(docs)
        if total == 0:
            if progress_callback:
                progress_callback(100, "没有需要分类的新增文档块")
            return 0

        classified = 0
        for start in range(0, total, batch_size):
            batch = docs[start:start + batch_size]
            result = ai_client.classify_scenes_batch([d["content"] for d in batch], SCENES)
            if result:
                for d, scenes in zip(batch, result):
                    d["ai_classified"] = True
                    if scenes:
                        d["scenes"] = scenes
                classified += len(batch)
            if progress_callback:
                progress_callback(
                    int((start + len(batch)) / max(total, 1) * 100),
                    f"AI场景分类中... {start + len(batch)}/{total} 块"
                )

        self._save_index()
        if progress_callback:
            progress_callback(100, "AI场景分类完成")
        return classified

    def auto_import_folder(self, progress_callback=None):
        """自动导入「知识库」文件夹中尚未索引的PDF（按文件名判断是否已索引）"""
        folder = os.path.join(get_app_dir(), "知识库")
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception:
            return []

        indexed = set(self.list_documents())
        pdf_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
        results = []
        for i, fname in enumerate(pdf_files):
            if fname in indexed:
                continue
            if progress_callback:
                pct = int((i / max(len(pdf_files), 1)) * 100)
                progress_callback(pct, f"自动导入知识库：{fname}")
            try:
                r = self.add_pdf(os.path.join(folder, fname), fname)
                results.append({
                    "title": fname,
                    "success": r.get("success", False),
                    "count": r.get("count", 0),
                    "error": r.get("error"),
                })
            except Exception as e:
                results.append({"title": fname, "success": False, "count": 0, "error": str(e)})
        if progress_callback:
            progress_callback(100, "知识库导入完成")
        return results

    def list_documents(self):
        """列出所有文档标题"""
        titles = set()
        for doc in self.documents:
            titles.add(doc["title"])
        return list(titles)

    def get_document_info(self, title):
        """获取文档信息"""
        chunks = [d for d in self.documents if d["title"] == title]
        if not chunks:
            return None
        has_embedding = any(c.get("has_embedding") for c in chunks)
        return {
            "title": title,
            "chunks": len(chunks),
            "mode": "vector" if has_embedding else "keyword"
        }

    def remove_document(self, title, delete_file=True):
        """删除指定文档（默认同步删除「知识库」文件夹中的PDF源文件）"""
        original_count = len(self.documents)
        self.documents = [d for d in self.documents if d["title"] != title]
        self._save_index()
        removed = original_count - len(self.documents)

        if removed > 0 and delete_file:
            try:
                # 只用文件名（basename）拼接，防止 title 含路径分隔符时越界删除
                # 「知识库」目录之外的文件
                kb_folder = os.path.join(get_app_dir(), "知识库")
                f = os.path.join(kb_folder, os.path.basename(title))
                if os.path.exists(f):
                    os.remove(f)
                else:
                    print(f"[知识库] 未在「知识库」目录找到对应 PDF（{title}），"
                          f"仅从索引中移除，磁盘文件未删除")
            except Exception as e:
                print(f"删除知识库PDF文件失败: {e}")
        return removed

    def clear(self):
        """清空知识库"""
        self.documents = []
        self._save_index()

    def count(self):
        """返回文档块数量"""
        return len(self.documents)


_knowledge_base = None
_kb_singleton_lock = threading.Lock()


def get_knowledge_base():
    """获取知识库单例（加锁）。

    索引文件约 15.7MB，解析耗时约 0.2s。启动路径不再预先构造它（改为首次
    真正需要时加载，可由后台线程触发），因此可能出现"主线程打开知识库页"与
    "后台自动导入线程"同时首次构造的竞争——加锁保证只构造一次。
    """
    global _knowledge_base
    if _knowledge_base is None:
        with _kb_singleton_lock:
            if _knowledge_base is None:
                _knowledge_base = KnowledgeBase()
    return _knowledge_base
