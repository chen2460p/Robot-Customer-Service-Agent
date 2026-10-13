"""
================================================================================
文件：bench_rag.py
职责：RAG 链路的【性能基准脚本】——产出可复现、可写进简历的真实指标。

测量内容：
    1. 知识库构建阶段：每个文件的切片数、冷启动全量入库耗时、向量数
    2. 增量入库阶段：二次运行（全部命中 MD5）耗时与跳过率
    3. 检索阶段：向量检索延迟 P50 / P95（含 Embedding 调用 + 相似度搜索）
    4. 端到端阶段：RAG 总结延迟（检索 + 大模型总结），即用户实际等待时间

用法：
    python bench_rag.py                        # 使用项目真实配置（会写入真实向量库）
    python bench_rag.py --persist-dir tmp_dir  # 隔离到指定目录，可反复复现"冷启动"
    python bench_rag.py --queries 50 --e2e 5   # 自定义采样次数

为什么脚本要支持 --persist-dir？
    向量库是持久化的，第二次运行必然"已入库"，测不到冷启动耗时。
    指定一个全新的目录即可复现冷启动，且不会污染真实向量库。
================================================================================
"""

import argparse
import os
import shutil
import statistics
import sys
import time

# 强制用 UTF-8 输出，避免 Windows 控制台默认 GBK 导致中文指标乱码
sys.stdout.reconfigure(encoding="utf-8")

# 采样用的典型业务问题（覆盖选购 / 故障 / 维护 / 环境适配四类意图）
QUERIES = [
    "小户型适合哪些扫地机器人",
    "扫地机器人回不了充电座怎么办",
    "边刷多久需要更换一次",
    "木地板和瓷砖分别怎么设置吸力",
    "扫地机器人迷路了怎么处理",
    "拖布怎么清洗才不容易发臭",
    "养猫家庭选扫地机器人要注意什么",
    "扫地机器人工作时噪音大正常吗",
    "尘盒多久清理一次比较合适",
    "扫拖一体机器人拖地水量怎么调",
    "扫地机器人能清理地毯吗",
    "电池续航变短了是什么原因",
    "滤网可以用水洗吗",
    "扫地机器人适合多大面积的房子",
    "机器人被电线缠住了经常报错",
    "新买的机器人需要先充满电吗",
    "扫地机器人能识别宠物粪便吗",
    "清扫覆盖率低是什么原因",
    "扫地机器人需要定期保养哪些部件",
    "扫地机器人过门槛困难怎么办",
]


def count_vectors(vs) -> int:
    """读取当前集合内的向量条数——用来判断这次运行是"冷启动"还是"增量"。"""
    try:
        return vs.vector_store._collection.count()
    except Exception:
        # 兜底：_collection 属于内部实现，若不可用则退回 get()
        return len(vs.vector_store.get().get("ids", []))


def stat_line(name: str, samples: list[float]) -> str:
    """把一组耗时样本格式化成 均值 / P50 / P95 / 极值 的一行摘要。"""
    ordered = sorted(samples)
    # P95 用"最近邻序位"取整，样本少时不至于越界
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return (f"{name:16s} 均值 {statistics.mean(samples) * 1000:8.1f} ms | "
            f"P50 {statistics.median(ordered) * 1000:8.1f} ms | "
            f"P95 {p95 * 1000:8.1f} ms | "
            f"最快 {ordered[0] * 1000:7.1f} ms | "
            f"最慢 {ordered[-1] * 1000:7.1f} ms")


def main():
    parser = argparse.ArgumentParser(description="RAG 链路性能基准")
    parser.add_argument("--persist-dir", default=None,
                        help="向量库持久化目录；指定则隔离测试，不污染真实库")
    parser.add_argument("--queries", type=int, default=20, help="检索延迟采样次数")
    parser.add_argument("--e2e", type=int, default=3, help="端到端 RAG 总结采样次数")
    args = parser.parse_args()

    # ---- 必须在构造 VectorStoreService 之前改配置，__init__ 里才读到新值 ----
    if args.persist_dir:
        from utils import config_handler
        # 用绝对路径，避免再次踩到"相对路径受 CWD 影响"的坑
        config_handler.chroma_conf["persist_directory"] = os.path.abspath(args.persist_dir)
        if os.path.exists(args.persist_dir):
            shutil.rmtree(args.persist_dir)   # 清空以获得真正的冷启动
        print(f"[隔离模式] 向量库将写入并清空于: {args.persist_dir}")

    from rag.vector_store import VectorStoreService
    from utils.config_handler import chroma_conf
    from utils.path_tool import get_abs_path

    vs = VectorStoreService()
    print(f"\n向量库目录 : {vs.persist_dir}")
    print(f"去重记录   : {vs._md5_store_path()}")
    print(f"集合名称   : {chroma_conf['collection_name']}")
    print(f"分片参数   : chunk_size={chroma_conf['chunk_size']}, "
          f"chunk_overlap={chroma_conf['chunk_overlap']}, top_k={chroma_conf['k']}")

    before = count_vectors(vs)
    print(f"入库前向量数: {before}  -> {'冷启动（全量入库）' if before == 0 else '增量运行（可跳过已入库文件）'}")

    # ================= 1. 每个文件的切片数（纯本地计算，不消耗 API） =================
    from utils.file_handler import (pdf_loader, txt_loader, listdir_with_allowed_type,
                                    get_file_md5_hex)

    data_dir = get_abs_path(chroma_conf["data_path"])
    files = listdir_with_allowed_type(data_dir, tuple(chroma_conf["allow_knowledge_file_type"]))

    print(f"\n{'=' * 78}\n【1】知识库文件构成\n{'=' * 78}")
    print(f"{'文件':<34}{'加载Document':>12}{'切片数':>10}{'字符数':>10}")
    total_slices = total_chars = 0
    for path in files:
        docs = pdf_loader(path) if path.endswith("pdf") else txt_loader(path)
        slices = vs.spliter.split_documents(docs)
        chars = sum(len(d.page_content) for d in docs)
        total_slices += len(slices)
        total_chars += chars
        print(f"{os.path.basename(path):<34}{len(docs):>12}{len(slices):>10}{chars:>10}")
    print(f"{'合计':<34}{'':>12}{total_slices:>10}{total_chars:>10}")
    print(f"平均切片长度: {total_chars / total_slices:.0f} 字符/片")

    # ================= 2. 入库耗时 =================
    print(f"\n{'=' * 78}\n【2】知识库入库耗时\n{'=' * 78}")

    t0 = time.perf_counter()
    vs.load_document()
    cold_elapsed = time.perf_counter() - t0

    after = count_vectors(vs)
    new_vectors = after - before
    print(f"首次入库   : {cold_elapsed:8.2f} s  | 新增向量 {new_vectors} 条")
    if new_vectors > 0:
        print(f"入库吞吐   : {new_vectors / cold_elapsed:8.1f} 向量/秒")

    # 二次运行：所有文件应命中 MD5 记录而被跳过
    t0 = time.perf_counter()
    vs.load_document()
    warm_elapsed = time.perf_counter() - t0

    # 【实测跳过率】逐文件重新算 MD5 并与去重记录比对，得到真实命中数，
    # 而不是想当然地认为"全都跳过了"。
    recorded_path = vs._md5_store_path()
    recorded = set()
    if os.path.exists(recorded_path):
        with open(recorded_path, encoding="utf-8") as f:
            recorded = {line.strip() for line in f if line.strip()}
    skipped = sum(1 for p in files if get_file_md5_hex(p) in recorded)

    print(f"增量入库   : {warm_elapsed * 1000:8.1f} ms | 向量数不变（{count_vectors(vs)} 条）")
    print(f"文件跳过率 : {skipped}/{len(files)} = {skipped / len(files) * 100:.0f}%"
          f"   | 冷启动 {cold_elapsed:.2f} s → 增量 {warm_elapsed * 1000:.1f} ms")

    # ================= 3. 纯检索延迟 =================
    print(f"\n{'=' * 78}\n【3】向量检索延迟（Embedding + 相似度搜索）\n{'=' * 78}")
    retriever = vs.get_retriever()
    retriever.invoke("预热查询，不计入统计")   # 预热：排除首次连接建立的固定开销

    latencies, doc_counts = [], []
    for q in QUERIES[:args.queries]:
        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        latencies.append(time.perf_counter() - t0)
        doc_counts.append(len(docs))

    print(stat_line("检索延迟", latencies))
    print(f"{'召回情况':16s} 平均召回 {statistics.mean(doc_counts):.1f} 条 / top_k={chroma_conf['k']}")

    # ================= 4. 端到端：检索 + 大模型总结 =================
    print(f"\n{'=' * 78}\n【4】端到端 RAG 回答延迟（检索 + qwen3-max 总结）\n{'=' * 78}")
    if args.e2e > 0:
        import contextlib
        import io

        from rag.rag_service import RagSummarizeService
        rag = RagSummarizeService()   # 内部会复用同一个向量库目录
        e2e = []
        for q in QUERIES[:args.e2e]:
            t0 = time.perf_counter()
            # rag_service 里的 print_prompt 是开发期调试钩子，会打印完整提示词，
            # 这里临时把 stdout 重定向掉，避免它淹没基准输出（只静音，不影响被测逻辑）
            with contextlib.redirect_stdout(io.StringIO()):
                answer = rag.rag_summarize(q)
            e2e.append(time.perf_counter() - t0)
            print(f"  Q: {q}\n  A: {answer[:70].replace(chr(10), ' ')}...\n")
        print(stat_line("端到端延迟", e2e))

    print(f"\n{'=' * 78}\n完成。向量库当前共 {count_vectors(vs)} 条向量。\n{'=' * 78}")


if __name__ == "__main__":
    main()
