#!/usr/bin/env python3
"""
dump_project.py —— 项目全量快照工具
====================================

【用途】
    把项目的文件树 + 所有文本文件内容，导出成一个 Markdown 文件。
    便于喂给 LLM（如 ChatGPT/Claude）做代码分析，或归档。

【用法】
    cd <项目根目录>
    python dump_project.py                    # 默认输出 项目快照.md
    python dump_project.py -o 我的快照.md     # 自定义输出文件
    python dump_project.py -r D:\\其他\\路径   # 指定根目录
    python dump_project.py --max-size 200     # 单文件大小上限（KB）

【设计原则】
    1. 安全优先：显式黑名单排除 .env 等敏感文件
    2. 性能优先：os.walk 剪枝，不进入无关目录
    3. 鲁棒性：跳过二进制、超大、不可读文件并记录警告
    4. 单次遍历：文件树和内容同时收集，避免不一致
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


# =============================================================================
# 配置区（可通过命令行覆盖）
# =============================================================================

# 默认输出文件名（中文，便于识别）
DEFAULT_OUTPUT_NAME = "项目快照.md"

# 排除的目录名（支持 *.egg-info 这类通配）
EXCLUDE_DIRS: set[str] = {
    # 版本控制 & 环境
    ".git", ".svn", ".hg",
    # Python
    ".venv", "venv", "env", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "*.egg-info",
    # Node
    "node_modules", ".next", ".nuxt", "dist", "build",
    # IDE & OS
    ".idea", ".vscode", ".settings",
    # 数据 & 缓存（可按需删除）
    "data", "datasets", "uploads", "uploaded_images",
    ".cache", "logs",
}

# 精确排除的文件名（黑名单，最高优先级，必须排除）
EXCLUDE_FILES: set[str] = {
    ".env", ".env.local", ".env.production", ".env.development",
    # 各种 dump 产物（含中英文）
    "project_dump.md", "项目快照.md", "项目全量导出.md",
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock",
}

# 包含的文件扩展名
INCLUDE_EXTS: set[str] = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml",
    ".ini", ".cfg", ".sql", ".sh", ".bat", ".ps1",
    ".js", ".ts", ".jsx", ".tsx", ".html", ".css", ".scss",
    ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp",
}

# 包含的特殊文件名（无扩展名，但需要 dump）
INCLUDE_NAMES: set[str] = {
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    "Makefile", ".env.example", ".gitignore", ".dockerignore",
}

# 后缀 → Markdown 语言标记
LANG_MAP: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "jsx", ".tsx": "tsx", ".md": "markdown",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".toml": "toml", ".ini": "ini", ".cfg": "ini", ".sql": "sql",
    ".sh": "bash", ".bat": "batch", ".ps1": "powershell",
    ".html": "html", ".css": "css", ".scss": "scss",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
    "Dockerfile": "dockerfile",
    ".gitignore": "gitignore",
    ".dockerignore": "dockerignore",
    "Makefile": "makefile",
}

# 单文件大小上限（KB）—— 超过则跳过
DEFAULT_MAX_SIZE_KB = 1024  # 1 MB


# =============================================================================
# 工具函数
# =============================================================================

def _match_dir(name: str, patterns: set[str]) -> bool:
    """支持 `*.egg-info` 这类通配匹配。"""
    for p in patterns:
        if p.startswith("*") and name.endswith(p[1:]):
            return True
        if name == p:
            return True
    return False


def _is_text_file(path: Path, sample_size: int = 8192) -> bool:
    """
    判断文件是否为文本（粗略算法：读前 8KB，看是否含 NUL 字节）。
    含 NUL 字节 → 二进制。
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" not in chunk
    except OSError:
        return False


def _get_lang(path: Path) -> str:
    """返回 Markdown 代码块的语言标记。"""
    if path.name in LANG_MAP:
        return LANG_MAP[path.name]
    return LANG_MAP.get(path.suffix.lower(), "")


def _read_text(path: Path) -> str | None:
    """
    尝试多种编码读取文件，失败返回 None。
    编码顺序：UTF-8 → GBK → Latin-1（永远成功）
    """
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


# =============================================================================
# 核心逻辑
# =============================================================================

def collect_files(
    root: Path,
    exclude_dirs: set[str],
    exclude_files: set[str],
    include_exts: set[str],
    include_names: set[str],
    max_size_kb: int,
) -> list[Path]:
    """
    单次遍历收集所有待 dump 的文件（已排序）。

    ✅ 关键优化：用 os.walk + dirnames[:] 原地剪枝，
       被排除的目录**根本不会被进入**（不浪费 IO）。
    """
    result: list[Path] = []
    skipped_big: list[tuple[str, int]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)

        # ✅ 原地剪枝：不进入被排除的目录
        dirnames[:] = [d for d in dirnames if not _match_dir(d, exclude_dirs)]

        for fname in filenames:
            fpath = cur / fname

            # 1. 精确文件名黑名单（最高优先级）
            if fname in exclude_files:
                continue

            # 2. 扩展名/特殊文件名判断
            if fpath.suffix.lower() not in include_exts and fname not in include_names:
                continue

            # 3. 文件大小限制
            try:
                size_kb = fpath.stat().st_size / 1024
            except OSError:
                continue
            if size_kb > max_size_kb:
                skipped_big.append((str(fpath.relative_to(root)), int(size_kb)))
                continue

            # 4. 二进制文件跳过
            if not _is_text_file(fpath):
                continue

            result.append(fpath)

    # 输出跳过警告
    if skipped_big:
        print(f"⚠️  跳过 {len(skipped_big)} 个大文件 (> {max_size_kb} KB):")
        for name, kb in skipped_big[:10]:
            print(f"    - {name} ({kb} KB)")
        if len(skipped_big) > 10:
            print(f"    ... 还有 {len(skipped_big) - 10} 个")

    return sorted(result)


def build_markdown(root: Path, files: list[Path], elapsed: float) -> str:
    """组装最终的 Markdown 输出。"""
    lines: list[str] = []

    # ---------- 头部元信息 ----------
    lines.append(f"# Project Dump: {root.name}")
    lines.append("")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 根目录:   `{root}`")
    lines.append(f"> 文件总数: {len(files)}")
    lines.append(f"> 生成耗时: {elapsed:.2f}s")
    lines.append("")

    # ---------- 文件树 ----------
    lines.append("## File Tree")
    lines.append("```text")
    for p in files:
        lines.append(str(p.relative_to(root)).replace(os.sep, "/"))
    lines.append("```")
    lines.append("")

    # ---------- 文件内容 ----------
    lines.append("## File Contents")
    lines.append("")

    failed: list[str] = []
    for p in files:
        rel = str(p.relative_to(root)).replace(os.sep, "/")
        text = _read_text(p)

        lines.append(f"### {rel}")
        if text is None:
            lines.append("```text")
            lines.append("[无法读取该文件]")
            lines.append("```")
            failed.append(rel)
        else:
            lang = _get_lang(p)
            lines.append(f"```{lang}")
            lines.append(text)
            lines.append("```")
        lines.append("")

    if failed:
        lines.append("---")
        lines.append("")
        lines.append(f"⚠️ 以下 {len(failed)} 个文件读取失败：")
        for f in failed:
            lines.append(f"- {f}")

    return "\n".join(lines)


# =============================================================================
# CLI 入口
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="把项目导出成单个 Markdown 文件（用于喂给 LLM 或归档）"
    )
    parser.add_argument(
        "-r", "--root", type=Path, default=Path.cwd(),
        help="项目根目录（默认：当前工作目录）"
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help=f"输出文件名（默认：{DEFAULT_OUTPUT_NAME}）"
    )
    parser.add_argument(
        "--max-size", type=int, default=DEFAULT_MAX_SIZE_KB,
        help=f"单文件大小上限 KB（默认：{DEFAULT_MAX_SIZE_KB}）"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"❌ 目录不存在: {root}", file=sys.stderr)
        return 1

    # ✅ 输出文件名默认用中文
    out = root / (args.output or DEFAULT_OUTPUT_NAME)
    # 输出文件本身加入黑名单（防止递归）
    exclude_files = EXCLUDE_FILES | {out.name}

    print(f"🔍 开始扫描: {root}")
    t0 = time.time()

    files = collect_files(
        root=root,
        exclude_dirs=EXCLUDE_DIRS,
        exclude_files=exclude_files,
        include_exts=INCLUDE_EXTS,
        include_names=INCLUDE_NAMES,
        max_size_kb=args.max_size,
    )

    content = build_markdown(root, files, time.time() - t0)

    try:
        out.write_text(content, encoding="utf-8")
    except OSError as e:
        print(f"❌ 写入失败: {e}", file=sys.stderr)
        return 1

    size_kb = out.stat().st_size / 1024
    print(f"✅ 已生成 {out}  ({size_kb:.1f} KB, {len(files)} 文件, {time.time() - t0:.2f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())