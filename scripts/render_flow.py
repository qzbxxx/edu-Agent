# scripts/render_flow.py
# 生成 docs/流程图.svg：只依赖标准库，改完布局重跑本脚本即可。
from html import escape
from pathlib import Path

W, H = 1840, 1580
BASE = Path(__file__).resolve().parent.parent

COLORS = {
    "blue": "#4a7fb5",
    "red": "#b85c52",
    "purple": "#7b5ea7",
    "green": "#4f9464",
    "yellow": "#b8912f",
    "gray": "#6b7280",
}

parts = [
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
    "<defs>",
]
for name, color in COLORS.items():
    parts.append(
        f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
    )
parts.append("</defs>")
parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')


def txt(x, y, s, size=14, color="#243447", weight="normal", anchor="middle", halo=False):
    stroke = ' stroke="#ffffff" stroke-width="3" paint-order="stroke"' if halo else ""
    return (
        f'<text x="{x}" y="{y}" font-family="Microsoft YaHei, Segoe UI, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}"{stroke}>{escape(str(s))}</text>'
    )


def box(x, y, w, h, fill, stroke, dash=None):
    dash_attr = ' stroke-dasharray="7 5"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"{dash_attr}/>'
    )


def node(x, y, w, h, title, lines, fill, stroke, title_color="#102a43",
         size=14, line_h=22, start=58):
    out = [box(x, y, w, h, fill, stroke), txt(x + w / 2, y + 31, title, 18, title_color, "bold")]
    cy = y + start
    for line in lines:
        if isinstance(line, tuple):
            text, color = line
        else:
            text, color = line, "#243447"
        out.append(txt(x + w / 2, cy, text, size, color))
        cy += line_h
    return out


def end_node(x, y, w, h, title, sub):
    return [
        box(x, y, w, h, "#eaf7ee", "#4f9464"),
        txt(x + w / 2, y + 31, title, 18, "#1f5132", "bold"),
        txt(x + w / 2, y + 63, sub, 14, "#2f6b45"),
    ]


def diamond(cx, cy, w, h, title, sub):
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    return [
        f'<polygon points="{pts}" fill="#fff8e1" stroke="#b8912f" stroke-width="1.8"/>',
        txt(cx, cy - 6, title, 17, "#4a3a12", "bold"),
        txt(cx, cy + 21, sub, 13, "#6b5a2a"),
    ]


def arrow(points, color, label=None, label_xy=None, dashed=False):
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    dash_attr = ' stroke-dasharray="7 5"' if dashed else ""
    out = [
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" '
        f'stroke-linejoin="round" marker-end="url(#arrow-{color_key(color)})"{dash_attr}/>'
    ]
    if label and label_xy:
        out.append(txt(label_xy[0], label_xy[1], label, 13, color, "bold", halo=True))
    return out


def color_key(color):
    for name, value in COLORS.items():
        if value == color:
            return name
    return "gray"


# ---------- 标题 ----------
parts.append(txt(60, 58, "教育客服 Agent：一次提问从进入到回答的完整链路", 34, "#0f2b46", "bold", "start"))
parts.append(txt(60, 92, "LangGraph 状态机 · 混合检索 RAG · 工具调用 · 人工审批 · 长期记忆", 15, "#5b6b7c", "normal", "start"))

# ---------- 顶部主链路 ----------
parts.append(box(620, 120, 600, 86, "#eef4fa", "#4a7fb5"))
parts.append(txt(920, 153, "学员提问", 20, "#173a5e", "bold"))
parts.append(txt(920, 181, "CLI: interactive_cli.py ｜ HTTP: POST /api/chat", 14, "#31577a"))
parts += arrow([(920, 206), (920, 246)], COLORS["blue"])

parts += node(620, 246, 600, 92, "① load_memory_node", [
    "读画像 user_profiles（长期记忆库）",
    ("清空本轮 tool_events；画像写入 State", "#2f6f5e"),
], "#eef4fa", "#4a7fb5")

parts += arrow([(920, 338), (920, 370)], COLORS["blue"])
parts += diamond(920, 430, 320, 120, "② router_node", "咨询还是退款？")

# ---------- 咨询链路 ----------
parts += arrow([(760, 430), (360, 430), (360, 520)], COLORS["blue"], "consult", (545, 416))
parts += node(100, 520, 520, 100, "③ query_rewrite_node", [
    "多轮消解代词（1000 token 窗口）",
    "首轮无指代 → 直通",
], "#eef4fa", "#4a7fb5")

parts += arrow([(360, 620), (360, 660)], COLORS["blue"])
parts += node(100, 660, 520, 220, "④ consult_rag_node", [
    "BM25 + 向量各 Top10",
    "→ RRF 融合前 12 → 覆盖率精排 0.8 : 0.2",
    "→ 阈值 0.35 过滤 → 取 Top4",
    ("注入：学员画像 + Top3 情景记忆", "#2f6f5e"),
    "＋ 最近 5 条对话 → 生成答案",
    ("无命中 → 固定话术转人工，不调模型", "#6b7280"),
], "#eef4fa", "#4a7fb5")

parts += arrow([(360, 880), (360, 915)], COLORS["blue"])
parts += node(100, 915, 520, 104, "⑤ memory_extractor_node", [
    "抽取画像并合并写入",
    ("覆盖旧值 → 记 profile_change 事件", "#2f6f5e"),
], "#eef4fa", "#4a7fb5")

parts += arrow([(360, 1019), (360, 1054)], COLORS["blue"])
parts += end_node(100, 1054, 520, 90, "END 咨询", "答案 + 引用 [《文件》第 X 页]")

# ---------- 售后链路 ----------
parts += arrow([(1080, 430), (1450, 430), (1450, 520)], COLORS["red"], "refund / 其他", (1265, 416))
parts += node(1180, 520, 540, 186, "④' after_sales_agent_node", [
    "LLM 自主选择并调用工具",
    "4 个订单：list / query / policy / submit",
    "3 个记忆：get_profile / save / search",
    ("画像进 System Prompt；最多 4 轮", "#2f6f5e"),
    ("明确退款未提交 → 确定性兜底", "#6b7280"),
], "#fdf0ef", "#b85c52")

parts += arrow([(1450, 706), (1450, 742)], COLORS["red"])
parts += diamond(1450, 800, 340, 116, "已登记退款申请？", "refund_requested")

parts += arrow([(1280, 800), (1000, 800)], COLORS["red"], "否", (1140, 786))
parts += end_node(700, 746, 300, 96, "END 售后", "直接回答（不退款）")

parts += arrow([(1450, 858), (1450, 896)], COLORS["red"], "是", (1466, 881))
parts += diamond(1450, 954, 360, 116, "需要人工审批？", "超 7 天 或 进度 > 10%")

parts += arrow([(1450, 1012), (1450, 1050)], COLORS["red"], "是（高风险）", (1466, 1035))
parts += node(1180, 1050, 540, 118, "human_approval_node", [
    "interrupt() 挂起本轮，状态落盘",
    "POST /api/audit/resume 恢复",
    ("驳回 → 写记忆 + END 驳回话术", "#6b5a2a"),
], "#f3eefb", "#7b5ea7", title_color="#3f2a63", size=13, line_h=19, start=52)

# 低风险绕过人工，沿右侧走到打款
parts += arrow(
    [(1630, 954), (1760, 954), (1760, 1240), (1720, 1240)],
    COLORS["red"], "否（低风险）", (1645, 940),
)

parts += arrow([(1450, 1168), (1450, 1218)], COLORS["purple"], "approved", (1466, 1195))
parts += node(1180, 1218, 540, 130, "⑥ execute_refund_node", [
    "二次校验归属 → 预占 PROCESSING",
    "订单 REFUNDED → 标记 SUCCESS",
    ("成功 → 写一条退款记忆", "#2f6f5e"),
], "#fdf0ef", "#b85c52")

parts += arrow([(1180, 1109), (1000, 1109)], COLORS["red"], "rejected", (1090, 1095))
parts += end_node(700, 1050, 300, 96, "END 驳回", "「未通过人工审核」")

parts += arrow([(1450, 1348), (1450, 1368)], COLORS["green"])
parts += end_node(1180, 1368, 540, 84, "END 退款", "退款成功，金额原路退回")

# ---------- 左下角：长期记忆说明 ----------
parts.append(box(100, 1180, 520, 210, "#f4faf7", "#4f9d8a", dash=True))
parts.append(txt(360, 1216, "长期记忆库 long_term_memory.db", 17, "#1f5132", "bold"))
memory_lines = [
    "user_profiles：画像（目标 / 基础 / 偏好）",
    "user_memories：退款、笔记、画像变更事件",
    "读：load_memory / consult_rag / 售后工具",
    "写：memory_extractor / 退款节点 / save_memory",
    "检索：0.65 词重合 + 0.25 新鲜度 + 0.10 重要性",
]
cy = 1248
for line in memory_lines:
    parts.append(txt(360, cy, line, 13.5, "#2f5f50"))
    cy += 25

# ---------- 图例 ----------
legend = [
    (100, COLORS["blue"], "咨询链路"),
    (280, COLORS["red"], "退费链路"),
    (460, COLORS["yellow"], "判定节点"),
    (640, COLORS["purple"], "人工审批"),
    (820, COLORS["green"], "结束"),
]
for x, color, label in legend:
    parts.append(f'<rect x="{x}" y="1466" width="18" height="18" rx="4" fill="{color}"/>')
    parts.append(txt(x + 28, 1481, label, 13.5, "#4b5563", "normal", "start"))

parts.append(txt(100, 1516, "状态：thread_id → agent_state.db；interrupt 后可用 POST /api/audit/resume 跨进程恢复。", 13, "#5b6b7c", "normal", "start"))
parts.append(txt(100, 1542, "记忆：画像全量注入；情景记忆按问题检索 Top3；与文档冲突时以文档为准。", 13, "#5b6b7c", "normal", "start"))

parts.append("</svg>")

(BASE / "docs" / "流程图.svg").write_text("\n".join(parts), encoding="utf-8")
print("written:", BASE / "docs" / "流程图.svg")
