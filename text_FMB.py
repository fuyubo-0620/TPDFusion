"""
fmb_annotation_tool.py

FMB (Full-time Multi-modality Benchmark) 数据集标注工具
  — 含批量修改 & 筛选跳转功能 + GT 标签图可视化

适配 FMB 特点:
  ✅ 任务类型: fusion / segmentation
  ✅ 目标类别: car, person, bike, bump, guardrail, color_cone, car_stop, curve …
  ✅ 全时段条件: day_normal, overexposure, night, low_light, challenge …
  ✅ 分割专用字段: boundary (边界清晰度), seg_complexity, challenge
  ✏️ 批量修改:   选字段 + 旧值 + 新值 → 一键替换所有匹配图
  🔍 筛选跳转:   按字段值筛选，只在匹配图之间 ◀ ▶ 跳转
  📋 筛选列表:   列出所有匹配图，点击跳转
  🏷️ GT 可视化:  加载已着色的语义分割 GT 标签图直接显示

用法:
  python fmb_annotation_tool.py
  python fmb_annotation_tool.py --ir_dir FMB/ir --vis_dir FMB/vi --output_dir FMB/text --seg_dir FMB/seg_gt
"""

import os
import sys
import csv
import glob
import argparse
import tkinter as tk
from tkinter import ttk, messagebox
from collections import OrderedDict, Counter

try:
    from PIL import Image, ImageTk
except ImportError:
    print("❌ pip install Pillow")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
#  Prompt 模板
# ══════════════════════════════════════════════════════════════════

TEMPLATES = {
    "fusion": (
        "Infrared and visible image fusion{in_scene}, "
        "focusing on {target}. "
        "{thermal} thermal contrast.{condition_suffix}"
    ),
    "segmentation": (
        "Infrared and visible image fusion for semantic segmentation"
        "{in_scene}, focusing on {target}. "
        "{boundary} boundary clarity with {thermal} thermal contrast. "
        "{seg_complexity} segmentation complexity.{condition_suffix}"
    ),
    "segmentation_hard": (
        "Infrared and visible image fusion for semantic segmentation"
        "{in_scene}. "
        "Enhance {target} segmentation. {boundary} boundary clarity, "
        "{thermal} thermal contrast, {seg_complexity} complexity due to "
        "{challenge}.{condition_suffix}"
    ),
}


# ══════════════════════════════════════════════════════════════════
#  字段值映射  —  FMB 专用
# ══════════════════════════════════════════════════════════════════

TARGET_MAP = {
    "default":                  "",
    "person":                   "pedestrian",
    "car":                      "car",
    "bike":                     "bicycle",
    "bus":                      "bus",
    "truck":                    "truck",
    "motorcycle":               "motorcycle",
    "guardrail":                "guardrail",
    "color_cone":               "color cone",
    "bump":                     "speed bump",
    "curve":                    "road curve marking",
    "car_stop":                 "car stop barrier",
    "person_car":               "pedestrian and car",
    "person_bike":              "pedestrian and bicycle",
    "person_car_bike":          "pedestrian, car and bicycle",
    "car_bike":                 "car and bicycle",
    "car_bus":                  "car and bus",
    "car_truck":                "car and truck",
    "person_car_bus":           "pedestrian, car and bus",
    "person_car_truck":         "pedestrian, car and truck",
    "road_objects":             "road infrastructure (guardrail, bump, cone)",
    "vehicles":                 "vehicle",
    "person_vehicles":          "pedestrian and vehicle",
    "all":                      "all semantic categories",
}

THERMAL_MAP = {
    "default":  "",
    "strong":   "Strong",
    "moderate": "Moderate",
    "weak":     "Weak",
}

CONDITION_MAP = {
    "default":              "",
    # ── 白天 ──
    "day_normal":
        "Daytime with balanced visible and infrared imaging",
    "day_bright":
        "Bright daytime with clear visible details and moderate infrared contrast",
    "day_overcast":
        "Overcast daytime with soft lighting and mild infrared contrast",
    "day_shadow":
        "Daytime with strong shadows partially hiding targets in visible image",
    "day_overexposure":
        "Overexposed daytime with washed-out visible image and degraded details",
    "day_backlight":
        "Backlit daytime with silhouetted targets in visible image",
    "day_partial_overexposure":
        "Partially overexposed daytime with locally saturated visible regions",
    # ── 夜间 ──
    "night_lit":
        "Nighttime with street lighting and clear infrared targets",
    "night_low_light":
        "Dark nighttime with very limited visible information",
    "night_dark":
        "Nearly black visible image heavily relying on infrared",
    "night_glare":
        "Nighttime with artificial light glare in visible image",
    "night_mixed":
        "Nighttime with mixed lighting from streetlights and headlights",
    # ── 其他 ──
    "fog_light":
        "Light fog reducing visible contrast",
    "fog_heavy":
        "Heavy fog severely degrading visible image",
    "haze":
        "Light haze with slightly reduced visible contrast and clarity",
    "rain":
        "Rainy conditions with wet reflections and blurred visible image",
    "dusk_dawn":
        "Dusk or dawn with rapidly changing illumination",
}

SCENE_FEATURE_MAP = {
    "default":          "",
    "campus_road":
        "a campus road with pedestrians and cyclists",
    "campus_intersection":
        "a campus intersection with mixed traffic",
    "campus_parking":
        "a campus parking area with parked vehicles",
    "campus_sidewalk":
        "a campus sidewalk with dense pedestrians",
    "urban_road":
        "an urban road with buildings and trees on both sides",
    "wide_road":
        "a wide multi-lane road with open surroundings",
    "narrow_road":
        "a narrow road with closely spaced structures",
    "intersection":
        "an intersection with traffic lights and crosswalks",
    "residential":
        "a residential area with low-rise buildings",
    "commercial":
        "a commercial district with shops and signs",
    "tree_lined":
        "a tree-lined road with partial canopy cover",
    "open_area":
        "an open area with few obstructions",
    "bridge_overpass":
        "a bridge or overpass with distant view",
    "bus_stop":
        "a bus stop area with waiting pedestrians",
    "construction":
        "a road near a construction site with barriers and cones",
    "speed_bump_zone":
        "a road segment with speed bumps and warning signs",
    "curved_road":
        "a curved road with limited forward visibility",
    "straight_road":
        "a long straight road with clear sightlines",
}

# ── 分割专用字段 ──

BOUNDARY_MAP = {
    "default":  "",
    "clear":    "Clear",
    "moderate": "Moderate",
    "blurry":   "Blurry",
    "mixed":    "Mixed",
}

SEG_COMPLEXITY_MAP = {
    "default":  "",
    "low":      "Low",
    "moderate": "Moderate",
    "high":     "High",
}

CHALLENGE_MAP = {
    "default":              "",
    "none":                 "",
    "overexposure":         "overexposure washing out visible boundaries",
    "low_light":            "low light degrading visible texture",
    "low_contrast":         "very low thermal contrast between objects",
    "partial_occlusion":    "partial occlusion among objects",
    "heavy_occlusion":      "heavy occlusion among objects",
    "dense_crowd":          "dense clustering of pedestrians",
    "small_objects":        "small distant objects hard to segment",
    "similar_appearance":   "similar appearance between adjacent categories",
    "shadow":               "strong shadows creating false boundaries",
    "glare":                "light glare interfering with visible boundaries",
    "clutter":              "complex background clutter",
    "motion_blur":          "motion blur degrading object boundaries",
    "thermal_crossover":    "thermal crossover reducing infrared contrast",
}

# 字段名 → MAP 的对应关系
FIELD_TO_MAP = {
    "task":             None,
    "scene_feature":    SCENE_FEATURE_MAP,
    "target":           TARGET_MAP,
    "thermal":          THERMAL_MAP,
    "condition":        CONDITION_MAP,
    "boundary":         BOUNDARY_MAP,
    "seg_complexity":   SEG_COMPLEXITY_MAP,
    "challenge":        CHALLENGE_MAP,
}


# ══════════════════════════════════════════════════════════════════
#  Prompt 构建
# ══════════════════════════════════════════════════════════════════

def build_prompt(task="fusion", target="default", thermal="default",
                 condition="default", scene_feature="default",
                 boundary="default", seg_complexity="default",
                 challenge="default", **kwargs):

    raw = {
        "target":         TARGET_MAP.get(target, ""),
        "thermal":        THERMAL_MAP.get(thermal, ""),
        "condition":      CONDITION_MAP.get(condition, ""),
        "scene_feature":  SCENE_FEATURE_MAP.get(scene_feature, ""),
        "boundary":       BOUNDARY_MAP.get(boundary, ""),
        "seg_complexity":  SEG_COMPLEXITY_MAP.get(seg_complexity, ""),
        "challenge":      CHALLENGE_MAP.get(challenge, ""),
    }

    # ── 组装各片段 ──
    in_scene = f" in {raw['scene_feature']}" if raw["scene_feature"] else ""
    target_text = raw["target"] if raw["target"] else "salient targets"
    thermal_text = raw["thermal"] if raw["thermal"] else "Moderate"
    condition_suffix = f" {raw['condition']}." if raw["condition"] else ""
    boundary_text = raw["boundary"] if raw["boundary"] else "Moderate"
    seg_complexity_text = raw["seg_complexity"] if raw["seg_complexity"] else "Moderate"
    challenge_text = raw["challenge"]

    fields = {
        "in_scene":         in_scene,
        "target":           target_text,
        "thermal":          thermal_text,
        "condition_suffix": condition_suffix,
        "boundary":         boundary_text,
        "seg_complexity":   seg_complexity_text,
        "challenge":        challenge_text,
    }

    if task == "segmentation":
        if challenge and challenge not in ("default", "none", "") and challenge_text:
            tmpl = TEMPLATES["segmentation_hard"]
        else:
            tmpl = TEMPLATES["segmentation"]
    else:
        tmpl = TEMPLATES["fusion"]

    prompt = tmpl.format(**fields)
    prompt = " ".join(prompt.split())
    prompt = prompt.replace("due to .", ".")
    prompt = prompt.replace("due to ,", ",")
    prompt = prompt.replace("..", ".")
    return prompt


def guess_condition(vis_path, ir_path, name):
    """根据文件名 / 路径推断 FMB 的拍摄条件"""
    s = (vis_path + ir_path + name).lower()

    if "overexposure" in s or "over" in s:
        return "day_overexposure"
    if any(k in s for k in ["night", "dark"]):
        return "night_lit"
    if "low_light" in s or "lowlight" in s:
        return "night_low_light"
    if any(k in s for k in ["fog", "haze"]):
        return "fog_light"
    if "rain" in s:
        return "rain"
    if "dusk" in s or "dawn" in s:
        return "dusk_dawn"
    if "backlight" in s:
        return "day_backlight"
    if "shadow" in s:
        return "day_shadow"
    return None


# ══════════════════════════════════════════════════════════════════
#  图像对收集
# ══════════════════════════════════════════════════════════════════

def collect_pairs(ir_dir, vis_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]

    def _get(folder):
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        return {os.path.splitext(os.path.basename(p))[0]: p
                for p in sorted(set(files))}

    ir = _get(ir_dir)
    vis = _get(vis_dir)
    common = sorted(set(ir) & set(vis))
    if not common:
        messagebox.showerror("错误",
            f"未找到匹配图像对!\nIR: {ir_dir} ({len(ir)}张)\n"
            f"VIS: {vis_dir} ({len(vis)}张)")
        sys.exit(1)
    print(f"  找到 {len(common)} 对图像")
    return [(n, vis[n], ir[n]) for n in common]


def collect_seg_labels(seg_dir):
    """收集 GT 标签图路径，返回 {name: path}"""
    if not seg_dir or not os.path.isdir(seg_dir):
        return {}
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(seg_dir, ext)))
        files.extend(glob.glob(os.path.join(seg_dir, ext.upper())))
    result = {os.path.splitext(os.path.basename(p))[0]: p
              for p in sorted(set(files))}
    print(f"  找到 {len(result)} 张 GT 标签图")
    return result


# ══════════════════════════════════════════════════════════════════
#  标注字段定义  —  FMB 版
# ══════════════════════════════════════════════════════════════════

FIELDS = OrderedDict([
    ("task", {
        "label": "📋 任务类型 Task",
        "options": ["fusion", "segmentation"],
        "default": "segmentation",
        "group": "common",
    }),
    ("scene_feature", {
        "label": "🏙️ 场景特点 Scene Feature",
        "options": list(SCENE_FEATURE_MAP.keys()),
        "default": "default",
        "group": "common",
    }),
    ("target", {
        "label": "🎯 目标类别 Target",
        "options": list(TARGET_MAP.keys()),
        "default": "default",
        "group": "common",
    }),
    ("thermal", {
        "label": "🌡️ 热对比度 Thermal",
        "options": list(THERMAL_MAP.keys()),
        "default": "default",
        "group": "common",
    }),
    ("condition", {
        "label": "🌤️ 环境条件 Condition",
        "options": list(CONDITION_MAP.keys()),
        "default": "default",
        "group": "common",
    }),
    ("boundary", {
        "label": "🔲 边界清晰度 Boundary",
        "options": list(BOUNDARY_MAP.keys()),
        "default": "default",
        "group": "segmentation",
    }),
    ("seg_complexity", {
        "label": "🧩 分割复杂度 Complexity",
        "options": list(SEG_COMPLEXITY_MAP.keys()),
        "default": "default",
        "group": "segmentation",
    }),
    ("challenge", {
        "label": "⚠️ 困难因素 Challenge",
        "options": list(CHALLENGE_MAP.keys()),
        "default": "default",
        "group": "segmentation",
    }),
])


# ══════════════════════════════════════════════════════════════════
#  主界面
# ══════════════════════════════════════════════════════════════════

class App:

    def __init__(self, root, ir_dir, vis_dir, output_dir, csv_path, seg_dir=None):
        self.root = root
        self.root.title("FMB 标注工具  |  含批量修改 & 筛选跳转 & GT可视化")
        self.root.configure(bg="#2b2b2b")

        self.output_dir = output_dir
        self.csv_path = csv_path
        os.makedirs(output_dir, exist_ok=True)

        self.pairs = collect_pairs(ir_dir, vis_dir)
        self.seg_labels = collect_seg_labels(seg_dir) if seg_dir else {}
        self.has_seg = bool(self.seg_labels)

        self.name_to_idx = {n: i for i, (n, _, _) in enumerate(self.pairs)}
        self.idx = 0
        self.annotations = {}
        self._load_csv()

        # 筛选状态
        self.filter_field = None
        self.filter_value = None
        self.filter_indices = []
        self.filter_pos = -1

        self.field_vars = {}
        # 有GT时三列，图缩小一点放得下
        self.img_size = (380, 285) if self.has_seg else (480, 360)

        self._build_ui()
        self._goto_first_unannotated()
        self._show()

        root.bind("<Left>",      lambda e: self._prev())
        root.bind("<Right>",     lambda e: self._next())
        root.bind("<Return>",    lambda e: self._save_next())
        root.bind("<Control-s>", lambda e: self._export_csv())
        root.bind("<Escape>",    lambda e: self._quit())
        root.protocol("WM_DELETE_WINDOW", self._quit)

    # ──────────────────────────────────────────
    #  CSV 读取
    # ──────────────────────────────────────────

    def _load_csv(self):
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = row.get("name", "").strip()
                    if name:
                        self.annotations[name] = {
                            k: v.strip() for k, v in row.items() if k != "name"
                        }
            print(f"  ✅ 加载 {len(self.annotations)} 条已有标注")
        except Exception as e:
            print(f"  ⚠️ 加载CSV失败: {e}")

    def _goto_first_unannotated(self):
        for i, (name, _, _) in enumerate(self.pairs):
            if name not in self.annotations:
                self.idx = i
                return

    # ──────────────────────────────────────────
    #  构建界面
    # ──────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",    background="#2b2b2b")
        style.configure("TLabel",    background="#2b2b2b", foreground="#e0e0e0",
                         font=("Microsoft YaHei", 10))
        style.configure("T.TLabel",  background="#2b2b2b", foreground="#00ccff",
                         font=("Microsoft YaHei", 12, "bold"))
        style.configure("S.TLabel",  background="#1e1e1e", foreground="#aaa",
                         font=("Consolas", 9))
        style.configure("D.TLabel",  background="#2b2b2b", foreground="#ffaa00",
                         font=("Microsoft YaHei", 10, "bold"))
        style.configure("H.TLabel",  background="#2b2b2b", foreground="#888",
                         font=("Microsoft YaHei", 9))
        style.configure("F.TLabel",  background="#2b2b2b", foreground="#ff6666",
                         font=("Microsoft YaHei", 10, "bold"))

        # ═══ 顶部导航 ═══
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=(10, 5))

        self.progress_var = tk.StringVar()
        ttk.Label(top, textvariable=self.progress_var,
                  style="T.TLabel").pack(side="left")

        nav = ttk.Frame(top)
        nav.pack(side="right")
        ttk.Button(nav, text="◀ 上一张",
                   command=self._prev).pack(side="left", padx=2)
        ttk.Button(nav, text="下一张 ▶",
                   command=self._next).pack(side="left", padx=2)
        self.jump_var = tk.StringVar()
        ttk.Entry(nav, textvariable=self.jump_var, width=8,
                  font=("Consolas", 10)).pack(side="left", padx=2)
        ttk.Button(nav, text="跳转",
                   command=self._jump).pack(side="left", padx=2)

        # ═══ 图像 ═══
        img_frame = ttk.Frame(self.root)
        img_frame.pack(padx=10, pady=5)

        panels = [
            ("left", "📷 Visible (可见光)", "vis_canvas"),
            ("left", "🌡 Infrared (红外)",  "ir_canvas"),
        ]
        if self.has_seg:
            panels.append(("left", "🏷️ Seg GT (分割真值)", "seg_canvas"))

        for side, title, attr in panels:
            c = ttk.Frame(img_frame)
            c.pack(side=side, padx=5)
            ttk.Label(c, text=title, style="T.TLabel").pack()
            canvas = tk.Canvas(c, width=self.img_size[0],
                               height=self.img_size[1],
                               bg="#1a1a1a", highlightthickness=1,
                               highlightbackground="#444")
            canvas.pack()
            setattr(self, attr, canvas)

        if not self.has_seg:
            self.seg_canvas = None

        self.info_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.info_var,
                  style="S.TLabel").pack(fill="x", padx=10)

        # ═══ 筛选栏 ═══
        filter_frame = ttk.Frame(self.root)
        filter_frame.pack(fill="x", padx=10, pady=(5, 0))

        ttk.Label(filter_frame, text="🔍 筛选",
                  style="F.TLabel").pack(side="left")

        ttk.Label(filter_frame, text="  字段:").pack(side="left")
        self.filter_field_var = tk.StringVar(value="condition")
        filter_field_options = [f for f in FIELDS.keys()]
        ttk.Combobox(filter_frame, textvariable=self.filter_field_var,
                      values=filter_field_options, state="readonly",
                      width=14, font=("Consolas", 10)).pack(side="left", padx=3)
        self.filter_field_var.trace_add("write",
            lambda *a: self._update_filter_value_options())

        ttk.Label(filter_frame, text="  值:").pack(side="left")
        self.filter_value_var = tk.StringVar()
        self.filter_value_combo = ttk.Combobox(
            filter_frame, textvariable=self.filter_value_var,
            state="readonly", width=20, font=("Consolas", 10))
        self.filter_value_combo.pack(side="left", padx=3)

        ttk.Button(filter_frame, text="🔍 筛选跳转",
                   command=self._apply_filter).pack(side="left", padx=3)
        ttk.Button(filter_frame, text="◀ 上一个",
                   command=self._filter_prev).pack(side="left", padx=2)
        ttk.Button(filter_frame, text="下一个 ▶",
                   command=self._filter_next).pack(side="left", padx=2)
        ttk.Button(filter_frame, text="📋 列表",
                   command=self._filter_list).pack(side="left", padx=2)
        ttk.Button(filter_frame, text="✖ 清除筛选",
                   command=self._clear_filter).pack(side="left", padx=2)

        self.filter_status_var = tk.StringVar()
        ttk.Label(filter_frame, textvariable=self.filter_status_var,
                  style="F.TLabel").pack(side="left", padx=8)

        self._update_filter_value_options()

        # ═══ 标注区域 ═══
        fo = ttk.Frame(self.root)
        fo.pack(fill="x", padx=10, pady=5)

        # ── 任务选择 ──
        task_row = ttk.Frame(fo)
        task_row.pack(fill="x", pady=(0, 5))

        ttk.Label(task_row, text="📋 任务类型 Task",
                  style="D.TLabel", width=20, anchor="w").pack(side="left")
        self.field_vars["task"] = tk.StringVar(value="segmentation")
        ttk.Combobox(task_row, textvariable=self.field_vars["task"],
                      values=["fusion", "segmentation"], state="readonly",
                      width=22, font=("Consolas", 11)).pack(side="left", padx=5)
        self.field_vars["task"].trace_add("write",
                                           lambda *a: self._on_task_change())
        self.task_desc_var = tk.StringVar()
        ttk.Label(task_row, textvariable=self.task_desc_var,
                  style="S.TLabel").pack(side="left", padx=10)

        ttk.Separator(fo, orient="horizontal").pack(fill="x", pady=3)

        ttk.Label(fo, text="━━ 公共字段 (Common) ━━",
                  style="T.TLabel").pack(anchor="w", pady=(3, 2))

        # ── 场景特点 ──
        sf_row = ttk.Frame(fo)
        sf_row.pack(fill="x", pady=2)

        ttk.Label(sf_row, text="🏙️ 场景特点 Scene Feature",
                  width=28, anchor="w").pack(side="left")
        self.field_vars["scene_feature"] = tk.StringVar(value="default")
        self.field_vars["scene_feature"].trace_add("write",
            lambda *a: self._update_preview())
        ttk.Combobox(sf_row,
                      textvariable=self.field_vars["scene_feature"],
                      values=list(SCENE_FEATURE_MAP.keys()),
                      state="readonly", width=28,
                      font=("Consolas", 10)).pack(side="left", padx=5)

        self.sf_hint = tk.StringVar()
        ttk.Label(sf_row, textvariable=self.sf_hint,
                  style="H.TLabel").pack(side="left", padx=5)
        self.field_vars["scene_feature"].trace_add("write",
            lambda *a: self.sf_hint.set(
                ("→ " + SCENE_FEATURE_MAP.get(
                    self.field_vars["scene_feature"].get(), "")[:55])
                if SCENE_FEATURE_MAP.get(
                    self.field_vars["scene_feature"].get(), "")
                else "→ (不描述)"))

        # ── target / thermal ──
        row2 = ttk.Frame(fo)
        row2.pack(fill="x", pady=2)

        for fname in ["target", "thermal"]:
            fdef = FIELDS[fname]
            f = ttk.Frame(row2)
            f.pack(side="left", padx=8)
            ttk.Label(f, text=fdef["label"]).pack(anchor="w")
            var = tk.StringVar(value=fdef["default"])
            var.trace_add("write", lambda *a: self._update_preview())
            ttk.Combobox(f, textvariable=var, values=fdef["options"],
                          state="readonly", width=22,
                          font=("Consolas", 10)).pack()
            self.field_vars[fname] = var

        # ── condition ──
        cond_row = ttk.Frame(fo)
        cond_row.pack(fill="x", pady=2)

        ttk.Label(cond_row, text="🌤️ 环境条件 Condition",
                  width=28, anchor="w").pack(side="left")
        self.field_vars["condition"] = tk.StringVar(value="default")
        self.field_vars["condition"].trace_add("write",
            lambda *a: self._update_preview())
        ttk.Combobox(cond_row,
                      textvariable=self.field_vars["condition"],
                      values=list(CONDITION_MAP.keys()),
                      state="readonly", width=28,
                      font=("Consolas", 10)).pack(side="left", padx=5)

        self.cond_hint = tk.StringVar()
        cond_hint_frame = ttk.Frame(fo)
        cond_hint_frame.pack(fill="x", pady=(2, 5))
        self.cond_hint_label = tk.Label(
            cond_hint_frame, textvariable=self.cond_hint,
            bg="#1a2a1a", fg="#66cc66", font=("Consolas", 9),
            anchor="w", justify="left", wraplength=700, padx=10, pady=4)
        self.cond_hint_label.pack(fill="x")
        self.field_vars["condition"].trace_add("write",
            lambda *a: self.cond_hint.set(
                ("🌤️ → " + CONDITION_MAP.get(
                    self.field_vars["condition"].get(), ""))
                if CONDITION_MAP.get(
                    self.field_vars["condition"].get(), "")
                else "🌤️ → (不描述)"))

        # ── 分割专用 ──
        self.seg_title = ttk.Label(fo,
                                    text="━━ 分割专用字段 (Segmentation Only) ━━",
                                    style="D.TLabel")
        self.seg_title.pack(anchor="w", pady=(8, 2))

        self.seg_row = ttk.Frame(fo)
        self.seg_row.pack(fill="x", pady=2)

        for fname in ["boundary", "seg_complexity", "challenge"]:
            fdef = FIELDS[fname]
            f = ttk.Frame(self.seg_row)
            f.pack(side="left", padx=8)
            ttk.Label(f, text=fdef["label"]).pack(anchor="w")
            var = tk.StringVar(value=fdef["default"])
            var.trace_add("write", lambda *a: self._update_preview())
            ttk.Combobox(f, textvariable=var, values=fdef["options"],
                          state="readonly", width=22,
                          font=("Consolas", 10)).pack()
            self.field_vars[fname] = var

        # ═══ Prompt 预览 ═══
        pf = ttk.Frame(self.root)
        pf.pack(fill="x", padx=10, pady=5)

        ttk.Label(pf, text="━━━ Prompt 预览 ━━━",
                  style="T.TLabel").pack(anchor="w")

        self.prompt_text = tk.Text(pf, height=3, wrap="word",
                                    bg="#1a1a2e", fg="#00ff88",
                                    font=("Consolas", 10), relief="flat",
                                    padx=10, pady=8)
        self.prompt_text.pack(fill="x")
        self.prompt_text.configure(state="disabled")

        self.token_var = tk.StringVar()
        ttk.Label(pf, textvariable=self.token_var,
                  style="S.TLabel").pack(anchor="e")

        # ═══ 按钮 ═══
        btn = ttk.Frame(self.root)
        btn.pack(fill="x", padx=10, pady=(5, 3))

        for text, cmd in [
            ("💾 保存&下一张 (Enter)", self._save_next),
            ("💾 仅保存",              self._save),
            ("📋 复制",                self._copy),
            ("📁 CSV",                 self._export_csv),
            ("📁 全部.txt",            self._export_txt),
        ]:
            ttk.Button(btn, text=text, command=cmd).pack(side="left", padx=3)

        self.status_var = tk.StringVar()
        ttk.Label(btn, textvariable=self.status_var,
                  style="S.TLabel").pack(side="right")

        # ── 批量 + 统计 + 重置 + 批量修改 ──
        batch_frame = ttk.Frame(self.root)
        batch_frame.pack(fill="x", padx=10, pady=(0, 5))

        ttk.Button(batch_frame, text="⚡ 批量标注剩余",
                   command=self._batch).pack(side="left", padx=3)
        ttk.Button(batch_frame, text="✏️ 批量修改",
                   command=self._batch_modify).pack(side="left", padx=3)
        ttk.Button(batch_frame, text="📊 统计",
                   command=self._stats).pack(side="left", padx=3)
        ttk.Button(batch_frame, text="🔄 重置default",
                   command=self._reset_all).pack(side="left", padx=3)

        ttk.Label(self.root,
                  text="  ← → 翻页 | Enter 保存下一张 | Ctrl+S CSV | Esc 退出",
                  style="S.TLabel").pack(fill="x", side="bottom", ipady=3)

        self._on_task_change()

    # ──────────────────────────────────────────
    #  任务切换
    # ──────────────────────────────────────────

    def _on_task_change(self):
        task = self.field_vars["task"].get()
        if task == "segmentation":
            self.seg_title.pack(anchor="w", pady=(8, 2))
            self.seg_row.pack(fill="x", pady=2)
            self.task_desc_var.set("🧩 分割模式: +boundary/complexity/challenge")
        else:
            self.seg_title.pack_forget()
            self.seg_row.pack_forget()
            self.task_desc_var.set("🔗 融合模式: 仅场景+目标+环境")
        self._update_preview()

    def _reset_all(self):
        for fname, fdef in FIELDS.items():
            self.field_vars[fname].set(fdef["default"])
        self.status_var.set("🔄 已重置")

    def _update_preview(self):
        kwargs = {k: v.get() for k, v in self.field_vars.items()}
        prompt = build_prompt(**kwargs)
        self.prompt_text.configure(state="normal")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", prompt)
        self.prompt_text.configure(state="disabled")
        self.token_var.set(f"  ~{len(prompt.split())} words | {len(prompt)} chars")

    # ══════════════════════════════════════════════════════════════
    #  ★ 筛选功能
    # ══════════════════════════════════════════════════════════════

    def _update_filter_value_options(self):
        field = self.filter_field_var.get()
        used_values = set()
        for ann in self.annotations.values():
            v = ann.get(field, "")
            if v:
                used_values.add(v)
        opts = sorted(used_values)
        if not opts:
            opts = FIELDS.get(field, {}).get("options", [])
        self.filter_value_combo["values"] = opts
        if opts:
            self.filter_value_var.set(opts[0])

    def _apply_filter(self):
        field = self.filter_field_var.get()
        value = self.filter_value_var.get()

        if not value:
            messagebox.showwarning("筛选", "请选择筛选值")
            return

        self.filter_field = field
        self.filter_value = value
        self.filter_indices = []

        for i, (name, _, _) in enumerate(self.pairs):
            if name in self.annotations:
                if self.annotations[name].get(field, "") == value:
                    self.filter_indices.append(i)

        if not self.filter_indices:
            self.filter_status_var.set(f"⚠️ 未找到 {field}={value} 的图像")
            self.filter_field = None
            return

        self.filter_pos = 0
        self.idx = self.filter_indices[0]
        n = len(self.filter_indices)
        self.filter_status_var.set(
            f"🔍 筛选: {field}={value}  共 {n} 张  [1/{n}]")
        self._show()

    def _filter_prev(self):
        if not self.filter_indices:
            self.filter_status_var.set("⚠️ 请先筛选")
            return
        if self.filter_pos > 0:
            self.filter_pos -= 1
        else:
            self.filter_pos = len(self.filter_indices) - 1
        self.idx = self.filter_indices[self.filter_pos]
        n = len(self.filter_indices)
        self.filter_status_var.set(
            f"🔍 {self.filter_field}={self.filter_value}  "
            f"[{self.filter_pos + 1}/{n}]")
        self._show()

    def _filter_next(self):
        if not self.filter_indices:
            self.filter_status_var.set("⚠️ 请先筛选")
            return
        if self.filter_pos < len(self.filter_indices) - 1:
            self.filter_pos += 1
        else:
            self.filter_pos = 0
        self.idx = self.filter_indices[self.filter_pos]
        n = len(self.filter_indices)
        self.filter_status_var.set(
            f"🔍 {self.filter_field}={self.filter_value}  "
            f"[{self.filter_pos + 1}/{n}]")
        self._show()

    def _filter_list(self):
        if not self.filter_indices:
            self.filter_status_var.set("⚠️ 请先筛选")
            return

        win = tk.Toplevel(self.root)
        win.title(f"筛选结果: {self.filter_field}={self.filter_value}  "
                  f"({len(self.filter_indices)} 张)")
        win.geometry("400x500")
        win.configure(bg="#1e1e1e")

        ttk.Label(win, text=f"🔍 {self.filter_field} = {self.filter_value}",
                  style="T.TLabel").pack(pady=5)

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(frame, bg="#1e1e1e", fg="#00ff88",
                              font=("Consolas", 11), selectmode="single",
                              yscrollcommand=scrollbar.set)
        listbox.pack(fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        for idx in self.filter_indices:
            name = self.pairs[idx][0]
            listbox.insert("end", f"  [{idx + 1}]  {name}")

        def on_select(event):
            sel = listbox.curselection()
            if sel:
                pos = sel[0]
                self.filter_pos = pos
                self.idx = self.filter_indices[pos]
                n = len(self.filter_indices)
                self.filter_status_var.set(
                    f"🔍 {self.filter_field}={self.filter_value}  "
                    f"[{pos + 1}/{n}]")
                self._show()

        listbox.bind("<<ListboxSelect>>", on_select)

    def _clear_filter(self):
        self.filter_field = None
        self.filter_value = None
        self.filter_indices = []
        self.filter_pos = -1
        self.filter_status_var.set("")

    # ══════════════════════════════════════════════════════════════
    #  ★ 批量修改
    # ══════════════════════════════════════════════════════════════

    def _batch_modify(self):
        if not self.annotations:
            messagebox.showinfo("批量修改", "暂无标注数据")
            return

        win = tk.Toplevel(self.root)
        win.title("✏️ 批量修改标注")
        win.geometry("550x480")
        win.configure(bg="#2b2b2b")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="✏️ 批量修改标注",
                  style="T.TLabel").pack(pady=(15, 10))

        ttk.Label(win, text="将符合条件的图像的某个字段值批量替换",
                  style="S.TLabel").pack()

        main_f = ttk.Frame(win)
        main_f.pack(fill="x", padx=20, pady=15)

        # 选择字段
        ttk.Label(main_f, text="① 选择要修改的字段:").pack(anchor="w", pady=(5, 2))
        mod_field_var = tk.StringVar(value="condition")
        mod_field_options = [f for f in FIELDS.keys()]
        mod_field_combo = ttk.Combobox(main_f, textvariable=mod_field_var,
                                        values=mod_field_options, state="readonly",
                                        width=25, font=("Consolas", 11))
        mod_field_combo.pack(anchor="w", padx=10)

        # 旧值
        ttk.Label(main_f, text="② 当前的 (错误的) 值:").pack(anchor="w", pady=(12, 2))
        old_val_var = tk.StringVar()
        old_val_combo = ttk.Combobox(main_f, textvariable=old_val_var,
                                      state="readonly", width=25,
                                      font=("Consolas", 11))
        old_val_combo.pack(anchor="w", padx=10)

        # 新值
        ttk.Label(main_f, text="③ 替换为 (正确的) 值:").pack(anchor="w", pady=(12, 2))
        new_val_var = tk.StringVar()
        new_val_combo = ttk.Combobox(main_f, textvariable=new_val_var,
                                      state="readonly", width=25,
                                      font=("Consolas", 11))
        new_val_combo.pack(anchor="w", padx=10)

        # 预览
        preview_var = tk.StringVar(value="选择字段和值后显示预览...")
        preview_label = tk.Label(win, textvariable=preview_var,
                                  bg="#1a2a1a", fg="#ffcc00",
                                  font=("Consolas", 10),
                                  anchor="w", justify="left",
                                  wraplength=500, padx=15, pady=10)
        preview_label.pack(fill="x", padx=20, pady=10)

        def update_old_options(*args):
            field = mod_field_var.get()
            used = set()
            for ann in self.annotations.values():
                v = ann.get(field, "")
                if v:
                    used.add(v)
            opts = sorted(used)
            old_val_combo["values"] = opts
            if opts:
                old_val_var.set(opts[0])

            all_opts = FIELDS.get(field, {}).get("options", [])
            new_val_combo["values"] = all_opts
            if all_opts:
                new_val_var.set(all_opts[0])

            update_preview()

        def update_preview(*args):
            field = mod_field_var.get()
            old_v = old_val_var.get()
            new_v = new_val_var.get()

            if not old_v:
                preview_var.set("请选择当前值")
                return

            count = sum(1 for ann in self.annotations.values()
                        if ann.get(field, "") == old_v)

            preview_var.set(
                f"将修改 {count} 张图像:\n"
                f"  {field}: {old_v}  →  {new_v}"
            )

        mod_field_var.trace_add("write", update_old_options)
        old_val_var.trace_add("write", update_preview)
        new_val_var.trace_add("write", update_preview)
        update_old_options()

        # 按钮
        btn_f = ttk.Frame(win)
        btn_f.pack(pady=10)

        def do_modify():
            field = mod_field_var.get()
            old_v = old_val_var.get()
            new_v = new_val_var.get()

            if old_v == new_v:
                messagebox.showwarning("无变化", "旧值和新值相同")
                return

            names = [name for name, ann in self.annotations.items()
                     if ann.get(field, "") == old_v]

            if not names:
                messagebox.showinfo("无匹配", f"没有 {field}={old_v} 的图像")
                return

            if not messagebox.askyesno("确认批量修改",
                f"确认将 {len(names)} 张图的\n"
                f"  {field}: {old_v}\n"
                f"改为:\n"
                f"  {field}: {new_v}\n\n"
                f"此操作不可撤销！"):
                return

            for name in names:
                self.annotations[name][field] = new_v

            self._export_csv(silent=True)
            for name in names:
                p = build_prompt(**{k: v for k, v in
                    self.annotations[name].items() if k in FIELDS})
                txt_path = os.path.join(self.output_dir, f"{name}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(p)

            messagebox.showinfo("修改完成",
                f"已修改 {len(names)} 张图像的 {field}:\n"
                f"  {old_v} → {new_v}\n\n"
                f"CSV 和 .txt 已更新")

            update_old_options()
            self._show()

        ttk.Button(btn_f, text="✅ 执行批量修改", command=do_modify).pack(side="left", padx=5)
        ttk.Button(btn_f, text="❌ 取消", command=win.destroy).pack(side="left", padx=5)

    # ──────────────────────────────────────────
    #  图像显示
    # ──────────────────────────────────────────

    def _show(self):
        if not self.pairs:
            return

        name, vis_path, ir_path = self.pairs[self.idx]

        try:
            vis_img = Image.open(vis_path)
            ir_img = Image.open(ir_path)
        except Exception as e:
            messagebox.showerror("图像加载失败", str(e))
            return

        self._vis_photo = ImageTk.PhotoImage(self._fit(vis_img))
        self._ir_photo = ImageTk.PhotoImage(self._fit(ir_img))

        cx, cy = self.img_size[0] // 2, self.img_size[1] // 2

        self.vis_canvas.delete("all")
        self.vis_canvas.create_image(cx, cy, anchor="center",
                                      image=self._vis_photo)
        self.ir_canvas.delete("all")
        self.ir_canvas.create_image(cx, cy, anchor="center",
                                     image=self._ir_photo)

        # ── 显示 GT 标签图（已着色，直接加载） ──
        if self.has_seg and self.seg_canvas:
            seg_path = self.seg_labels.get(name)
            if seg_path:
                try:
                    seg_img = Image.open(seg_path).convert("RGB")
                    self._seg_photo = ImageTk.PhotoImage(self._fit(seg_img))
                    self.seg_canvas.delete("all")
                    self.seg_canvas.create_image(cx, cy, anchor="center",
                                                  image=self._seg_photo)
                except Exception as e:
                    self.seg_canvas.delete("all")
                    self.seg_canvas.create_text(cx, cy, text=f"加载失败\n{e}",
                                                fill="#ff4444",
                                                font=("Consolas", 10))
            else:
                self.seg_canvas.delete("all")
                self.seg_canvas.create_text(cx, cy, text="无GT",
                                            fill="#666",
                                            font=("Consolas", 11))

        n_total = len(self.pairs)
        n_done = len(self.annotations)
        pct = n_done / n_total * 100 if n_total > 0 else 0
        self.progress_var.set(
            f"[{self.idx + 1}/{n_total}] {name}  "
            f"(已标 {n_done}/{n_total} {pct:.0f}%)")

        self.info_var.set(
            f"  {name} | "
            f"VIS: {vis_img.size[0]}×{vis_img.size[1]}  "
            f"IR: {ir_img.size[0]}×{ir_img.size[1]}")

        if name in self.annotations:
            for k, v in self.field_vars.items():
                val = self.annotations[name].get(k, "")
                if val:
                    v.set(val)
            self.status_var.set("✅ 已标注 (可修改)")
        else:
            g = guess_condition(vis_path, ir_path, name)
            if g:
                self.field_vars["condition"].set(g)
            self.status_var.set("⬜ 未标注")

        self._update_preview()
        self.jump_var.set(str(self.idx + 1))

    def _fit(self, img):
        tw, th = self.img_size
        r = min(tw / img.width, th / img.height)
        return img.resize((int(img.width * r), int(img.height * r)),
                          Image.LANCZOS)

    # ──────────────────────────────────────────
    #  保存 / 导出
    # ──────────────────────────────────────────

    def _save(self):
        name = self.pairs[self.idx][0]
        ann = {k: v.get() for k, v in self.field_vars.items()}
        self.annotations[name] = ann

        txt_path = os.path.join(self.output_dir, f"{name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(build_prompt(**ann))

        self._export_csv(silent=True)
        self.status_var.set(f"✅ 已保存 {name}")
        self._show()

    def _save_next(self):
        self._save()
        if self.filter_indices:
            self._filter_next()
        else:
            self._next()

    def _export_csv(self, silent=False):
        cols = ["name"] + list(FIELDS.keys()) + ["prompt"]
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for name, _, _ in self.pairs:
                    if name in self.annotations:
                        row = {"name": name}
                        row.update(self.annotations[name])
                        row["prompt"] = build_prompt(
                            **{k: v for k, v in self.annotations[name].items()
                               if k in FIELDS})
                        w.writerow(row)
            if not silent:
                messagebox.showinfo("导出成功",
                    f"CSV: {self.csv_path}\n共 {len(self.annotations)} 条")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _export_txt(self):
        count = 0
        for name, _, _ in self.pairs:
            if name in self.annotations:
                p = build_prompt(**{k: v for k, v in
                    self.annotations[name].items() if k in FIELDS})
                with open(os.path.join(self.output_dir, f"{name}.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(p)
                count += 1
        messagebox.showinfo("导出完成", f"已导出 {count} 个 .txt")

    # ──────────────────────────────────────────
    #  批量标注
    # ──────────────────────────────────────────

    def _batch(self):
        un = [n for n, _, _ in self.pairs if n not in self.annotations]
        if not un:
            messagebox.showinfo("完成", "全部已标注!")
            return

        cur = {k: v.get() for k, v in self.field_vars.items()}
        preview = build_prompt(**cur)

        msg = (
            f"将当前设置应用到 {len(un)} 张未标注图像:\n\n"
            f"预览:\n{preview}\n\n"
            f"(condition 根据文件名自动推断)\n继续?"
        )

        if not messagebox.askyesno("批量标注确认", msg):
            return

        for name in un:
            a = cur.copy()
            for n, vp, ip in self.pairs:
                if n == name:
                    g = guess_condition(vp, ip, name)
                    if g:
                        a["condition"] = g
                    break
            self.annotations[name] = a

        self._export_csv(silent=True)
        self._export_txt()
        messagebox.showinfo("批量完成",
            f"已标注 {len(un)} 张\n总计: {len(self.annotations)}/{len(self.pairs)}")
        self._show()

    # ──────────────────────────────────────────
    #  复制 / 统计
    # ──────────────────────────────────────────

    def _copy(self):
        kwargs = {k: v.get() for k, v in self.field_vars.items()}
        self.root.clipboard_clear()
        self.root.clipboard_append(build_prompt(**kwargs))
        self.status_var.set("📋 已复制")

    def _stats(self):
        if not self.annotations:
            messagebox.showinfo("统计", "暂无")
            return

        n = len(self.annotations)
        n_total = len(self.pairs)

        lines = [
            f"═══ FMB 标注统计 ═══\n",
            f"总图像对:  {n_total}",
            f"已标注:    {n}  ({n / n_total * 100:.1f}%)",
            f"未标注:    {n_total - n}\n",
        ]

        for field in FIELDS:
            c = Counter(a.get(field, "?") for a in self.annotations.values())
            lines.append(f"── {field} ──")
            for val, cnt in c.most_common():
                bar = "█" * max(1, int(cnt / n * 30))
                lines.append(f"  {val:<28s} {cnt:>4d}  {bar}")
            lines.append("")

        lines.append(f"── default 占比 (未描述) ──")
        for field in FIELDS:
            if field == "task":
                continue
            dc = sum(1 for a in self.annotations.values()
                     if a.get(field, "") == "default")
            pct = dc / n * 100 if n > 0 else 0
            lines.append(f"  {field:<28s} {dc:>4d} ({pct:.0f}%)")

        win = tk.Toplevel(self.root)
        win.title("FMB 统计")
        win.geometry("620x750")
        win.configure(bg="#1e1e1e")
        t = tk.Text(win, bg="#1e1e1e", fg="#00ff88",
                    font=("Consolas", 10), padx=15, pady=15)
        t.pack(fill="both", expand=True)
        t.insert("1.0", "\n".join(lines))
        t.configure(state="disabled")

    # ──────────────────────────────────────────
    #  导航
    # ──────────────────────────────────────────

    def _prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._show()

    def _next(self):
        if self.idx < len(self.pairs) - 1:
            self.idx += 1
            self._show()

    def _jump(self):
        try:
            i = int(self.jump_var.get()) - 1
            if 0 <= i < len(self.pairs):
                self.idx = i
                self._show()
            else:
                messagebox.showwarning("范围", f"请输入 1~{len(self.pairs)}")
        except ValueError:
            target = self.jump_var.get().strip()
            for i, (n, _, _) in enumerate(self.pairs):
                if target in n:
                    self.idx = i
                    self._show()
                    return
            messagebox.showwarning("未找到", f"'{target}' 不存在")

    def _quit(self):
        if self.annotations:
            self._export_csv(silent=True)
            print(f"  💾 退出时自动保存 {len(self.annotations)} 条")
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════
#  启动
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FMB 标注工具 (含批量修改 & 筛选跳转 & GT可视化)"
    )
    parser.add_argument("--ir_dir", type=str,
                        default=r"D:\paper\datasets\MSRS-main\test\ir")
    parser.add_argument("--vis_dir", type=str,
                        default=r"D:\paper\datasets\MSRS-main\test\vi")
    parser.add_argument("--output_dir", type=str,
                        default=r"D:\paper\datasets\MSRS-main\test\text_seg")
    parser.add_argument("--csv_path", type=str,
                        default="msrs.csv")
    parser.add_argument("--seg_dir", type=str,
                        default=r"D:\paper\datasets\MSRS-main\test\color",
                        help="语义分割GT标签图目录 (可选, 已着色的彩色图)")
    args = parser.parse_args()

    for d, n in [(args.ir_dir, "IR"), (args.vis_dir, "VIS")]:
        if not os.path.isdir(d):
            print(f"❌ {n} 目录不存在: {d}")
            sys.exit(1)

    if args.seg_dir and not os.path.isdir(args.seg_dir):
        print(f"⚠️ GT目录不存在: {args.seg_dir}, 跳过GT显示")
        args.seg_dir = None

    print(f"\n{'═' * 58}")
    print(f"  FMB 标注工具 (含批量修改 & 筛选跳转 & GT可视化)")
    print(f"{'═' * 58}")
    print(f"  IR:     {args.ir_dir}")
    print(f"  VIS:    {args.vis_dir}")
    print(f"  SEG GT: {args.seg_dir or '(无)'}")
    print(f"  OUT:    {args.output_dir}")
    print(f"  CSV:    {args.csv_path}")
    print(f"{'═' * 58}\n")

    root = tk.Tk()
    root.geometry("1280x960" if args.seg_dir else "1080x960")
    root.minsize(1200 if args.seg_dir else 1040, 920)

    App(root, args.ir_dir, args.vis_dir, args.output_dir,
        args.csv_path, seg_dir=args.seg_dir)

    root.mainloop()


if __name__ == "__main__":
    main()
