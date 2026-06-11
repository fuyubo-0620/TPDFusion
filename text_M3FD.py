"""
m3fd_annotation_tool.py  (M3FD 专用优化版 v2)

M3FD 数据集标注工具 — 融合 + 目标检测双任务

改进 v2:
  ★ 支持 det_done 字段: 在融合标注基础上继续标注检测
  ★ GUI 紧凑布局: 所有按钮和字段都可见, 不再被截断
  ★ 检测模式自动跳转 det_done=no 的图
  ★ 保存时自动标记 det_done=yes
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
    "detection": (
        "Infrared and visible image fusion for object detection"
        "{in_scene}, focusing on {target}. "
        "{size} targets with {thermal} thermal contrast. "
        "{difficulty} detection difficulty.{condition_suffix}"
    ),
    "detection_hard": (
        "Infrared and visible image fusion for object detection"
        "{in_scene}. "
        "Enhance {target} detectability. {size} targets, "
        "{thermal} thermal contrast, {difficulty} difficulty due to "
        "{challenge}.{condition_suffix}"
    ),
}


# ══════════════════════════════════════════════════════════════════
#  字段值映射
# ══════════════════════════════════════════════════════════════════

TARGET_MAP = {
    "default":              "",
    "people":               "pedestrian",
    "car":                  "car",
    "bus":                  "bus",
    "truck":                "truck",
    "motorcycle":           "motorcycle",
    "lamp":                 "lamp",
    "people_car":           "pedestrian and car",
    "people_car_bus":       "pedestrian, car and bus",
    "people_car_truck":     "pedestrian, car and truck",
    "people_car_bus_truck": "pedestrian, car, bus and truck",
    "car_bus":              "car and bus",
    "car_truck":            "car and truck",
    "car_bus_truck":        "car, bus and truck",
    "car_motorcycle":       "car and motorcycle",
    "people_motorcycle":    "pedestrian and motorcycle",
    "people_car_motorcycle":"pedestrian, car and motorcycle",
    "car_truck_bus_motorcycle": "car, truck, bus and motorcycle",
    "vehicles":             "vehicle",
    "people_vehicles":      "pedestrian and vehicle",
    "all":                  "all targets",
}

THERMAL_MAP = {
    "default":  "",
    "strong":   "Strong",
    "moderate": "Moderate",
    "weak":     "Weak",
}

CONDITION_MAP = {
    "default":          "",
    "day_bright":
        "Bright daytime with clear visible details and moderate infrared contrast",
    "day_overcast":
        "Overcast daytime with soft lighting and mild infrared contrast",
    "day_shadow":
        "Daytime with strong shadows hiding targets in visible image",
    "day_overexposure":
        "Overexposed daytime with washed-out visible image and saturated regions",
    "day_backlight":
        "Backlit daytime with silhouetted targets in visible image",
    "night_lit":
        "Nighttime with dim street lighting and clear infrared targets",
    "night_low_light":
        "Dark nighttime with very limited visible information",
    "night_dark":
        "Nearly black visible image relying on infrared",
    "night_glare":
        "Nighttime with headlight glare in visible image",
    "challenge_fog":
        "Foggy conditions severely degrading visible image contrast",
    "challenge_rain":
        "Rainy conditions with wet reflections and blurred visible image",
    "challenge_haze":
        "Light haze with reduced visible contrast and clarity",
    "challenge_smoke":
        "Local smoke partially obscuring targets in visible image",
    "challenge_dust":
        "Dusty conditions with scattered particles degrading visible image",
    "challenge_mixed":
        "Complex challenging conditions with multiple degradation factors",
}

SCENE_FEATURE_MAP = {
    "default":          "",
    "urban_road":       "an urban road with buildings and trees on both sides",
    "wide_road":        "a wide multi-lane road with open surroundings",
    "narrow_road":      "a narrow road with closely spaced buildings",
    "intersection":     "a busy intersection with traffic lights and crosswalks",
    "roundabout":       "a roundabout with converging traffic",
    "highway":          "a highway with fast-moving vehicles and guardrails",
    "bridge":           "a bridge or overpass with distant city view",
    "parking_lot":      "a parking lot with parked and moving vehicles",
    "gas_station":      "a gas station area with parked vehicles",
    "bus_stop":         "a bus stop area with waiting pedestrians",
    "campus_road":      "a campus road with pedestrians and cyclists",
    "residential":      "a residential area with low-rise buildings",
    "commercial":       "a commercial district with shops and signs",
    "industrial":       "an industrial area with warehouses and trucks",
    "tree_lined":       "a tree-lined road with partial canopy cover",
    "open_area":        "an open area with few obstructions",
    "dense_buildings":  "a densely built-up area with tall buildings",
    "sidewalk_crowd":   "a road with crowded sidewalks and pedestrians",
    "construction":     "a road near a construction site with barriers",
    "tunnel_entrance":  "a tunnel entrance with sharp light transition",
}

SIZE_MAP = {
    "default": "", "large": "Large", "medium": "Medium",
    "small": "Small distant", "mixed": "Mixed-size",
}

DIFFICULTY_MAP = {
    "default": "", "easy": "Easy", "moderate": "Moderate", "hard": "Hard",
}

CHALLENGE_MAP = {
    "default":           "",
    "none":              "",
    "partial_occlusion": "partial occlusion",
    "heavy_occlusion":   "heavy occlusion",
    "dense_crowd":       "dense clustering",
    "low_contrast":      "very low thermal contrast",
    "small_size":        "extremely small target size",
    "clutter":           "complex background clutter",
    "glare":             "headlight glare interference",
    "shadow":            "strong shadows",
    "fog_degradation":   "fog degradation",
    "smoke_obscure":     "local smoke or haze partially obscuring targets",
    "overexposure":      "overexposure in visible image",
}

FIELD_TO_MAP = {
    "task": None, "scene_feature": SCENE_FEATURE_MAP,
    "target": TARGET_MAP, "thermal": THERMAL_MAP,
    "condition": CONDITION_MAP, "size": SIZE_MAP,
    "difficulty": DIFFICULTY_MAP, "challenge": CHALLENGE_MAP,
}


# ══════════════════════════════════════════════════════════════════
#  Prompt 构建
# ══════════════════════════════════════════════════════════════════

def build_prompt(task="fusion", target="default", thermal="default",
                 condition="default", scene_feature="default",
                 size="default", difficulty="default",
                 challenge="default", **kwargs):

    raw = {
        "target":        TARGET_MAP.get(target, ""),
        "thermal":       THERMAL_MAP.get(thermal, ""),
        "condition":     CONDITION_MAP.get(condition, ""),
        "scene_feature": SCENE_FEATURE_MAP.get(scene_feature, ""),
        "size":          SIZE_MAP.get(size, ""),
        "difficulty":    DIFFICULTY_MAP.get(difficulty, ""),
        "challenge":     CHALLENGE_MAP.get(challenge, ""),
    }

    in_scene = f" in {raw['scene_feature']}" if raw["scene_feature"] else ""
    target_text = raw["target"] if raw["target"] else "salient targets"
    thermal_text = raw["thermal"] if raw["thermal"] else "Moderate"
    condition_suffix = f" {raw['condition']}." if raw["condition"] else ""
    size_text = raw["size"] if raw["size"] else "Medium"
    difficulty_text = raw["difficulty"] if raw["difficulty"] else "Moderate"
    challenge_text = raw["challenge"]

    fields = {
        "in_scene": in_scene, "target": target_text,
        "thermal": thermal_text, "condition_suffix": condition_suffix,
        "size": size_text, "difficulty": difficulty_text,
        "challenge": challenge_text,
    }

    if task == "detection":
        if challenge and challenge not in ("default", "none", "") and challenge_text:
            tmpl = TEMPLATES["detection_hard"]
        else:
            tmpl = TEMPLATES["detection"]
    else:
        tmpl = TEMPLATES["fusion"]

    prompt = tmpl.format(**fields)
    prompt = " ".join(prompt.split())
    prompt = prompt.replace("due to .", ".").replace("due to ,", ",").replace("..", ".")
    return prompt


def guess_condition_m3fd(vis_path, ir_path, name):
    full_path = (vis_path + ir_path).lower().replace("\\", "/")
    name_lower = name.lower()
    if "/overexposure/" in full_path or "overexp" in full_path:
        return "day_overexposure"
    if "/challenge/" in full_path:
        if any(k in full_path for k in ["fog", "mist"]):
            return "challenge_fog"
        if "rain" in full_path:
            return "challenge_rain"
        if any(k in full_path for k in ["haze", "smog"]):
            return "challenge_haze"
        return "challenge_mixed"
    if "/night/" in full_path or "/dark/" in full_path:
        return "night_lit"
    if "/day/" in full_path or "/daytime/" in full_path:
        return "day_bright"
    if any(k in name_lower for k in ["night", "dark", "nig"]):
        return "night_lit"
    if any(k in name_lower for k in ["overexp", "over", "bright"]):
        return "day_overexposure"
    if any(k in name_lower for k in ["fog", "haze"]):
        return "challenge_fog"
    if any(k in name_lower for k in ["rain", "wet"]):
        return "challenge_rain"
    if any(k in name_lower for k in ["smoke", "smog"]):
        return "challenge_smoke"
    return None


def build_dual_prompt(ann: dict) -> dict:
    fusion_ann = dict(ann)
    fusion_ann["task"] = "fusion"
    det_ann = dict(ann)
    det_ann["task"] = "detection"
    return {
        "fusion":    build_prompt(**{k: v for k, v in fusion_ann.items() if k in FIELDS}),
        "detection": build_prompt(**{k: v for k, v in det_ann.items() if k in FIELDS}),
    }


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
            files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
            files.extend(glob.glob(os.path.join(folder, "**", ext.upper()), recursive=True))
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


# ══════════════════════════════════════════════════════════════════
#  标注字段定义
# ══════════════════════════════════════════════════════════════════

FIELDS = OrderedDict([
    ("task", {
        "label": "📋 Task", "options": ["fusion", "detection"],
        "default": "detection", "group": "common",
    }),
    ("scene_feature", {
        "label": "🏙️ Scene", "options": list(SCENE_FEATURE_MAP.keys()),
        "default": "default", "group": "common",
    }),
    ("target", {
        "label": "🎯 Target", "options": list(TARGET_MAP.keys()),
        "default": "default", "group": "common",
    }),
    ("thermal", {
        "label": "🌡️ Thermal", "options": list(THERMAL_MAP.keys()),
        "default": "default", "group": "common",
    }),
    ("condition", {
        "label": "🌤️ Condition", "options": list(CONDITION_MAP.keys()),
        "default": "default", "group": "common",
    }),
    ("size", {
        "label": "📐 Size", "options": list(SIZE_MAP.keys()),
        "default": "default", "group": "detection",
    }),
    ("difficulty", {
        "label": "⚡ Difficulty", "options": list(DIFFICULTY_MAP.keys()),
        "default": "default", "group": "detection",
    }),
    ("challenge", {
        "label": "⚠️ Challenge", "options": list(CHALLENGE_MAP.keys()),
        "default": "default", "group": "detection",
    }),
])


# ══════════════════════════════════════════════════════════════════
#  ★ 主界面 — 紧凑布局 + det_done 支持
# ══════════════════════════════════════════════════════════════════

class App:

    def __init__(self, root, ir_dir, vis_dir, output_dir, csv_path,
                 dual_task=False):
        self.root = root
        self.root.title("M3FD 标注工具 v2  |  融合 + 检测  |  紧凑布局")
        self.root.configure(bg="#2b2b2b")

        self.output_dir = output_dir
        self.csv_path = csv_path
        self.dual_task = dual_task
        os.makedirs(output_dir, exist_ok=True)

        self.pairs = collect_pairs(ir_dir, vis_dir)
        self.name_to_idx = {n: i for i, (n, _, _) in enumerate(self.pairs)}
        self.idx = 0
        self.annotations = {}
        self._load_csv()

        # ★ 检测标注模式自动检测
        self.det_annotation_mode = any(
            "det_done" in ann for ann in self.annotations.values()
        )
        if self.det_annotation_mode:
            n_todo = sum(1 for ann in self.annotations.values()
                         if ann.get("det_done", "yes") == "no")
            n_done = sum(1 for ann in self.annotations.values()
                         if ann.get("det_done", "yes") == "yes")
            print(f"  ★ 检测标注模式: {n_todo} 张待标注, {n_done} 张已完成")

        self.filter_field = None
        self.filter_value = None
        self.filter_indices = []
        self.filter_pos = -1

        self.field_vars = {}
        self.img_size = (380, 285)  # ★ 缩小图像以腾出空间

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

    # ★ 支持 det_done 模式
    def _goto_first_unannotated(self):
        if self.det_annotation_mode:
            for i, (name, _, _) in enumerate(self.pairs):
                if name in self.annotations:
                    if self.annotations[name].get("det_done", "yes") == "no":
                        self.idx = i
                        return
            self.idx = 0  # 全部完成
        else:
            for i, (name, _, _) in enumerate(self.pairs):
                if name not in self.annotations:
                    self.idx = i
                    return

    # ──────────────────────────────────────────
    #  ★ 紧凑布局 GUI
    # ──────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",    background="#2b2b2b")
        style.configure("TLabel",    background="#2b2b2b", foreground="#e0e0e0",
                         font=("Microsoft YaHei", 9))
        style.configure("T.TLabel",  background="#2b2b2b", foreground="#00ccff",
                         font=("Microsoft YaHei", 10, "bold"))
        style.configure("S.TLabel",  background="#1e1e1e", foreground="#aaa",
                         font=("Consolas", 8))
        style.configure("D.TLabel",  background="#2b2b2b", foreground="#ffaa00",
                         font=("Microsoft YaHei", 9, "bold"))
        style.configure("H.TLabel",  background="#2b2b2b", foreground="#888",
                         font=("Microsoft YaHei", 8))
        style.configure("F.TLabel",  background="#2b2b2b", foreground="#ff6666",
                         font=("Microsoft YaHei", 9, "bold"))
        style.configure("Det.TLabel", background="#2b2b2b", foreground="#ff9900",
                         font=("Microsoft YaHei", 10, "bold"))

        # ═══ 顶栏: 进度 + 导航 + 模式指示 ═══
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=(5, 2))

        self.progress_var = tk.StringVar()
        ttk.Label(top, textvariable=self.progress_var,
                  style="T.TLabel").pack(side="left")

        # ★ 检测模式指示
        if self.det_annotation_mode:
            self.mode_var = tk.StringVar(value="🔍 检测标注模式")
            ttk.Label(top, textvariable=self.mode_var,
                      style="Det.TLabel").pack(side="left", padx=20)

        nav = ttk.Frame(top)
        nav.pack(side="right")
        ttk.Button(nav, text="◀", width=3,
                   command=self._prev).pack(side="left", padx=1)
        ttk.Button(nav, text="▶", width=3,
                   command=self._next).pack(side="left", padx=1)
        self.jump_var = tk.StringVar()
        ttk.Entry(nav, textvariable=self.jump_var, width=6,
                  font=("Consolas", 9)).pack(side="left", padx=1)
        ttk.Button(nav, text="Go", width=3,
                   command=self._jump).pack(side="left", padx=1)

        # ═══ 主体: 左=图像+信息  右=标注字段 ═══
        main_pane = ttk.Frame(self.root)
        main_pane.pack(fill="both", expand=True, padx=8, pady=2)

        # ── 左侧: 图像 ──
        left = ttk.Frame(main_pane)
        left.pack(side="left", fill="y", padx=(0, 5))

        for title, attr in [
            ("📷 VIS", "vis_canvas"),
            ("🌡 IR",  "ir_canvas"),
        ]:
            f = ttk.Frame(left)
            f.pack(pady=2)
            ttk.Label(f, text=title, style="T.TLabel").pack()
            canvas = tk.Canvas(f, width=self.img_size[0],
                               height=self.img_size[1],
                               bg="#1a1a1a", highlightthickness=1,
                               highlightbackground="#444")
            canvas.pack()
            setattr(self, attr, canvas)

        self.info_var = tk.StringVar()
        ttk.Label(left, textvariable=self.info_var,
                  style="S.TLabel").pack(fill="x")

        self.scene_type_var = tk.StringVar()
        ttk.Label(left, textvariable=self.scene_type_var,
                  style="D.TLabel").pack(fill="x", pady=2)

        # ── 右侧: 标注区域 (使用 Scrollable) ──
        right = ttk.Frame(main_pane)
        right.pack(side="left", fill="both", expand=True)

        # ★ 状态栏 (检测模式下显示融合信息)
        self.status_var = tk.StringVar()
        status_lbl = tk.Label(right, textvariable=self.status_var,
                               bg="#1a2a1a", fg="#66ff66",
                               font=("Microsoft YaHei", 9, "bold"),
                               anchor="w", padx=8, pady=3)
        status_lbl.pack(fill="x", pady=(0, 3))

        # ── 任务 + 双任务勾选 ──
        task_row = ttk.Frame(right)
        task_row.pack(fill="x", pady=1)

        ttk.Label(task_row, text="📋 Task", width=10, anchor="w").pack(side="left")
        self.field_vars["task"] = tk.StringVar(value="detection")
        ttk.Combobox(task_row, textvariable=self.field_vars["task"],
                      values=["fusion", "detection"], state="readonly",
                      width=14, font=("Consolas", 9)).pack(side="left", padx=3)
        self.field_vars["task"].trace_add("write", lambda *a: self._on_task_change())

        self.dual_var = tk.BooleanVar(value=self.dual_task)
        ttk.Checkbutton(task_row, text="★ 双任务输出",
                         variable=self.dual_var).pack(side="right", padx=5)

        self.task_desc_var = tk.StringVar()
        ttk.Label(task_row, textvariable=self.task_desc_var,
                  style="H.TLabel").pack(side="left", padx=5)

        # ── 公共字段 (2列 grid) ──
        ttk.Label(right, text="━ 公共字段 ━", style="T.TLabel").pack(anchor="w", pady=(3, 1))

        common_grid = ttk.Frame(right)
        common_grid.pack(fill="x", pady=1)

        # Row 0: scene_feature (宽)
        ttk.Label(common_grid, text="🏙️ Scene", width=10, anchor="w").grid(
            row=0, column=0, sticky="w", padx=2, pady=1)
        self.field_vars["scene_feature"] = tk.StringVar(value="default")
        self.field_vars["scene_feature"].trace_add("write", lambda *a: self._update_preview())
        ttk.Combobox(common_grid, textvariable=self.field_vars["scene_feature"],
                      values=list(SCENE_FEATURE_MAP.keys()), state="readonly",
                      width=20, font=("Consolas", 9)).grid(
            row=0, column=1, sticky="w", padx=2, pady=1)
        self.sf_hint = tk.StringVar()
        ttk.Label(common_grid, textvariable=self.sf_hint, style="H.TLabel").grid(
            row=0, column=2, sticky="w", padx=5, pady=1)
        self.field_vars["scene_feature"].trace_add("write",
            lambda *a: self.sf_hint.set(
                ("→ " + SCENE_FEATURE_MAP.get(
                    self.field_vars["scene_feature"].get(), "")[:40])
                if SCENE_FEATURE_MAP.get(
                    self.field_vars["scene_feature"].get(), "")
                else "→ (不描述)"))

        # Row 1: target
        ttk.Label(common_grid, text="🎯 Target", width=10, anchor="w").grid(
            row=1, column=0, sticky="w", padx=2, pady=1)
        self.field_vars["target"] = tk.StringVar(value="default")
        self.field_vars["target"].trace_add("write", lambda *a: self._update_preview())
        ttk.Combobox(common_grid, textvariable=self.field_vars["target"],
                      values=list(TARGET_MAP.keys()), state="readonly",
                      width=20, font=("Consolas", 9)).grid(
            row=1, column=1, sticky="w", padx=2, pady=1)

        # Row 2: thermal
        ttk.Label(common_grid, text="🌡️ Thermal", width=10, anchor="w").grid(
            row=2, column=0, sticky="w", padx=2, pady=1)
        self.field_vars["thermal"] = tk.StringVar(value="default")
        self.field_vars["thermal"].trace_add("write", lambda *a: self._update_preview())
        ttk.Combobox(common_grid, textvariable=self.field_vars["thermal"],
                      values=list(THERMAL_MAP.keys()), state="readonly",
                      width=20, font=("Consolas", 9)).grid(
            row=2, column=1, sticky="w", padx=2, pady=1)

        # Row 3: condition
        ttk.Label(common_grid, text="🌤️ Condition", width=10, anchor="w").grid(
            row=3, column=0, sticky="w", padx=2, pady=1)
        self.field_vars["condition"] = tk.StringVar(value="default")
        self.field_vars["condition"].trace_add("write", lambda *a: self._update_preview())
        ttk.Combobox(common_grid, textvariable=self.field_vars["condition"],
                      values=list(CONDITION_MAP.keys()), state="readonly",
                      width=20, font=("Consolas", 9)).grid(
            row=3, column=1, sticky="w", padx=2, pady=1)
        self.cond_hint = tk.StringVar()
        ttk.Label(common_grid, textvariable=self.cond_hint, style="H.TLabel").grid(
            row=3, column=2, sticky="w", padx=5, pady=1)
        self.field_vars["condition"].trace_add("write",
            lambda *a: self.cond_hint.set(
                ("→ " + CONDITION_MAP.get(
                    self.field_vars["condition"].get(), "")[:45])
                if CONDITION_MAP.get(
                    self.field_vars["condition"].get(), "")
                else "→ (不描述)"))

        # ── 检测专用字段 (也用 grid) ──
        self.det_title = ttk.Label(right, text="━ 检测专用 (size/difficulty/challenge) ━",
                                    style="D.TLabel")
        self.det_title.pack(anchor="w", pady=(4, 1))

        self.det_grid = ttk.Frame(right)
        self.det_grid.pack(fill="x", pady=1)

        for row_i, fname in enumerate(["size", "difficulty", "challenge"]):
            fdef = FIELDS[fname]
            ttk.Label(self.det_grid, text=fdef["label"], width=12, anchor="w").grid(
                row=row_i, column=0, sticky="w", padx=2, pady=1)
            var = tk.StringVar(value=fdef["default"])
            var.trace_add("write", lambda *a: self._update_preview())
            ttk.Combobox(self.det_grid, textvariable=var,
                          values=fdef["options"], state="readonly",
                          width=20, font=("Consolas", 9)).grid(
                row=row_i, column=1, sticky="w", padx=2, pady=1)
            self.field_vars[fname] = var

        # ── 筛选栏 (紧凑单行) ──
        filter_frame = ttk.Frame(right)
        filter_frame.pack(fill="x", pady=(4, 1))

        ttk.Label(filter_frame, text="🔍", style="F.TLabel").pack(side="left")

        self.filter_field_var = tk.StringVar(value="condition")
        ttk.Combobox(filter_frame, textvariable=self.filter_field_var,
                      values=list(FIELDS.keys()), state="readonly",
                      width=10, font=("Consolas", 8)).pack(side="left", padx=1)
        self.filter_field_var.trace_add("write",
            lambda *a: self._update_filter_value_options())

        self.filter_value_var = tk.StringVar()
        self.filter_value_combo = ttk.Combobox(
            filter_frame, textvariable=self.filter_value_var,
            state="readonly", width=14, font=("Consolas", 8))
        self.filter_value_combo.pack(side="left", padx=1)

        for txt, cmd in [("🔍", self._apply_filter), ("◀", self._filter_prev),
                          ("▶", self._filter_next), ("📋", self._filter_list),
                          ("✖", self._clear_filter)]:
            ttk.Button(filter_frame, text=txt, width=3,
                       command=cmd).pack(side="left", padx=1)

        self.filter_status_var = tk.StringVar()
        ttk.Label(filter_frame, textvariable=self.filter_status_var,
                  style="F.TLabel").pack(side="left", padx=3)

        self._update_filter_value_options()

        # ── Prompt 预览 ──
        ttk.Label(right, text="━ Prompt 预览 ━", style="T.TLabel").pack(anchor="w", pady=(4, 1))

        self.prompt_text = tk.Text(right, height=3, wrap="word",
                                    bg="#1a1a2e", fg="#00ff88",
                                    font=("Consolas", 9), relief="flat",
                                    padx=8, pady=4)
        self.prompt_text.pack(fill="x")
        self.prompt_text.configure(state="disabled")

        self.token_var = tk.StringVar()
        ttk.Label(right, textvariable=self.token_var,
                  style="S.TLabel").pack(anchor="e")

        # ── ★ 按钮区域 (两行紧凑排列) ──
        btn_frame1 = ttk.Frame(right)
        btn_frame1.pack(fill="x", pady=(4, 1))

        for text, cmd in [
            ("💾 保存&下一张 (Enter)", self._save_next),
            ("💾 仅保存",              self._save),
            ("📋 复制",                self._copy),
            ("📁 CSV",                 self._export_csv),
            ("📁 全部txt",             self._export_txt),
        ]:
            ttk.Button(btn_frame1, text=text, command=cmd).pack(side="left", padx=2)

        btn_frame2 = ttk.Frame(right)
        btn_frame2.pack(fill="x", pady=(1, 2))

        for text, cmd in [
            ("⚡ 批量标注剩余",  self._batch),
            ("✏️ 批量修改",     self._batch_modify),
            ("📊 统计",         self._stats),
            ("🔄 重置",         self._reset_all),
            ("★ 双任务txt",     self._export_dual_txt),
        ]:
            ttk.Button(btn_frame2, text=text, command=cmd).pack(side="left", padx=2)

        # ── 底部快捷键提示 ──
        ttk.Label(self.root,
                  text="  ← → 翻页 | Enter 保存下一张 | Ctrl+S CSV | Esc 退出",
                  style="S.TLabel").pack(fill="x", side="bottom", ipady=2)

        self._on_task_change()

    # ──────────────────────────────────────────
    #  任务切换
    # ──────────────────────────────────────────

    def _on_task_change(self):
        task = self.field_vars["task"].get()
        if task == "detection":
            self.det_title.pack(anchor="w", pady=(4, 1))
            self.det_grid.pack(fill="x", pady=1)
            self.task_desc_var.set("+size/diff/challenge")
        else:
            self.det_title.pack_forget()
            self.det_grid.pack_forget()
            self.task_desc_var.set("仅公共字段")
        self._update_preview()

    def _reset_all(self):
        for fname, fdef in FIELDS.items():
            self.field_vars[fname].set(fdef["default"])
        self.status_var.set("🔄 已重置为 default")

    def _update_preview(self):
        kwargs = {k: v.get() for k, v in self.field_vars.items()}
        prompt = build_prompt(**kwargs)

        self.prompt_text.configure(state="normal")
        self.prompt_text.delete("1.0", "end")

        if self.dual_var.get():
            dual = build_dual_prompt(kwargs)
            self.prompt_text.insert("1.0",
                f"[FUSION] {dual['fusion']}\n[DET] {dual['detection']}")
        else:
            self.prompt_text.insert("1.0", prompt)

        self.prompt_text.configure(state="disabled")
        self.token_var.set(f"  ~{len(prompt.split())}w | {len(prompt)}c")

    # ══════════════════════════════════════════════════════════════
    #  筛选功能
    # ══════════════════════════════════════════════════════════════

    def _update_filter_value_options(self):
        field = self.filter_field_var.get()

        # ★ 检测模式: 筛选列表增加 det_done 选项
        if field == "det_done" or (self.det_annotation_mode and field not in FIELDS):
            opts = ["no", "yes"]
        else:
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
            self.filter_status_var.set(f"⚠️ 0张匹配")
            self.filter_field = None
            return

        self.filter_pos = 0
        self.idx = self.filter_indices[0]
        n = len(self.filter_indices)
        self.filter_status_var.set(f"共{n}张 [1/{n}]")
        self._show()

    def _filter_prev(self):
        if not self.filter_indices:
            self.filter_status_var.set("⚠️ 请先筛选")
            return
        self.filter_pos = (self.filter_pos - 1) % len(self.filter_indices)
        self.idx = self.filter_indices[self.filter_pos]
        n = len(self.filter_indices)
        self.filter_status_var.set(f"[{self.filter_pos + 1}/{n}]")
        self._show()

    def _filter_next(self):
        if not self.filter_indices:
            self.filter_status_var.set("⚠️ 请先筛选")
            return
        self.filter_pos = (self.filter_pos + 1) % len(self.filter_indices)
        self.idx = self.filter_indices[self.filter_pos]
        n = len(self.filter_indices)
        self.filter_status_var.set(f"[{self.filter_pos + 1}/{n}]")
        self._show()

    def _filter_list(self):
        if not self.filter_indices:
            self.filter_status_var.set("⚠️ 请先筛选")
            return

        win = tk.Toplevel(self.root)
        win.title(f"筛选: {self.filter_field}={self.filter_value}  "
                  f"({len(self.filter_indices)}张)")
        win.geometry("400x500")
        win.configure(bg="#1e1e1e")

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=5)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        listbox = tk.Listbox(frame, bg="#1e1e1e", fg="#00ff88",
                              font=("Consolas", 10), selectmode="single",
                              yscrollcommand=scrollbar.set)
        listbox.pack(fill="both", expand=True)
        scrollbar.config(command=listbox.yview)

        for idx in self.filter_indices:
            name = self.pairs[idx][0]
            # ★ 显示 det_done 状态
            if self.det_annotation_mode and name in self.annotations:
                done = self.annotations[name].get("det_done", "?")
                tag = "✅" if done == "yes" else "⬜"
            else:
                tag = ""
            listbox.insert("end", f"  [{idx + 1}]  {name}  {tag}")

        def on_select(event):
            sel = listbox.curselection()
            if sel:
                pos = sel[0]
                self.filter_pos = pos
                self.idx = self.filter_indices[pos]
                n = len(self.filter_indices)
                self.filter_status_var.set(f"[{pos + 1}/{n}]")
                self._show()

        listbox.bind("<<ListboxSelect>>", on_select)

    def _clear_filter(self):
        self.filter_field = None
        self.filter_value = None
        self.filter_indices = []
        self.filter_pos = -1
        self.filter_status_var.set("")

    # ══════════════════════════════════════════════════════════════
    #  批量修改
    # ══════════════════════════════════════════════════════════════

    def _batch_modify(self):
        if not self.annotations:
            messagebox.showinfo("批量修改", "暂无标注数据")
            return

        win = tk.Toplevel(self.root)
        win.title("✏️ 批量修改标注")
        win.geometry("550x420")
        win.configure(bg="#2b2b2b")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="✏️ 批量修改标注",
                  style="T.TLabel").pack(pady=(10, 5))

        main_f = ttk.Frame(win)
        main_f.pack(fill="x", padx=20, pady=10)

        # ★ 字段选项增加 det_done
        field_options = list(FIELDS.keys())
        if self.det_annotation_mode:
            field_options.append("det_done")

        ttk.Label(main_f, text="① 字段:").pack(anchor="w", pady=(3, 1))
        mod_field_var = tk.StringVar(value="condition")
        ttk.Combobox(main_f, textvariable=mod_field_var,
                      values=field_options, state="readonly",
                      width=25, font=("Consolas", 10)).pack(anchor="w", padx=10)

        ttk.Label(main_f, text="② 当前值 (旧):").pack(anchor="w", pady=(8, 1))
        old_val_var = tk.StringVar()
        old_val_combo = ttk.Combobox(main_f, textvariable=old_val_var,
                                      state="readonly", width=25,
                                      font=("Consolas", 10))
        old_val_combo.pack(anchor="w", padx=10)

        ttk.Label(main_f, text="③ 替换为 (新):").pack(anchor="w", pady=(8, 1))
        new_val_var = tk.StringVar()
        new_val_combo = ttk.Combobox(main_f, textvariable=new_val_var,
                                      state="readonly", width=25,
                                      font=("Consolas", 10))
        new_val_combo.pack(anchor="w", padx=10)

        preview_var = tk.StringVar(value="...")
        tk.Label(win, textvariable=preview_var, bg="#1a2a1a", fg="#ffcc00",
                 font=("Consolas", 9), anchor="w", wraplength=500,
                 padx=10, pady=6).pack(fill="x", padx=20, pady=8)

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

            if field == "det_done":
                all_opts = ["no", "yes"]
            else:
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
                preview_var.set("请选择")
                return
            count = sum(1 for ann in self.annotations.values()
                        if ann.get(field, "") == old_v)
            preview_var.set(f"将修改 {count} 张:  {field}: {old_v} → {new_v}")

        mod_field_var.trace_add("write", update_old_options)
        old_val_var.trace_add("write", update_preview)
        new_val_var.trace_add("write", update_preview)
        update_old_options()

        btn_f = ttk.Frame(win)
        btn_f.pack(pady=8)

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
            if not messagebox.askyesno("确认",
                f"将 {len(names)} 张图的 {field}: {old_v} → {new_v}\n不可撤销！"):
                return

            for name in names:
                self.annotations[name][field] = new_v

            self._export_csv(silent=True)
            # 仅对非 det_done 字段重新生成 txt
            if field != "det_done":
                for name in names:
                    self._write_txt_for(name)

            messagebox.showinfo("完成", f"已修改 {len(names)} 张")
            update_old_options()
            self._show()

        ttk.Button(btn_f, text="✅ 执行", command=do_modify).pack(side="left", padx=5)
        ttk.Button(btn_f, text="❌ 取消", command=win.destroy).pack(side="left", padx=5)

    # ──────────────────────────────────────────
    #  ★ 图像显示 (支持 det_done)
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

        # ★ 进度显示
        n_total = len(self.pairs)
        if self.det_annotation_mode:
            n_det_done = sum(1 for ann in self.annotations.values()
                             if ann.get("det_done", "yes") == "yes")
            n_det_todo = sum(1 for ann in self.annotations.values()
                             if ann.get("det_done", "yes") == "no")
            pct = n_det_done / max(n_total, 1) * 100
            self.progress_var.set(
                f"[{self.idx + 1}/{n_total}] {name}  "
                f"检测: ✅{n_det_done} ⬜{n_det_todo} ({pct:.0f}%)")
        else:
            n_done = len(self.annotations)
            pct = n_done / max(n_total, 1) * 100
            self.progress_var.set(
                f"[{self.idx + 1}/{n_total}] {name}  "
                f"(已标 {n_done}/{n_total} {pct:.0f}%)")

        self.info_var.set(
            f"  {name} | VIS:{vis_img.size[0]}×{vis_img.size[1]}  "
            f"IR:{ir_img.size[0]}×{ir_img.size[1]}")

        guessed = guess_condition_m3fd(vis_path, ir_path, name)
        if guessed:
            self.scene_type_var.set(f"🏷️ 推断: {guessed}")
        else:
            self.scene_type_var.set(f"🏷️ 未自动识别")

        if name in self.annotations:
            ann = self.annotations[name]
            for k, v in self.field_vars.items():
                val = ann.get(k, "")
                if val:
                    v.set(val)

            # ★ 检测模式下区分状态
            if self.det_annotation_mode:
                det_done = ann.get("det_done", "no")
                if det_done == "yes":
                    self.status_var.set(
                        f"✅ 检测已标注  |  cond={ann.get('condition','?')}  "
                        f"tgt={ann.get('target','?')}  "
                        f"size={ann.get('size','?')}  diff={ann.get('difficulty','?')}")
                else:
                    self.status_var.set(
                        f"🔍 待标检测  |  融合已有: cond={ann.get('condition','?')}  "
                        f"tgt={ann.get('target','?')}  thermal={ann.get('thermal','?')}  "
                        f"→ 请补充 size/difficulty/challenge")
            else:
                self.status_var.set("✅ 已标注 (可修改)")
        else:
            if guessed:
                self.field_vars["condition"].set(guessed)
            self.status_var.set("⬜ 未标注")

        self._update_preview()
        self.jump_var.set(str(self.idx + 1))

    def _fit(self, img):
        tw, th = self.img_size
        r = min(tw / img.width, th / img.height)
        return img.resize((int(img.width * r), int(img.height * r)),
                          Image.LANCZOS)

    # ──────────────────────────────────────────
    #  通用 txt 写入
    # ──────────────────────────────────────────

    def _write_txt_for(self, name):
        ann = self.annotations.get(name)
        if ann is None:
            return
        filtered_ann = {k: v for k, v in ann.items() if k in FIELDS}

        if self.dual_var.get():
            fusion_dir = os.path.join(self.output_dir, "text_fusion")
            det_dir    = os.path.join(self.output_dir, "text_det")
            os.makedirs(fusion_dir, exist_ok=True)
            os.makedirs(det_dir, exist_ok=True)
            dual = build_dual_prompt(filtered_ann)
            with open(os.path.join(fusion_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(dual["fusion"])
            with open(os.path.join(det_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(dual["detection"])
        else:
            prompt = build_prompt(**filtered_ann)
            with open(os.path.join(self.output_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)

    # ──────────────────────────────────────────
    #  ★ 保存 (支持 det_done)
    # ──────────────────────────────────────────

    def _save(self):
        name = self.pairs[self.idx][0]
        ann = {k: v.get() for k, v in self.field_vars.items()}

        # ★ 检测模式: 保存时标记完成
        if self.det_annotation_mode:
            ann["det_done"] = "yes"

        self.annotations[name] = ann
        self._write_txt_for(name)
        self._export_csv(silent=True)
        self.status_var.set(f"✅ 已保存 {name}")
        self._show()

    # ★ 保存并跳到下一个未完成的
    def _save_next(self):
        self._save()

        if self.filter_indices:
            self._filter_next()
            return

        if self.det_annotation_mode:
            # ★ 跳到下一个 det_done=no
            start = self.idx + 1
            for offset in range(len(self.pairs)):
                check_idx = (start + offset) % len(self.pairs)
                check_name = self.pairs[check_idx][0]
                if check_name in self.annotations:
                    if self.annotations[check_name].get("det_done", "yes") == "no":
                        self.idx = check_idx
                        self._show()
                        return
            # 全部完成
            n_done = sum(1 for ann in self.annotations.values()
                         if ann.get("det_done", "") == "yes")
            messagebox.showinfo("🎉 完成!",
                f"所有图像的检测标注已完成!\n共 {n_done} 张")
            self._next()
        else:
            self._next()

    # ★ CSV 导出 (包含 det_done)
    def _export_csv(self, silent=False):
        base_cols = ["name"] + list(FIELDS.keys())
        if self.det_annotation_mode:
            base_cols.append("det_done")
        cols = base_cols + ["prompt_fusion", "prompt_detection"]

        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for name, _, _ in self.pairs:
                    if name in self.annotations:
                        row = {"name": name}
                        row.update(self.annotations[name])
                        filtered = {k: v for k, v in
                                    self.annotations[name].items() if k in FIELDS}
                        dual = build_dual_prompt(filtered)
                        row["prompt_fusion"]    = dual["fusion"]
                        row["prompt_detection"] = dual["detection"]
                        w.writerow(row)
            if not silent:
                if self.det_annotation_mode:
                    n_done = sum(1 for ann in self.annotations.values()
                                 if ann.get("det_done", "no") == "yes")
                    messagebox.showinfo("导出",
                        f"CSV: {self.csv_path}\n"
                        f"共 {len(self.annotations)} 条\n"
                        f"检测已完成: {n_done}/{len(self.annotations)}")
                else:
                    messagebox.showinfo("导出",
                        f"CSV: {self.csv_path}\n共 {len(self.annotations)} 条")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _export_txt(self):
        count = 0
        for name, _, _ in self.pairs:
            if name in self.annotations:
                self._write_txt_for(name)
                count += 1
        msg = f"已导出 {count} 个 .txt"
        if self.dual_var.get():
            msg += f"\n  fusion → text_fusion/\n  det → text_det/"
        messagebox.showinfo("导出完成", msg)

    def _export_dual_txt(self):
        if not self.annotations:
            messagebox.showinfo("提示", "暂无标注数据")
            return

        fusion_dir = os.path.join(self.output_dir, "text_fusion")
        det_dir    = os.path.join(self.output_dir, "text_det")
        os.makedirs(fusion_dir, exist_ok=True)
        os.makedirs(det_dir, exist_ok=True)

        count = 0
        for name, _, _ in self.pairs:
            if name in self.annotations:
                filtered = {k: v for k, v in
                            self.annotations[name].items() if k in FIELDS}
                dual = build_dual_prompt(filtered)
                with open(os.path.join(fusion_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                    f.write(dual["fusion"])
                with open(os.path.join(det_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                    f.write(dual["detection"])
                count += 1

        messagebox.showinfo("双任务导出",
            f"已导出 {count} 张:\n"
            f"  🔗 fusion → {fusion_dir}/\n"
            f"  🔍 detection → {det_dir}/")

    # ──────────────────────────────────────────
    #  批量标注
    # ──────────────────────────────────────────

    def _batch(self):
        if self.det_annotation_mode:
            # ★ 检测模式: 批量标注所有 det_done=no 的
            un = [n for n, ann in self.annotations.items()
                  if ann.get("det_done", "yes") == "no"]
            if not un:
                messagebox.showinfo("完成", "全部检测已标注!")
                return

            cur = {k: v.get() for k, v in self.field_vars.items()}
            msg = (
                f"将当前检测字段应用到 {len(un)} 张未完成的图像:\n\n"
                f"  size={cur.get('size','default')}\n"
                f"  difficulty={cur.get('difficulty','default')}\n"
                f"  challenge={cur.get('challenge','default')}\n\n"
                f"(公共字段保留原有融合标注)\n继续?"
            )

            if not messagebox.askyesno("批量检测标注", msg):
                return

            for name in un:
                # 只更新检测字段, 保留融合字段
                self.annotations[name]["size"]       = cur.get("size", "default")
                self.annotations[name]["difficulty"]  = cur.get("difficulty", "default")
                self.annotations[name]["challenge"]   = cur.get("challenge", "default")
                self.annotations[name]["task"]        = "detection"
                self.annotations[name]["det_done"]    = "yes"

        else:
            un = [n for n, _, _ in self.pairs if n not in self.annotations]
            if not un:
                messagebox.showinfo("完成", "全部已标注!")
                return

            cur = {k: v.get() for k, v in self.field_vars.items()}
            preview = build_prompt(**cur)
            msg = (
                f"将当前设置应用到 {len(un)} 张未标注图像:\n\n"
                f"预览:\n{preview}\n\n继续?"
            )

            if not messagebox.askyesno("批量标注确认", msg):
                return

            for name in un:
                a = cur.copy()
                for n, vp, ip in self.pairs:
                    if n == name:
                        g = guess_condition_m3fd(vp, ip, name)
                        if g:
                            a["condition"] = g
                        break
                self.annotations[name] = a

        self._export_csv(silent=True)
        self._export_txt()
        messagebox.showinfo("批量完成",
            f"已处理 {len(un)} 张\n总计: {len(self.annotations)}/{len(self.pairs)}")
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

        lines = [f"═══ M3FD 标注统计 ═══\n",
                 f"总图像对:  {n_total}",
                 f"已标注:    {n}  ({n / n_total * 100:.1f}%)",
                 f"未标注:    {n_total - n}\n"]

        # ★ 检测标注进度
        if self.det_annotation_mode:
            n_det_done = sum(1 for ann in self.annotations.values()
                             if ann.get("det_done", "no") == "yes")
            n_det_todo = sum(1 for ann in self.annotations.values()
                             if ann.get("det_done", "no") == "no")
            lines.append(f"── 检测标注进度 ──")
            lines.append(f"  ✅ 已完成:  {n_det_done}")
            lines.append(f"  ⬜ 待标注:  {n_det_todo}")
            lines.append(f"  进度:       {n_det_done / max(n, 1) * 100:.1f}%\n")

        for field in FIELDS:
            c = Counter(a.get(field, "?") for a in self.annotations.values())
            lines.append(f"── {field} ──")
            for val, cnt in c.most_common():
                bar = "█" * max(1, int(cnt / n * 30))
                lines.append(f"  {val:<28s} {cnt:>4d}  {bar}")
            lines.append("")

        lines.append(f"── M3FD 场景分布 ──")
        scene_counter = Counter()
        for ann in self.annotations.values():
            cond = ann.get("condition", "default")
            if cond.startswith("day"):     scene_counter["Day"] += 1
            elif cond.startswith("night"): scene_counter["Night"] += 1
            elif "overexposure" in cond or "backlight" in cond:
                scene_counter["Overexposure"] += 1
            elif cond.startswith("challenge"): scene_counter["Challenge"] += 1
            else: scene_counter["Other"] += 1
        for cat, cnt in scene_counter.most_common():
            pct = cnt / n * 100
            bar = "█" * max(1, int(pct / 3))
            lines.append(f"  {cat:<18s} {cnt:>4d} ({pct:5.1f}%)  {bar}")
        lines.append("")

        lines.append(f"── default 占比 ──")
        for field in FIELDS:
            if field == "task":
                continue
            dc = sum(1 for a in self.annotations.values()
                     if a.get(field, "") == "default")
            pct = dc / n * 100 if n > 0 else 0
            lines.append(f"  {field:<28s} {dc:>4d} ({pct:.0f}%)")

        win = tk.Toplevel(self.root)
        win.title("M3FD 标注统计")
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
        description="M3FD 标注工具 v2 — 融合 + 检测 + det_done"
    )
    parser.add_argument("--ir_dir", type=str,
                        default=r"D:\paper\datasets\text\M3FD_det\ir")
    parser.add_argument("--vis_dir", type=str,
                        default=r"D:\paper\datasets\text\M3FD_det\vi")
    parser.add_argument("--output_dir", type=str,
                        default=r"D:\paper\datasets\text\M3FD_det\text")
    parser.add_argument("--csv_path", type=str,
                        default="m3fd_annotations.csv")
    parser.add_argument("--dual_task", action="store_true", default=False)
    args = parser.parse_args()

    for d, n in [(args.ir_dir, "IR"), (args.vis_dir, "VIS")]:
        if not os.path.isdir(d):
            print(f"❌ {n} 目录不存在: {d}")
            sys.exit(1)

    print(f"\n{'═' * 58}")
    print(f"  M3FD 标注工具 v2 (融合 + 检测 + det_done)")
    print(f"{'═' * 58}")
    print(f"  IR:   {args.ir_dir}")
    print(f"  VIS:  {args.vis_dir}")
    print(f"  OUT:  {args.output_dir}")
    print(f"  CSV:  {args.csv_path}")
    print(f"{'═' * 58}\n")

    root = tk.Tk()
    root.geometry("1150x780")
    root.minsize(1100, 720)

    App(root, args.ir_dir, args.vis_dir, args.output_dir,
        args.csv_path, dual_task=args.dual_task)

    root.mainloop()


if __name__ == "__main__":
    main()
