"""
annotation_tool.py

红外-可见光图像对标注工具 (GUI)

功能:
  ✅ 左右并排显示 VIS / IR 图像
  ✅ 下拉框选择各标注字段
  ✅ 根据 task 类型动态显示/隐藏检测/分割专用字段
  ✅ 实时预览生成的 CLIP prompt
  ✅ 保存标注到 CSV + 每张图一个 .txt
  ✅ 支持上一张 / 下一张 / 跳转
  ✅ 支持断点续标 (自动加载已有 CSV)
  ✅ 标注进度显示
  ✅ 键盘快捷键 (← → Enter)

用法:
  python annotation_tool.py
  python annotation_tool.py --ir_dir train_text/ir --vis_dir train_text/vi
  python annotation_tool.py --ir_dir train_text/ir --vis_dir train_text/vi --output_dir train_text/text --csv_path annotations.csv

依赖:
  pip install Pillow   (图像显示)
  其余全部 Python 标准库
"""

import os
import sys
import csv
import glob
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import OrderedDict

try:
    from PIL import Image, ImageTk
except ImportError:
    print("❌ 请先安装 Pillow:")
    print("   pip install Pillow")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════
#  Prompt 模板 & 构建器
# ══════════════════════════════════════════════════════════════════

TEMPLATES = {
    "detection": {
        "base": (
            "Infrared and visible image fusion for object detection, "
            "focusing on {target}. "
            "{size} targets with {thermal} thermal contrast. "
            "{difficulty} detection difficulty. {condition}."
        ),
        "hard_case": (
            "Infrared and visible image fusion for object detection. "
            "Enhance {target} detectability. {size} targets, "
            "{thermal} thermal contrast, {difficulty} difficulty due to "
            "{challenge}. {condition}."
        ),

    },
    "segmentation": {
        "base": (
            "Infrared and visible image fusion for semantic segmentation. "
            "Scene contains {seg_classes}. "
            "{boundary} between categories. "
            "{target} instances present. {condition}."
        ),
        "boundary_focus": (
            "Infrared and visible image fusion for pixel-level segmentation. "
            "Enhance category boundaries, especially {seg_classes}. "
            "Thermal cues help separate {target} from background. "
            "{boundary}. {condition}."
        ),
    },
    "general": {
        "base": (
            "Infrared and visible image fusion, "
            "focusing on {target}. "
            "{thermal} thermal contrast, {condition}."
        ),
    },
    "detection_segmentation": {
        "base": (
            "Infrared and visible image fusion for detection and segmentation. "
            "Preserve {target} thermal signatures for detection. "
            "Maintain {seg_classes} boundaries for segmentation. "
            "{thermal} contrast, {condition}."
        ),
    },
}


class PromptBuilder:
    TARGET_MAP = {
        "pedestrian": "pedestrian",
        "vehicle": "vehicle",
        "cyclist": "cyclist",
        "pedestrian_vehicle": "pedestrian and vehicle",
        "pedestrian_cyclist": "pedestrian and cyclist",
        "all_targets": "all thermal targets",
        "default": "all infrared information",
    }
    THERMAL_MAP = {
        "strong": "strong",
        "moderate": "moderate",
        "weak": "weak",
    }
    CONDITION_MAP = {
        "night_clear": "nighttime clear conditions",
        "night_low_light": "nighttime with very low ambient lighting",
        "day_clear": "daytime clear conditions",
        "day_overcast": "daytime overcast sky",
        "fog": "foggy conditions with reduced visibility",
        "rain": "rainy conditions",
    }
    SIZE_MAP = {
        "large": "Large nearby",
        "medium": "Medium",
        "small": "Small distant",
        "mixed": "Mixed-size",
    }
    DIFFICULTY_MAP = {
        "easy": "Easy",
        "moderate": "Moderate",
        "hard": "Hard",
    }
    SEG_CLASSES_MAP = {
        "road_building": "road and building",
        "road_sidewalk_building": "road, sidewalk and building",
        "road_vegetation_building": "road, vegetation and building",
        "road_sidewalk_vegetation_sky": "road, sidewalk, vegetation and sky",
    }
    BOUNDARY_MAP = {
        "clear_boundary": "Clear boundaries",
        "ambiguous_boundary": "Ambiguous boundaries due to similar materials",
        "complex_boundary": "Complex fragmented boundaries",
    }
    CHALLENGE_MAP = {
        "partial_occlusion": "partial occlusion",
        "heavy_occlusion": "heavy occlusion",
        "dense_crowd": "dense clustering",
        "low_contrast": "very low thermal contrast",
        "small_size": "extremely small target size",
        "edge_target": "targets near image edges",
        "": "",
    }

    def build(self, task="general", target="default", thermal="moderate",
              condition="night_clear", size="medium", difficulty="moderate",
              challenge="", seg_classes="road_building",
              boundary="clear_boundary", variant="base"):
        fields = {
            "target": self.TARGET_MAP.get(target, target),
            "thermal": self.THERMAL_MAP.get(thermal, thermal),
            "condition": self.CONDITION_MAP.get(condition, condition),
            "size": self.SIZE_MAP.get(size, size),
            "difficulty": self.DIFFICULTY_MAP.get(difficulty, difficulty),
            "seg_classes": self.SEG_CLASSES_MAP.get(seg_classes, seg_classes),
            "boundary": self.BOUNDARY_MAP.get(boundary, boundary),
            "challenge": self.CHALLENGE_MAP.get(challenge, challenge) if challenge else "",
        }
        if task in TEMPLATES and variant in TEMPLATES[task]:
            template = TEMPLATES[task][variant]
        elif task in TEMPLATES and "base" in TEMPLATES[task]:
            template = TEMPLATES[task]["base"]
        else:
            template = TEMPLATES["general"]["base"]
        prompt = template.format(**fields)
        prompt = " ".join(prompt.split())
        return prompt


# ══════════════════════════════════════════════════════════════════
#  图像对收集
# ══════════════════════════════════════════════════════════════════

def collect_image_pairs(ir_dir, vis_dir):
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]

    def _collect(folder):
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))
        return sorted(set(files))

    ir_files = _collect(ir_dir)
    vis_files = _collect(vis_dir)

    ir_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in ir_files}
    vis_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in vis_files}

    common = sorted(set(ir_dict.keys()) & set(vis_dict.keys()))
    if not common:
        messagebox.showerror("错误", f"未找到匹配的图像对!\nIR: {ir_dir}\nVIS: {vis_dir}")
        sys.exit(1)

    pairs = [(name, vis_dict[name], ir_dict[name]) for name in common]
    return pairs


# ══════════════════════════════════════════════════════════════════
#  标注字段定义
# ══════════════════════════════════════════════════════════════════

FIELD_DEFS = OrderedDict([
    ("task", {
        "label": "任务类型 Task",
        "options": ["detection", "segmentation", "general", "detection_segmentation"],
        "default": "detection",
        "group": "common",
    }),
    ("target", {
        "label": "目标类别 Target",
        "options": ["pedestrian", "vehicle", "cyclist",
                     "pedestrian_vehicle", "pedestrian_cyclist",
                     "all_targets", "default"],
        "default": "pedestrian",
        "group": "common",
    }),
    ("thermal", {
        "label": "热特征强度 Thermal",
        "options": ["strong", "moderate", "weak"],
        "default": "strong",
        "group": "common",
    }),
    ("condition", {
        "label": "场景条件 Condition",
        "options": ["night_clear", "night_low_light", "day_clear",
                     "day_overcast", "fog", "rain"],
        "default": "night_clear",
        "group": "common",
    }),
    ("variant", {
        "label": "模板变体 Variant",
        "options": ["base", "hard_case", "boundary_focus"],
        "default": "base",
        "group": "common",
    }),
    # ── 检测专用 ──
    ("size", {
        "label": "目标尺寸 Size (检测)",
        "options": ["large", "medium", "small", "mixed"],
        "default": "medium",
        "group": "detection",
    }),
    ("difficulty", {
        "label": "检测难度 Difficulty (检测)",
        "options": ["easy", "moderate", "hard"],
        "default": "moderate",
        "group": "detection",
    }),
    ("challenge", {
        "label": "困难原因 Challenge (检测,可选)",
        "options": ["", "partial_occlusion", "heavy_occlusion",
                     "dense_crowd", "low_contrast", "small_size", "edge_target"],
        "default": "",
        "group": "detection",
    }),
    # ── 分割专用 ──
    ("seg_classes", {
        "label": "语义类别 Seg Classes (分割)",
        "options": ["road_building", "road_sidewalk_building",
                     "road_vegetation_building", "road_sidewalk_vegetation_sky"],
        "default": "road_building",
        "group": "segmentation",
    }),
    ("boundary", {
        "label": "边界清晰度 Boundary (分割)",
        "options": ["clear_boundary", "ambiguous_boundary", "complex_boundary"],
        "default": "clear_boundary",
        "group": "segmentation",
    }),
])

# 各任务需要显示的字段组
TASK_GROUPS = {
    "detection": {"common", "detection"},
    "segmentation": {"common", "segmentation"},
    "general": {"common"},
    "detection_segmentation": {"common", "detection", "segmentation"},
}


# ══════════════════════════════════════════════════════════════════
#  主界面
# ══════════════════════════════════════════════════════════════════

class AnnotationTool:

    def __init__(self, root, ir_dir, vis_dir, output_dir, csv_path):
        self.root = root
        self.root.title("红外-可见光 图像对标注工具  |  IR-VIS Annotation Tool")
        self.root.configure(bg="#2b2b2b")

        self.ir_dir = ir_dir
        self.vis_dir = vis_dir
        self.output_dir = output_dir
        self.csv_path = csv_path
        os.makedirs(output_dir, exist_ok=True)

        # ── 数据 ──
        self.pairs = collect_image_pairs(ir_dir, vis_dir)
        self.current_idx = 0
        self.builder = PromptBuilder()

        # ── 已有标注 ──
        self.annotations = {}  # name → {field: value}
        self._load_existing_csv()

        # ── 界面变量 ──
        self.field_vars = {}   # field_name → tk.StringVar
        self.field_widgets = {}  # field_name → (label_widget, combo_widget)
        self.img_display_size = (420, 336)  # 显示尺寸

        # ── 构建界面 ──
        self._build_ui()

        # ── 显示第一张 ──
        self._find_first_unannotated()
        self._show_current()

        # ── 键盘绑定 ──
        self.root.bind("<Left>", lambda e: self._go_prev())
        self.root.bind("<Right>", lambda e: self._go_next())
        self.root.bind("<Return>", lambda e: self._save_and_next())
        self.root.bind("<Control-s>", lambda e: self._save_csv())
        self.root.bind("<Escape>", lambda e: self._on_close())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────────────────────────────────────
    #  加载已有标注
    # ──────────────────────────────────────────────────────

    def _load_existing_csv(self):
        if not os.path.exists(self.csv_path):
            return
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("name", "").strip()
                    if name:
                        self.annotations[name] = {
                            k: v.strip() for k, v in row.items() if k != "name"
                        }
            print(f"  已加载 {len(self.annotations)} 条标注 from {self.csv_path}")
        except Exception as e:
            print(f"  ⚠️ 加载 CSV 失败: {e}")

    def _find_first_unannotated(self):
        for i, (name, _, _) in enumerate(self.pairs):
            if name not in self.annotations:
                self.current_idx = i
                return
        self.current_idx = 0  # 全部已标注, 从头开始

    # ──────────────────────────────────────────────────────
    #  构建界面
    # ──────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="#e0e0e0",
                         font=("Microsoft YaHei", 10))
        style.configure("Title.TLabel", background="#2b2b2b", foreground="#00ccff",
                         font=("Microsoft YaHei", 12, "bold"))
        style.configure("Status.TLabel", background="#1e1e1e", foreground="#aaaaaa",
                         font=("Consolas", 9))
        style.configure("Prompt.TLabel", background="#1a1a2e", foreground="#00ff88",
                         font=("Consolas", 10), wraplength=900)
        style.configure("TCombobox", font=("Consolas", 10))
        style.configure("Accent.TButton", font=("Microsoft YaHei", 11, "bold"))

        # ═══ 顶部: 进度 + 导航 ═══
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x", padx=10, pady=(10, 5))

        self.progress_var = tk.StringVar()
        ttk.Label(top_frame, textvariable=self.progress_var,
                  style="Title.TLabel").pack(side="left")

        nav_frame = ttk.Frame(top_frame)
        nav_frame.pack(side="right")

        ttk.Button(nav_frame, text="◀ 上一张 (←)",
                   command=self._go_prev).pack(side="left", padx=3)
        ttk.Button(nav_frame, text="下一张 (→) ▶",
                   command=self._go_next).pack(side="left", padx=3)

        self.jump_var = tk.StringVar()
        ttk.Entry(nav_frame, textvariable=self.jump_var,
                  width=8, font=("Consolas", 10)).pack(side="left", padx=3)
        ttk.Button(nav_frame, text="跳转",
                   command=self._jump_to).pack(side="left", padx=3)

        # ═══ 图像区域 ═══
        img_frame = ttk.Frame(self.root)
        img_frame.pack(fill="x", padx=10, pady=5)

        # VIS
        vis_container = ttk.Frame(img_frame)
        vis_container.pack(side="left", padx=(0, 5))
        ttk.Label(vis_container, text="📷 Visible (可见光)",
                  style="Title.TLabel").pack()
        self.vis_canvas = tk.Canvas(vis_container,
                                     width=self.img_display_size[0],
                                     height=self.img_display_size[1],
                                     bg="#1a1a1a", highlightthickness=1,
                                     highlightbackground="#444")
        self.vis_canvas.pack()

        # IR
        ir_container = ttk.Frame(img_frame)
        ir_container.pack(side="left", padx=(5, 0))
        ttk.Label(ir_container, text="🌡 Infrared (红外)",
                  style="Title.TLabel").pack()
        self.ir_canvas = tk.Canvas(ir_container,
                                    width=self.img_display_size[0],
                                    height=self.img_display_size[1],
                                    bg="#1a1a1a", highlightthickness=1,
                                    highlightbackground="#444")
        self.ir_canvas.pack()

        # ═══ 图像信息 ═══
        self.img_info_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.img_info_var,
                  style="Status.TLabel").pack(fill="x", padx=10)

        # ═══ 标注字段区域 ═══
        fields_outer = ttk.Frame(self.root)
        fields_outer.pack(fill="x", padx=10, pady=5)

        ttk.Label(fields_outer, text="━━━ 标注选项 ━━━",
                  style="Title.TLabel").pack(anchor="w")

        fields_frame = ttk.Frame(fields_outer)
        fields_frame.pack(fill="x")

        # 分两列布局
        left_col = ttk.Frame(fields_frame)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right_col = ttk.Frame(fields_frame)
        right_col.pack(side="left", fill="both", expand=True)

        field_names = list(FIELD_DEFS.keys())
        mid = (len(field_names) + 1) // 2

        for i, fname in enumerate(field_names):
            fdef = FIELD_DEFS[fname]
            parent = left_col if i < mid else right_col

            row_frame = ttk.Frame(parent)
            row_frame.pack(fill="x", pady=2)

            lbl = ttk.Label(row_frame, text=fdef["label"], width=32, anchor="w")
            lbl.pack(side="left")

            var = tk.StringVar(value=fdef["default"])
            var.trace_add("write", lambda *a: self._update_prompt_preview())

            combo = ttk.Combobox(row_frame, textvariable=var,
                                  values=fdef["options"],
                                  state="readonly", width=28)
            combo.pack(side="left", padx=5)

            self.field_vars[fname] = var
            self.field_widgets[fname] = (lbl, combo, row_frame)

        # task 变化时更新可见字段
        self.field_vars["task"].trace_add("write", lambda *a: self._on_task_change())

        # ═══ Prompt 预览 ═══
        preview_frame = ttk.Frame(self.root)
        preview_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(preview_frame, text="━━━ 生成的 CLIP Prompt 预览 ━━━",
                  style="Title.TLabel").pack(anchor="w")

        self.prompt_text = tk.Text(preview_frame, height=4, wrap="word",
                                    bg="#1a1a2e", fg="#00ff88",
                                    font=("Consolas", 10),
                                    relief="flat", padx=10, pady=8,
                                    insertbackground="#00ff88")
        self.prompt_text.pack(fill="x")
        self.prompt_text.configure(state="disabled")

        # ═══ 按钮区 ═══
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10, pady=(5, 10))

        ttk.Button(btn_frame, text="💾 保存当前 & 下一张 (Enter)",
                   command=self._save_and_next,
                   style="Accent.TButton").pack(side="left", padx=5)
        ttk.Button(btn_frame, text="💾 仅保存当前",
                   command=self._save_current).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📋 复制 Prompt",
                   command=self._copy_prompt).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📁 导出 CSV",
                   command=self._save_csv).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="📁 导出全部 .txt",
                   command=self._export_all_txt).pack(side="left", padx=5)

        # 标注状态
        self.status_var = tk.StringVar()
        ttk.Label(btn_frame, textvariable=self.status_var,
                  style="Status.TLabel").pack(side="right", padx=10)

        # ═══ 底部状态栏 ═══
        status_bar = ttk.Frame(self.root)
        status_bar.pack(fill="x", side="bottom")
        self.bottom_status = tk.StringVar(
            value="  快捷键: ← 上一张 | → 下一张 | Enter 保存并下一张 | Ctrl+S 导出CSV | Esc 退出"
        )
        ttk.Label(status_bar, textvariable=self.bottom_status,
                  style="Status.TLabel").pack(fill="x", ipady=3)

        # 初始更新
        self._on_task_change()

    # ──────────────────────────────────────────────────────
    #  任务切换 → 显示/隐藏字段
    # ──────────────────────────────────────────────────────

    def _on_task_change(self):
        task = self.field_vars["task"].get()
        visible_groups = TASK_GROUPS.get(task, {"common"})

        for fname, fdef in FIELD_DEFS.items():
            lbl, combo, row_frame = self.field_widgets[fname]
            if fdef["group"] in visible_groups:
                row_frame.pack(fill="x", pady=2)
            else:
                row_frame.pack_forget()

        # 更新 variant 选项
        variant_options = ["base"]
        if task == "detection":
            variant_options = ["base", "hard_case"]
        elif task == "segmentation":
            variant_options = ["base", "boundary_focus"]
        elif task == "detection_segmentation":
            variant_options = ["base"]

        self.field_widgets["variant"][1].configure(values=variant_options)
        if self.field_vars["variant"].get() not in variant_options:
            self.field_vars["variant"].set("base")

        self._update_prompt_preview()

    # ──────────────────────────────────────────────────────
    #  Prompt 预览更新
    # ──────────────────────────────────────────────────────

    def _update_prompt_preview(self):
        try:
            kwargs = {k: v.get() for k, v in self.field_vars.items()}
            prompt = self.builder.build(**kwargs)
        except Exception as e:
            prompt = f"[生成失败: {e}]"

        self.prompt_text.configure(state="normal")
        self.prompt_text.delete("1.0", "end")
        self.prompt_text.insert("1.0", prompt)
        self.prompt_text.configure(state="disabled")

    # ──────────────────────────────────────────────────────
    #  图像显示
    # ──────────────────────────────────────────────────────

    def _show_current(self):
        if not self.pairs:
            return

        name, vis_path, ir_path = self.pairs[self.current_idx]

        # 加载图像
        try:
            vis_img = Image.open(vis_path)
            ir_img = Image.open(ir_path)
        except Exception as e:
            messagebox.showerror("图像加载失败", f"{name}: {e}")
            return

        orig_vis_size = vis_img.size
        orig_ir_size = ir_img.size

        # 缩放到显示尺寸 (保持比例)
        vis_display = self._resize_to_fit(vis_img, self.img_display_size)
        ir_display = self._resize_to_fit(ir_img, self.img_display_size)

        # 转为 tk 图像
        self._vis_photo = ImageTk.PhotoImage(vis_display)
        self._ir_photo = ImageTk.PhotoImage(ir_display)

        # 显示
        cw, ch = self.img_display_size
        self.vis_canvas.delete("all")
        self.vis_canvas.create_image(cw // 2, ch // 2,
                                      anchor="center", image=self._vis_photo)

        self.ir_canvas.delete("all")
        self.ir_canvas.create_image(cw // 2, ch // 2,
                                     anchor="center", image=self._ir_photo)

        # 图像信息
        self.img_info_var.set(
            f"  {name}  |  VIS: {orig_vis_size[0]}×{orig_vis_size[1]}  "
            f"IR: {orig_ir_size[0]}×{orig_ir_size[1]}  |  "
            f"文件: {os.path.basename(vis_path)} / {os.path.basename(ir_path)}"
        )

        # 进度
        n_total = len(self.pairs)
        n_done = len(self.annotations)
        self.progress_var.set(
            f"📌 [{self.current_idx + 1}/{n_total}]  {name}  "
            f"(已标注 {n_done}/{n_total}  "
            f"{n_done / n_total * 100:.1f}%)"
        )

        # 加载已有标注 (如果有)
        if name in self.annotations:
            self._load_annotation(name)
            self.status_var.set("✅ 已有标注")
        else:
            self.status_var.set("⬜ 未标注")

        self._update_prompt_preview()
        self.jump_var.set(str(self.current_idx + 1))

    def _resize_to_fit(self, img, target_size):
        tw, th = target_size
        iw, ih = img.size
        ratio = min(tw / iw, th / ih)
        new_w = int(iw * ratio)
        new_h = int(ih * ratio)
        return img.resize((new_w, new_h), Image.LANCZOS)

    # ──────────────────────────────────────────────────────
    #  标注加载/保存
    # ──────────────────────────────────────────────────────

    def _load_annotation(self, name):
        ann = self.annotations[name]
        for fname, var in self.field_vars.items():
            if fname in ann and ann[fname]:
                try:
                    var.set(ann[fname])
                except Exception:
                    pass

    def _get_current_annotation(self):
        ann = {}
        for fname, var in self.field_vars.items():
            ann[fname] = var.get()
        return ann

    def _get_current_prompt(self):
        kwargs = {k: v.get() for k, v in self.field_vars.items()}
        return self.builder.build(**kwargs)

    def _save_current(self):
        name = self.pairs[self.current_idx][0]
        ann = self._get_current_annotation()
        prompt = self._get_current_prompt()

        # 保存到内存
        self.annotations[name] = ann

        # 保存 .txt
        txt_path = os.path.join(self.output_dir, f"{name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        # 自动保存 CSV
        self._save_csv(silent=True)

        self.status_var.set(f"✅ 已保存 {name}")
        self._show_current()  # 刷新状态

    def _save_and_next(self):
        self._save_current()
        self._go_next()

    def _save_csv(self, silent=False):
        fieldnames = ["name"] + list(FIELD_DEFS.keys()) + ["prompt"]
        try:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for name, _, _ in self.pairs:
                    if name in self.annotations:
                        row = {"name": name}
                        row.update(self.annotations[name])
                        # 生成 prompt 写入 CSV
                        try:
                            kwargs = {k: v for k, v in self.annotations[name].items()}
                            row["prompt"] = self.builder.build(**kwargs)
                        except Exception:
                            row["prompt"] = ""
                        writer.writerow(row)

            if not silent:
                messagebox.showinfo("导出成功",
                                     f"CSV 已保存: {self.csv_path}\n"
                                     f"共 {len(self.annotations)} 条标注")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def _export_all_txt(self):
        count = 0
        for name, _, _ in self.pairs:
            if name in self.annotations:
                try:
                    kwargs = {k: v for k, v in self.annotations[name].items()}
                    prompt = self.builder.build(**kwargs)
                    txt_path = os.path.join(self.output_dir, f"{name}.txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(prompt)
                    count += 1
                except Exception:
                    pass

        messagebox.showinfo("导出完成",
                             f"已导出 {count} 个 .txt 到 {self.output_dir}")

    # ──────────────────────────────────────────────────────
    #  导航
    # ──────────────────────────────────────────────────────

    def _go_prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()

    def _go_next(self):
        if self.current_idx < len(self.pairs) - 1:
            self.current_idx += 1
            self._show_current()

    def _jump_to(self):
        try:
            idx = int(self.jump_var.get()) - 1
            if 0 <= idx < len(self.pairs):
                self.current_idx = idx
                self._show_current()
            else:
                messagebox.showwarning("范围错误",
                                        f"请输入 1 ~ {len(self.pairs)}")
        except ValueError:
            # 尝试按名称跳转
            target = self.jump_var.get().strip()
            for i, (name, _, _) in enumerate(self.pairs):
                if name == target:
                    self.current_idx = i
                    self._show_current()
                    return
            messagebox.showwarning("未找到", f"图像 '{target}' 不存在")

    def _copy_prompt(self):
        prompt = self._get_current_prompt()
        self.root.clipboard_clear()
        self.root.clipboard_append(prompt)
        self.status_var.set("📋 Prompt 已复制")

    def _on_close(self):
        # 退出前自动保存
        if self.annotations:
            self._save_csv(silent=True)
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════
#  启动
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IR-VIS 图像对标注工具"
    )
    parser.add_argument("--ir_dir", type=str, default=r"D:\paper\datasets\text\M3FD_det\ir",
                        help="红外图像文件夹")
    parser.add_argument("--vis_dir", type=str, default=r"D:\paper\datasets\text\M3FD_det\vi",
                        help="可见光图像文件夹")
    parser.add_argument("--output_dir", type=str, default=r"D:\paper\datasets\text\M3FD_det\text",
                        help=".txt prompt 输出目录")
    parser.add_argument("--csv_path", type=str, default="annotations.csv",
                        help="标注 CSV 路径")
    args = parser.parse_args()

    # 检查路径
    if not os.path.isdir(args.ir_dir):
        print(f"❌ IR 目录不存在: {args.ir_dir}")
        print(f"   请用 --ir_dir 指定正确路径")
        sys.exit(1)
    if not os.path.isdir(args.vis_dir):
        print(f"❌ VIS 目录不存在: {args.vis_dir}")
        print(f"   请用 --vis_dir 指定正确路径")
        sys.exit(1)

    print(f"\n{'═' * 60}")
    print(f"  IR-VIS 图像对标注工具")
    print(f"{'═' * 60}")
    print(f"  IR 目录:   {args.ir_dir}")
    print(f"  VIS 目录:  {args.vis_dir}")
    print(f"  输出目录:  {args.output_dir}")
    print(f"  CSV 路径:  {args.csv_path}")
    print(f"{'═' * 60}\n")

    root = tk.Tk()

    # 窗口大小
    root.geometry("920x880")
    root.minsize(900, 850)

    app = AnnotationTool(
        root=root,
        ir_dir=args.ir_dir,
        vis_dir=args.vis_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
    )

    root.mainloop()


if __name__ == "__main__":
    main()

