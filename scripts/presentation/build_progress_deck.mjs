import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  layers,
  panel,
  text,
  shape,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
} from "file:///C:/Users/gfddmw/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "../..");
const outDir = path.join(root, "documents");
const previewDir = path.join(outDir, "ppt_preview");
const pptxPath = path.join(outDir, "Lightweight-SRT_项目进度简洁汇报.pptx");

const W = 1920;
const H = 1080;

const C = {
  ink: "#122029",
  muted: "#5B6872",
  line: "#DCE5EA",
  paper: "#F7FAFB",
  white: "#FFFFFF",
  teal: "#16A085",
  blue: "#2F6FBB",
  gold: "#D39A2F",
  red: "#C44E45",
  dark: "#0E1D25",
  dark2: "#15303B",
};

const font = "Microsoft YaHei";

function T(value, opts = {}) {
  return text(value, {
    height: hug,
    ...opts,
    style: {
      fontFamily: font,
      color: C.ink,
      ...(opts.style || {}),
    },
  });
}

function baseSlide(presentation, title, subtitle, body, footer = "Lightweight-SRT | 阶段汇报") {
  const slide = presentation.slides.add();
  slide.compose(
    panel(
      { name: "slide-bg", width: fill, height: fill, fill: C.paper, padding: { x: 92, y: 62 } },
      column({ name: "root", width: fill, height: fill, gap: 34 }, [
        column({ name: "title-stack", width: fill, height: hug, gap: 12 }, [
          T(title, {
            name: "slide-title",
            width: fill,
            style: { fontSize: 54, bold: true, color: C.ink },
          }),
          subtitle
            ? T(subtitle, {
                name: "slide-subtitle",
                width: wrap(1420),
                style: { fontSize: 25, color: C.muted },
              })
            : rule({ name: "title-rule", width: fixed(220), stroke: C.teal, weight: 5 }),
        ]),
        body,
        row({ name: "footer", width: fill, height: hug, justify: "between", align: "center" }, [
          T(footer, { name: "footer-text", width: wrap(900), style: { fontSize: 15, color: "#8A99A3" } }),
          T("2026.05", { name: "footer-date", width: hug, style: { fontSize: 15, color: "#8A99A3" } }),
        ]),
      ]),
    ),
    { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
  );
  return slide;
}

function bullet(label, body, accent = C.teal) {
  return row({ name: `bullet-${label}`, width: fill, height: hug, gap: 16, align: "start" }, [
    shape({ name: `dot-${label}`, geometry: "ellipse", width: fixed(12), height: fixed(12), fill: accent }),
    column({ name: `bullet-copy-${label}`, width: fill, height: hug, gap: 6 }, [
      T(label, { name: `bullet-label-${label}`, width: fill, style: { fontSize: 28, bold: true, color: C.ink } }),
      T(body, { name: `bullet-body-${label}`, width: fill, style: { fontSize: 22, color: C.muted } }),
    ]),
  ]);
}

function metric(value, label, note, color) {
  return column({ name: `metric-${label}`, width: fill, height: hug, gap: 10 }, [
    T(value, { name: `metric-value-${label}`, width: fill, style: { fontSize: 70, bold: true, color } }),
    T(label, { name: `metric-label-${label}`, width: fill, style: { fontSize: 26, bold: true, color: C.ink } }),
    T(note, { name: `metric-note-${label}`, width: fill, style: { fontSize: 19, color: C.muted } }),
  ]);
}

function smallTag(label, color = C.teal) {
  return panel(
    { name: `tag-${label}`, padding: { x: 16, y: 8 }, fill: color, borderRadius: "rounded-full" },
    T(label, { name: `tag-text-${label}`, width: hug, style: { fontSize: 17, bold: true, color: C.white } }),
  );
}

function makeDeck() {
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  // 1. Cover
  {
    const slide = presentation.slides.add();
    slide.compose(
      panel({ name: "cover-bg", width: fill, height: fill, fill: C.dark, padding: { x: 96, y: 72 } },
        grid({ name: "cover-grid", width: fill, height: fill, columns: [fr(1.3), fr(0.7)], columnGap: 56 }, [
          column({ name: "cover-copy", width: fill, height: fill, justify: "center", gap: 26 }, [
            T("Lightweight-SRT", {
              name: "cover-title",
              width: fill,
              style: { fontSize: 92, bold: true, color: C.white },
            }),
            T("基于师生模型协同的轻量级手语识别阶段汇报", {
              name: "cover-subtitle",
              width: wrap(1180),
              style: { fontSize: 35, color: "#D2E4E5" },
            }),
            rule({ name: "cover-rule", width: fixed(420), stroke: C.teal, weight: 6 }),
            T("从骨骼点输入到 Android 实时识别，当前最佳 Top1 已突破 38%。", {
              name: "cover-promise",
              width: wrap(1030),
              style: { fontSize: 26, color: "#A7BDC4" },
            }),
          ]),
          column({ name: "cover-number", width: fill, height: fill, justify: "center", align: "end", gap: 10 }, [
            T("38.28%", { name: "cover-number-main", width: hug, style: { fontSize: 132, bold: true, color: C.gold } }),
            T("Top1 Accuracy", { name: "cover-number-label", width: hug, style: { fontSize: 26, color: "#D2E4E5" } }),
            T("WLASL2000 / 多流精炼版", { name: "cover-number-note", width: hug, style: { fontSize: 19, color: "#91A8AF" } }),
          ]),
        ]),
      ),
      { frame: { left: 0, top: 0, width: W, height: H }, baseUnit: 8 },
    );
  }

  // 2. Research target
  baseSlide(
    presentation,
    "研究目标：把高精度识别压到移动端可实时运行",
    "核心矛盾不是只追求准确率，而是在精度、速度和部署复杂度之间取得可用平衡。",
    grid({ name: "goal-grid", width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 70, alignItems: "center" }, [
      column({ name: "goal-left", width: fill, height: hug, gap: 30 }, [
        bullet("教师模型", "I3D 使用 RGB 视频作为高容量知识来源，提供蒸馏信号。", C.blue),
        bullet("学生模型", "ST-GCN 只使用手部骨骼点，参数和计算量显著降低。", C.teal),
        bullet("落地目标", "通过 PyTorch Lite + MediaPipe，在 Android 端完成实时识别闭环。", C.gold),
      ]),
      column({ name: "goal-metrics", width: fill, height: hug, gap: 30 }, [
        metric("91.77x", "推理提速", "相对 I3D CPU 延迟的量级收益", C.teal),
        metric("49.59%", "模型体积压缩", "INT8 与移动端优化链路结果", C.blue),
      ]),
    ]),
  );

  // 3. Technical route
  baseSlide(
    presentation,
    "技术路线：教师蒸馏 + 多流骨骼建模 + 移动端推理",
    "当前系统已经形成从数据预处理、训练优化到端侧推理的完整工程链路。",
    column({ name: "route-body", width: fill, height: grow(1), justify: "center", gap: 34 }, [
      grid({ name: "route-grid", width: fill, height: hug, columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 28 }, [
        panel({ name: "route-1", padding: { x: 26, y: 24 }, fill: C.white },
          column({ width: fill, height: hug, gap: 10 }, [
            T("01 数据", { name: "route-1-title", width: fill, style: { fontSize: 29, bold: true, color: C.blue } }),
            T("WLASL2000 视频\n骨骼点 / logits / feature", { name: "route-1-body", width: fill, style: { fontSize: 22, color: C.muted } }),
          ])),
        panel({ name: "route-2", padding: { x: 26, y: 24 }, fill: C.white },
          column({ width: fill, height: hug, gap: 10 }, [
            T("02 蒸馏", { name: "route-2-title", width: fill, style: { fontSize: 29, bold: true, color: C.teal } }),
            T("I3D 教师输出\nLogits KD + Hint Loss", { name: "route-2-body", width: fill, style: { fontSize: 22, color: C.muted } }),
          ])),
        panel({ name: "route-3", padding: { x: 26, y: 24 }, fill: C.white },
          column({ width: fill, height: hug, gap: 10 }, [
            T("03 多流", { name: "route-3-title", width: fill, style: { fontSize: 29, bold: true, color: C.gold } }),
            T("Joint / Bone / Motion\n融合互补特征", { name: "route-3-body", width: fill, style: { fontSize: 22, color: C.muted } }),
          ])),
        panel({ name: "route-4", padding: { x: 26, y: 24 }, fill: C.white },
          column({ width: fill, height: hug, gap: 10 }, [
            T("04 部署", { name: "route-4-title", width: fill, style: { fontSize: 29, bold: true, color: C.red } }),
            T("TorchScript Lite\nAndroid 实时识别", { name: "route-4-body", width: fill, style: { fontSize: 22, color: C.muted } }),
          ])),
      ]),
      row({ name: "route-tags", width: fill, height: hug, gap: 18, justify: "center" }, [
        smallTag("标签强制对齐", C.blue),
        smallTag("关闭错误归一化", C.teal),
        smallTag("结构化剪枝", C.gold),
        smallTag("滑动窗口推理", C.red),
      ]),
    ]),
  );

  // 4. Key breakthrough
  baseSlide(
    presentation,
    "关键突破：多流低性能不是模型上限，而是标签体系错位",
    "修复标签对齐后，多流模型从约 19% Top1 快速跃迁到 37%+，验证了骨骼多流特征的有效性。",
    grid({ name: "breakthrough-grid", width: fill, height: grow(1), columns: [fr(0.9), fr(1.1)], columnGap: 66, alignItems: "center" }, [
      column({ name: "problem-solution", width: fill, height: hug, gap: 24 }, [
        bullet("问题", "早期多流训练使用按 gloss 排序的 label_map，与教师模型 ID/action 标签不一致。", C.red),
        bullet("修复", "数据集直接读取 nslt_2000.json 的 action 索引，保证学生与教师标签对齐。", C.teal),
        bullet("附加优化", "保留全局空间信息，关闭错误归一化，并预计算 bone/motion 特征提升 IO。", C.blue),
      ]),
      chart({
        name: "accuracy-jump-chart",
        chartType: "bar",
        width: fill,
        height: fixed(500),
        config: {
          title: "多流训练关键节点 Top1",
          categories: ["初始多流", "单流 Fast-KD", "多流 aligned_v2"],
          series: [{ name: "Top1", values: [19.49, 28.24, 37.13] }],
        },
      }),
    ]),
  );

  // 5. Experiment result
  baseSlide(
    presentation,
    "实验结果：当前最佳模型达到 38.28% / 71.17%",
    "精炼阶段采用高起点权重 + 微量 KD/Hint 约束，绕开弱教师瓶颈，继续提升学生模型上限。",
    grid({ name: "result-grid", width: fill, height: grow(1), columns: [fr(0.78), fr(1.22)], columnGap: 70, alignItems: "center" }, [
      column({ name: "result-metrics", width: fill, height: hug, gap: 34 }, [
        metric("38.28%", "Top1 Accuracy", "精炼终极版 v4 记录值", C.gold),
        metric("71.17%", "Top5 Accuracy", "相似手势召回能力明显增强", C.teal),
      ]),
      column({ name: "result-table", width: fill, height: hug, gap: 22 }, [
        row({ name: "table-head", width: fill, height: hug, gap: 18 }, [
          T("版本", { name: "h1", width: fixed(220), style: { fontSize: 21, bold: true, color: C.ink } }),
          T("策略", { name: "h2", width: fixed(330), style: { fontSize: 21, bold: true, color: C.ink } }),
          T("Top1", { name: "h3", width: fixed(130), style: { fontSize: 21, bold: true, color: C.ink } }),
          T("Top5", { name: "h4", width: fixed(130), style: { fontSize: 21, bold: true, color: C.ink } }),
        ]),
        rule({ name: "table-rule-1", width: fill, stroke: C.line, weight: 2 }),
        ...[
          ["初始蒸馏版", "强引导 KD + Hint", "28.32%", "58.71%"],
          ["多流基准", "标签对齐 + 自主进化", "37.13%", "69.26%"],
          ["精炼终极版", "高起点权重 + 微量提纯", "38.28%", "71.17%"],
        ].map((r, i) =>
          row({ name: `table-row-${i}`, width: fill, height: hug, gap: 18, align: "center" }, [
            T(r[0], { name: `r${i}c0`, width: fixed(220), style: { fontSize: 21, bold: i === 2, color: i === 2 ? C.gold : C.ink } }),
            T(r[1], { name: `r${i}c1`, width: fixed(330), style: { fontSize: 20, color: C.muted } }),
            T(r[2], { name: `r${i}c2`, width: fixed(130), style: { fontSize: 23, bold: true, color: i === 2 ? C.gold : C.teal } }),
            T(r[3], { name: `r${i}c3`, width: fixed(130), style: { fontSize: 23, bold: true, color: i === 2 ? C.gold : C.teal } }),
          ]),
        ),
      ]),
    ]),
  );

  // 6. Lightweight deployment
  baseSlide(
    presentation,
    "轻量化部署：从可训练模型到可装进手机的模型",
    "学生模型的价值在于把识别能力转化为端侧可用的推理速度和体积。",
    grid({ name: "deploy-grid", width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 72, alignItems: "center" }, [
      chart({
        name: "latency-chart",
        chartType: "bar",
        width: fill,
        height: fixed(500),
        config: {
          title: "CPU 推理延迟对比 / ms",
          categories: ["I3D 教师", "ST-GCN FP32", "INT8 估计"],
          series: [{ name: "Latency", values: [937.94, 21.38, 10.22] }],
        },
      }),
      column({ name: "deploy-copy", width: fill, height: hug, gap: 30 }, [
        bullet("模型导出", "已具备 TorchScript / Lite Interpreter 端侧加载链路。", C.blue),
        bullet("量化优化", "报告中记录 INT8 优化后模型体积缩减约 49.59%。", C.teal),
        bullet("剪枝风险", "15% 剪枝模型已生成，但当前日志显示评估异常，需继续修复 channel_cfg / 权重加载链路。", C.red),
      ]),
    ]),
  );

  // 7. Android prototype
  baseSlide(
    presentation,
    "Android 原型：实时识别闭环已具备基础形态",
    "端侧工程已经包含相机输入、手部关键点检测、滑窗缓存、模型推理和 UI 展示。",
    grid({ name: "android-grid", width: fill, height: grow(1), columns: [fr(1), fr(1)], columnGap: 70, alignItems: "center" }, [
      column({ name: "android-flow", width: fill, height: hug, gap: 22 }, [
        bullet("CameraX", "实时视频帧采集与 ImageAnalysis 分析器。", C.blue),
        bullet("MediaPipe", "HandLandmarker 提取 21 点手部关键点。", C.teal),
        bullet("SignRecognitionEngine", "64 帧滑动窗口、缺失帧补偿、平滑和推理节流。", C.gold),
        bullet("PyTorch Lite", "加载 .ptl 学生模型并映射 WLASL class list。", C.red),
      ]),
      column({ name: "android-assets", width: fill, height: hug, gap: 26 }, [
        metric("64", "窗口帧数", "端侧按训练输入维度组织序列", C.teal),
        metric("0.3", "置信度阈值", "当前原型过滤低置信预测", C.blue),
      ]),
    ]),
  );

  // 8. Next steps
  baseSlide(
    presentation,
    "下一步：收尾识别模型，打通稳定端测，再扩展连续手语翻译",
    "当前最该投入的是工程收敛，而不是继续堆新功能。",
    grid({ name: "next-grid", width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 36, alignItems: "center" }, [
      column({ name: "next-1", width: fill, height: hug, gap: 18 }, [
        T("01", { name: "next-1-num", width: fill, style: { fontSize: 60, bold: true, color: C.teal } }),
        T("模型收尾", { name: "next-1-title", width: fill, style: { fontSize: 31, bold: true } }),
        T("确认最佳 checkpoint；修复剪枝后评估异常；导出最终端侧模型。", { name: "next-1-body", width: fill, style: { fontSize: 22, color: C.muted } }),
      ]),
      column({ name: "next-2", width: fill, height: hug, gap: 18 }, [
        T("02", { name: "next-2-num", width: fill, style: { fontSize: 60, bold: true, color: C.blue } }),
        T("Android 端测", { name: "next-2-title", width: fill, style: { fontSize: 31, bold: true } }),
        T("重跑 Gradle 构建；验证相机、模型加载、实时推理延迟和预测稳定性。", { name: "next-2-body", width: fill, style: { fontSize: 22, color: C.muted } }),
      ]),
      column({ name: "next-3", width: fill, height: hug, gap: 18 }, [
        T("03", { name: "next-3-num", width: fill, style: { fontSize: 60, bold: true, color: C.gold } }),
        T("CSLT 扩展", { name: "next-3-title", width: fill, style: { fontSize: 31, bold: true } }),
        T("基于已有识别模型，继续推进连续序列数据、Encoder-Decoder 和 BLEU 评估。", { name: "next-3-body", width: fill, style: { fontSize: 22, color: C.muted } }),
      ]),
    ]),
  );

  return presentation;
}

async function main() {
  const fs = await import("node:fs/promises");
  await fs.mkdir(outDir, { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });

  const presentation = makeDeck();
  const pptxBlob = await PresentationFile.exportPptx(presentation);
  await pptxBlob.save(pptxPath);

  const previewPaths = [];
  for (let i = 0; i < presentation.slides.count; i++) {
    const slide = presentation.slides.getItem(i);
    const png = await slide.export({ format: "png", width: W, height: H });
    const p = path.join(previewDir, `slide-${String(i + 1).padStart(2, "0")}.png`);
    await fs.writeFile(p, Buffer.from(await png.arrayBuffer()));
    previewPaths.push(p);
  }

  const layout = presentation.inspect({ format: "json" });
  await fs.writeFile(path.join(previewDir, "layout-inspect.json"), JSON.stringify(layout, null, 2), "utf8");

  console.log(JSON.stringify({ pptxPath, previewDir, previewPaths }, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
