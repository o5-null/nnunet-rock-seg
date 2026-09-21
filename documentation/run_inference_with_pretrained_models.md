# How to run inference with pretrained models

For the general inference workflow, including exporting and installing your own trained models, see:

- [Run inference](how-to/run-inference.md)

**Important:** Pretrained weights from nnU-Net v1 are NOT compatible with V2. You will need to retrain with the new
version. But honestly, you already have a fully trained model with which you can run inference (in v1), so
just continue using that!

Not yet available for V2 :-(
If you wish to run inference with pretrained models, check out the old nnU-Net for now. We are working on this full steam!

## Cross-dataset inference（跨数据集推理，本 fork 特有）

本 fork 额外提供了一个入口：用 **A 数据集训练好的 checkpoint**，对 **B 数据集（目标数据集）的全量图像**
跑滑窗推理并评估，全程零训练。与 `nnUNetv2_predict` 的差别在于它会读目标数据集的
`dataset.json` / `splits_final.json`，自动挑选 case、定位模型目录（`-p` 可省略），并直接产出
`summary.json`。

- [Cross-dataset inference 使用说明](how-to/cross-dataset-inference.md)
