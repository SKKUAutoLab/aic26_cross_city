<h1 align='center'>
    Rethinking Pre-Training and Augmentation<br>
    for Zero-Shot Cross-City Object Detection
</h1>


<!-- MARK: News -->

## 🎉 NEWS

- [2026.07.29] 💻 Part of our code is being released!
- [2026.07.24] 📄 Our paper is under review at ECCVW.


<!-- MARK: Abstract -->

## 📝 Abstract

Real-world deployment of traffic surveillance systems is bottlenecked by geographic domain shift, in which models trained in one city underperform when applied to an unseen target city. Conventional domain adaptation relies on hyperparameter-sensitive architectures or direct profiling of target data. Both are fundamentally precluded in privacy-conscious ecosystems that require completely blind training and evaluation loops. In this setting, we explore the effects of pre-training and augmentation on addressing the domain shift problem. Specifically, we propose a new modular training pipeline for object detection structured around two core orthogonal pillars: (1) a multi-dataset pre-training strategy featuring a class-agnostic objectness distillation to decouple structural vehicle geometry from semantic taxonomies, and (2) a domain-resilient augmentation stream featuring a novel Grayworld transformation that forces global attention headers to strip volatile chromatic shortcuts in favor of robust shape priors. When evaluated with the real-time transformer-based detector RF-DETR, our framework bridges cross-city distribution gaps while using limited GPU memory (16GB). Our optimized variants, RF-DETR-HR and RF-DETR-Grayworld, deliver substantial empirical +24.29 gains over the baseline, achieving 1st place (mAP 47.53) on the AI City Challenge Track 6 leaderboard.


<!-- MARK: Install -->

## 📥 Install

To test our solution, please clone this repository:

```bash
git clone https://github.com/SKKUAutoLab/aic26_cross_city
cd aic26_cross_city
poetry install
```

<details>
  <summary>Directory Structure</summary>

  ```text
  aic26_cross_city/                     # Project root.
  |__ data/                             # Custom pre-training datasets.
  |__ docker/                           # Docker files for experiments.
  |__ run/                              # Local training/evaluation artifacts.
  |__ src/
  |   |__ trainer_object_detection/     # Main project source code.
  |__ tools/                            # Some useful scripts.
  |__ .dockerignore
  |__ .gitignore
  |__ pyproject.toml
  |__ README.md
  ```
</details>


<!-- MARK: References -->

## 📚 References

Thanks to the developers and contributors of the following open-source repositories, whose invaluable work has greatly inspire our project:

**Datasets**:
- [TSBOW](https://github.com/SKKUAutoLab/TSBOW): A traffic surveillance benchmark for occluded vehicles under various weather conditions.
- [TrafficCAM](https://math-ml-x.github.io/TrafficCAM/): A versatile dataset for traffic flow segmentation.
- [FishEye8K](https://github.com/MoyoG/FishEye8K): A benchmark and dataset for fisheye camera object detection.
- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset): A drone-based detection and tracking dataset, including both image/video, and annotations.
- [MOT20](https://github.com/JonathonLuiten/TrackEval/tree/master): A benchmark for single-camera multiple target tracking.

**Github Repo**:
- [Hafnia Trainer Package](https://github.com/milestone-hafnia/trainer-object-detection): An object detection trainer package for Hafnia Training-as-a-Service.
- [RF-DETR](https://github.com/roboflow/rf-detr): A real-time object detection and segmentation model architecture developed by Roboflow.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics): Detection models for training and real-time inferencing.


<!-- MARK: Git Stats -->

<!-- ![Star History Chart](https://api.star-history.com/svg?repos=SKKUAutoLab/aic26_cross_city&type=Date) -->

<!-- ![](https://img.shields.io/github/downloads/SKKUAutoLab/aic26_cross_city/total.svg?style=for-the-badge) -->

<!-- <div style="position: relative; display: inline-block;">
  <img src="https://api.star-history.com/svg?repos=SKKUAutoLab/aic26_cross_city&type=Date" alt="Star History Graph">
  <img src="icons/TSBOW_icon_white_BG.png" style="position: absolute; top: 10px; left: 310px; width: 30px; height: 30px;" alt="Custom Avatar">
</div> -->

<div align="center"><a href="#top">🔝 Back to Top</a></div>
