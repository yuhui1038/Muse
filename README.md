# Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control

<p align="center">
  📄 <a href="https://arxiv.org/abs/2601.03973">Paper</a> • 📊 <a href="https://huggingface.co/datasets/bolshyC/Muse">Dataset</a> • 🤖 <a href="https://huggingface.co/bolshyC/models">Model</a> • 📚 <a href="#citation">Citation</a>
</p>

This repository is the official repository for "Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control". In this repository, we provide the Muse model, training and inference scripts, pretrained checkpoints, and evaluation pipelines.

## News and Updates

* **2026.01.11 🔥**: We are excited to announce that all datasets and models are now fully open-sourced! 🎶 The complete training dataset (116k songs), pretrained model weights, training and evaluation code, and data pipeline are publicly available on Hugging Face.

## Installation

To set up the environment for Muse, you need to install the following dependencies:

1. **[Qwen3](https://github.com/QwenLM/Qwen3)**: Base language model framework
2. **[ms-swift](https://github.com/modelscope/ms-swift)**: Training framework for fine-tuning
3. **[MuCodec](https://github.com/tencent-ailab/MuCodec)**: Discrete audio tokenization codec

Please refer to the respective repositories for detailed installation instructions. Make sure all dependencies are properly installed before running the training or inference scripts.

## Model Architecture

<p align="center">
  <img src="assets/intro.jpg" width="800"/>
</p>

## Acknowledgments

We thank [Qwen3](https://github.com/QwenLM/Qwen3) for providing the base language model, [ms-swift](https://github.com/modelscope/ms-swift) for the training framework, and [MuCodec](https://github.com/tencent-ailab/MuCodec) for discrete audio tokenization.

## Citation

If you find our work useful, please cite our paper:

```bibtex
@article{jiang2026muse,
  title={Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control},
  author={Jiang, Changhao and Chen, Jiahao and Xiang, Zhenghao and Yang, Zhixiong and Wang, Hanchen and Zhuang, Jiabao and Che, Xinmeng and Sun, Jiajun and Li, Hui and Cao, Yifei and others},
  journal={arXiv preprint arXiv:2601.03973},
  year={2026}
}
```
