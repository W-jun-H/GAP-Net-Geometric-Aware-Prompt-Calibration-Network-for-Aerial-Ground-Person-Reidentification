### GAP-Net: Geometric Aware Prompt Calibration Network for Aerial Gournd Person Reidentification
This repo is solely used for recording the project reproducing and modifications making. Not for any publication or commercial use.
### My Approach
In this study, a new framework is constructed, which can integrate geometric prior estimation and adaptive semantic prompt learning in a natural way, and the framework is named GAP-Net.The whole method is divided into three parts. 
1. The first part is the geometric prior extraction module, which can encode the deformation field into geometric embedding, but will not interfere with the original feature map.
2. The second part is the geometric awareness calibration module, which injects geometric prior into the calibration process, and the prompt has geometric awareness and semantic invariance.
3. The third part is the local feature refinement module, which can mine local features through bidirectional attention mechanism with the help of geometric awareness prompt.
Generally speaking, this study points out that in the process of cross-view person re-identification, the two types of information should be combined to achieve the purpose of robustness. The biggest advantage of this method is that it can unify the two technologies into a framework, which can provide a reliable path for the development of computer vision technology.
### Acknowledgement
This repo is based on [SeCap](https://github.com/wangshining681/SeCap-AGPReID)
