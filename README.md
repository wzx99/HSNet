# Hierarchical Semantics-fused Scene Text Detection
层级语义融合的场景文本检测代码实现，基于[DBNet](https://github.com/MhLiao/DB/tree/master)

## 环境配置

参考[DBNet](https://github.com/MhLiao/DB/tree/master)配置环境

```bash
pip install -r requirements.txt
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

cd assets/ops/dcn/
python setup.py build_ext --inplace
```

## 数据集配置

参考[DBNet](https://github.com/MhLiao/DB/tree/master)配置数据集，针对total-text数据集使用tools/Bezier_generator_totaltext.py调整标签格式

## 预训练模型
```bash
sh pretrain.sh
```
## 微调模型
```bash
sh train.sh
```
