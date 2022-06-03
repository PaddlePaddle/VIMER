### 安装PaddlePaddle
本代码库基于`PaddlePaddle develop`版, 可参考[paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick)进行环境配置，或者使用pip进行安装，根据CUDA版本不同，可自行选择对应适配版本的PaddlePaddle代码库:

```bash
# We only support the evaluation on GPU by using PaddlePaddle, the installation command follows:
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### 安装PaddleNlp
PaddleNlp代码可参考(https://github.com/PaddlePaddle/PaddleNLP) 或者使用pip进行安装:

```bash
pip install paddlenlp
```

* 环境要求
```bash
python 3.6+
numpy
Pillow
paddlenlp>=2.2.3
cuda>=10.1
cudnn>=7.6.4
gcc>=8.2
```

* 安装要求
ViSTA的依赖库已在`requirements.txt`中列出，你可以使用以下命令行进行依赖库安装：
```
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```
