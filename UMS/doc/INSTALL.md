### Install PaddlePaddle
This code base needs to be executed on the `PaddlePaddle develop` version. You can find how to prepare the environment from this [paddlepaddle-quick](https://www.paddlepaddle.org.cn/install/quick) or use pip, depending on the CUDA version, you can choose the PaddlePaddle code base corresponding to the adapted version:

```bash
# We only support the evaluation on GPU by using PaddlePaddle, the installation command follows:
python -m pip install paddlepaddle-gpu==0.0.0.post102 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
```
### Install PaddleNlp
The installation command of PaddleNlp can find from this (https://github.com/PaddlePaddle/PaddleNLP) or use pip:

```bash
pip install paddlenlp
```

* Environment requirements
```bash
python 3.6+
numpy
Pillow
paddlenlp>=2.2.3
matplotlib
faiss-cpu
cuda>=10.1
cudnn>=7.6.4
gcc>=8.2
```

* Install requirements
ViSTA dependencies are listed in file `requirements.txt`, you can use the following command to install the dependencies.
```
pip3 install --upgrade -r requirements.txt -i https://mirror.baidu.com/pypi/simple
```
