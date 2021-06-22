# pytorch_mmap

pytorch_mmap is a lightweight tool to save and load pytorch CPU model in the form of [*mmap*](https://en.wikipedia.org/wiki/Mmap). 

When a model is loaded in the form of *mmap*, model parameters are actually still located on the disk, they are only mapped, not copied, into memory. Therefore, loading a *mmap* model form takes up almost no memory space.



## Installation

```
git clone https://github.com/JiayiFeng/pytorch_mmap.git
cd ./pytorch_mmap
python ./setup.py install
```



## Usage

Save a pytorch model in *mmap* form:

```python
import pytorch_mmap

# 'model' is the pytorch model to be save.
# "model_dir" is the directory to save the mmap model. It will be created if it does not exist.
pytorch_mmap.save(model, "model_dir")
```

Load a *mmap* pytorch model:

```python
import pytorch_mmap

# "model_dir" is the directory that you saved mmap model in.
model = pytorch_mmap.load("model_dir")
```

