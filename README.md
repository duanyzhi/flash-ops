# Flash Ops Toy

## build
docker pull base image
```shell
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

build wheel
```shell
git submodule init
git submodule update

git submodule update --init --recursive

python setup.py bdist_wheel
```
