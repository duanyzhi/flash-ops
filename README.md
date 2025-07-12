# Flash Ops Toy

## build
docker pull base image
```shell
docker pull pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
```

build wheel
```shell
python setup.py bdist_wheel
```