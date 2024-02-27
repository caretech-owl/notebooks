### Notebooks

## Notes

Put downloaded models into

```sh
<project_dir>/models
```

Cache should be located in

```sh
<project_dir>/cache
<project_dir>/cache/hub    # hugging face
<project_dir>/cache/lora   # lora training
<project_dir>/cache/models # model merge result
```

### Issues with ctransformers

#### Compiling ctransformers on cluster with custom GCC

```shell
LDFLAGS="-static-libstdc++" CC=/home/neum_al/env/bin/gcc pip install ctransformers --no-binary ctransformers --no-cache-dir
```