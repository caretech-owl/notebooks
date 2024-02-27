### Notebooks


## Notes

### Issues with ctransformers

#### Compiling ctransformers on cluster with custom GCC

```shell
LDFLAGS="-static-libstdc++" CC=/home/neum_al/env/bin/gcc pip install ctransformers --no-binary ctransformers --no-cache-dir
```