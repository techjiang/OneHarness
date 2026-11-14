# Release 产物

平台不给本仓库的 tag / Release 写权限，所以发布物直接放在仓库里，跟着代码一起推。
每个版本的产物是一个目录，目录名即 tag 名。

```
release/
  build.sh          # 就地打包：读 pyproject 的版本号，产物落到 release/v<version>/
  v1.7.0/
    oneharness-1.7.0-py3-none-any.whl
    oneharness-1.7.0.tar.gz
    SHA256SUMS
    RELEASE.md      # 发布说明，等同 Release 正文
```

## 安装

```console
$ pip install https://cnb.cool/sleepyheade/OneHarness/-/raw/feat/oneharness-mvp/release/v1.7.0/oneharness-1.7.0-py3-none-any.whl
```

或者拿到本地装：

```console
$ pip install release/v1.7.0/oneharness-1.7.0-py3-none-any.whl
$ oneharness --version
oneharness 1.7.0
```

校验：

```console
$ cd release/v1.7.0 && sha256sum -c SHA256SUMS
```

## 重新打包

```console
$ bash release/build.sh
```

不引 `build` 包，直接调 `setuptools.build_meta`；纯标准库 + setuptools，和项目本身一样不欠依赖。
