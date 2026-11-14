#!/usr/bin/env bash
# 就地打一个版本的包，产物落到 release/<version>/。
# 平台不给 tag / release 权限时，用这里的文件当发布物。
set -euo pipefail

cd "$(dirname "$0")/.."
version=$(python3 -c 'import tomllib,sys;print(tomllib.load(open("pyproject.toml","rb"))["project"]["version"])')
out="release/v${version}"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

echo "==> building ${version}"
python3 - <<PY
import setuptools.build_meta as bm
import os
print(bm.build_wheel("$tmp"))
print(bm.build_sdist("$tmp"))
PY

mkdir -p "$out"
rm -f "$out"/*.whl "$out"/*.tar.gz
mv "$tmp"/*.whl "$tmp"/*.tar.gz "$out"/
cp "docs/RELEASE-v${version}.md" "$out/RELEASE.md"
(cd "$out" && sha256sum ./*.whl ./*.tar.gz > SHA256SUMS)

rm -rf oneharness.egg-info build
echo "==> ${out}"
ls -l "$out"
