# SPDX-FileCopyrightText: Copyright 2026 chjnett
# SPDX-License-Identifier: Apache-2.0

# 제출용 이미지. 저장소가 제공하는 예제(container/Dockerfile)와 같은 격리
# 조건을 지키되 기반 이미지만 바꿨다.
#
# 예제는 python:3.11-alpine을 고정한다. 우리 라우터는 numpy를 쓰는데 musl
# 환경에는 미리 만들어진 wheel이 없어 소스 빌드가 필요하다. Debian slim에는
# manylinux wheel이 있어 빌드 도구 없이 설치된다. RUNTIME.md는 예제 기반
# 이미지를 강제하지 않는다.
#
# 기반 이미지 출처와 라이선스는 container/ROUTER_BASE_IMAGE.md에 기록한다.
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

ARG SOURCE_MANIFEST_SHA256=unbound
LABEL org.opencontainers.image.source-manifest-sha256="${SOURCE_MANIFEST_SHA256}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/router \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/tmp

# 런타임 의존성은 numpy 하나뿐이다 (BSD-3-Clause).
COPY container/router-requirements.txt /tmp/router-requirements.txt
RUN python3 -m pip install --require-hashes --no-deps -r /tmp/router-requirements.txt \
 && python3 -m pip uninstall --yes pip setuptools wheel \
 && rm -rf /tmp/router-requirements.txt /root/.cache

COPY --chown=65532:65532 src /opt/router/
COPY --chown=65532:65532 container/router_entrypoint.py /opt/router/entrypoint.py

WORKDIR /opt/router
USER 65532:65532

ENTRYPOINT ["python3", "/opt/router/entrypoint.py"]
