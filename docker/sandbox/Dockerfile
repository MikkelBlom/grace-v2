# grace-sandbox: where Grace runs code (services/grace_tools/code.py, docs/tools/code.md).
# Every job is a fresh `docker run --rm` with no network, a read-only root, 2 GB / 2 CPUs and
# only its own scratch folder mounted at /work. This image only has to make that useful.
FROM node:22-bookworm-slim AS node

FROM python:3.12-slim-bookworm

# Node from the official image instead of Debian's (older, and Debian's npm pulls in hundreds
# of packages).
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

RUN pip install --no-cache-dir numpy pandas matplotlib scipy sympy openpyxl

# Every writable place points into /tmp or /work: the root filesystem is read-only at run time.
# Packages a networked job installs (pip install --user, npm install) land there and vanish
# with the job.
ENV HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg \
    PYTHONUSERBASE=/tmp/.local \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    npm_config_cache=/tmp/.npm \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --no-create-home --uid 1000 grace
USER grace
WORKDIR /work
