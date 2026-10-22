# grace-sandbox

The container Grace runs code in (`run_code`, `services/grace_tools/code.py`). The image is
Python 3.12 slim (Debian bookworm) with numpy, pandas, matplotlib, scipy, sympy and openpyxl,
plus Node 22 and npm copied from the official node image. It runs as user `grace` (uid 1000).

Build it with Grace's `build_sandbox` tool, or by hand:

    docker build -t grace-sandbox docker/sandbox

`sandbox_status` reports the image as out of date when this Dockerfile has changed since the
build (the build labels the image with the Dockerfile's hash, and a hand build has no label).

## How every job runs

    docker run --rm --name grace-job-<id>
      --network none                       (bridge only with network=true and a reason)
      --memory 2g --memory-swap 2g --cpus 2 --pids-limit 256
      --read-only --tmpfs /tmp:rw,exec,nosuid,size=512m
      --cap-drop ALL --security-opt no-new-privileges --user 1000:1000
      --mount type=bind,source=<scratch>/job-<id>,target=/work --workdir /work
      grace-sandbox python|node|bash /work/_job.<ext>

The one mount is the job's own folder under the tools server's scratch dir
(`.runtime/grace-data/scratch/job-<id>`), holding copies of the input files. Nothing else of
the PC is mounted. The job is killed after its timeout (60 s default, 300 s max).
`/tmp` allows exec so that packages a networked job installs with `pip install --user` can load
their compiled parts. Job folders older than a day are deleted on the next run.

Versions at the first build (2026-10-22): Python 3.12.15, numpy 2.5.4, pandas 3.0.6,
matplotlib 3.11.2, scipy 1.18.1, sympy 1.14.0, openpyxl 3.1.5, Node 22.23.4, npm 10.9.9.
The pip packages are not pinned, so a rebuild picks up newer ones.
