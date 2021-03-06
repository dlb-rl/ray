#!/usr/bin/env bash

ray_version=""
commit=""
ray_branch=""

usage() {
    echo "Start one microbenchmark trial."
}

for i in "$@"
do
case "$i" in
    --ray-version=*)
    ray_version="${i#*=}"

    ;;
    --commit=*)
    commit="${i#*=}"
    ;;
    --ray-branch=*)
    ray_branch="${i#*=}"
    ;;
    --workload=*)
    workload="${i#*=}"
    ;;
    --help)
    usage
    exit
    ;;
    *)
    echo "unknown arg, $2"
    exit 1
    ;;
esac
done

if [ -z "$ray_version" ] || [ -z "$commit" ] || [ -z "$ray_branch" ]
then
    echo "Provide --ray-version, --commit, and --ray-branch"
    exit 1
fi


echo "version: $ray_version"
echo "commit: $commit"
echo "branch: $ray_branch"
echo "workload: $workload"

wheel="https://s3-us-west-2.amazonaws.com/ray-wheels/$ray_branch/$commit/ray-$ray_version-cp36-cp36m-manylinux1_x86_64.whl"


echo set-window-option -g mouse on > ~/.tmux.conf
echo 'termcapinfo xterm* ti@:te@' > ~/.screenrc
set -x
pip uninstall -y -q ray
pip install --upgrade pip
pip install -U "$wheel"
set +x

unset RAY_ADDRESS
OMP_NUM_THREADS=64 ray microbenchmark
