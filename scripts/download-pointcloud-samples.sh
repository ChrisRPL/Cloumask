#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${1:-${ROOT_DIR}/tmp/pointclouds}"

mkdir -p "${OUT_DIR}"

download_file() {
	local url="$1"
	local name="$2"
	local path="${OUT_DIR}/${name}"

	if [[ -f "${path}" ]]; then
		echo "exists ${path}"
		return 0
	fi

	echo "download ${name}"
	curl -fL "${url}" -o "${path}"
}

download_file \
	"https://raw.githubusercontent.com/PointCloudLibrary/data/master/tutorials/lamppost.pcd" \
	"lamppost.pcd"

download_file \
	"https://raw.githubusercontent.com/PointCloudLibrary/data/master/tutorials/correspondence_grouping/milk.pcd" \
	"milk.pcd"

echo "ready ${OUT_DIR}"
