#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="${SCRIPT_DIR}/unitree_actuator_sdk"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAX_JOBS="${MAX_JOBS:-2}"

for command_name in cmake "${PYTHON_BIN}"; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "${command_name}" >&2
        exit 1
    fi
done

python_soabi="$("${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_config_var("SOABI"))')"
if [[ -z "${python_soabi}" || "${python_soabi}" == "None" ]]; then
    printf 'Could not determine the Python SOABI for %s\n' "${PYTHON_BIN}" >&2
    exit 1
fi
BUILD_DIR="${UNITREE_SDK_BUILD_DIR:-${SDK_DIR}/build-${python_soabi}}"

case "$(uname -m)" in
    x86_64|aarch64)
        ;;
    *)
        printf 'Unsupported host architecture: %s\n' "$(uname -m)" >&2
        exit 1
        ;;
esac

cmake \
    -S "${SDK_DIR}" \
    -B "${BUILD_DIR}" \
    -DPYTHON_EXECUTABLE="$("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"

cmake --build "${BUILD_DIR}" \
    --target unitree_actuator_sdk example_goM8010_6_motor \
    --parallel "${MAX_JOBS}"

extension_suffix="$("${PYTHON_BIN}" -c \
    'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
module_path="${SDK_DIR}/lib/unitree_actuator_sdk${extension_suffix}"
if [[ ! -f "${module_path}" ]]; then
    printf 'Built Python module was not found: %s\n' "${module_path}" >&2
    exit 1
fi

LD_LIBRARY_PATH="${SDK_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
PYTHONPATH="${SDK_DIR}/lib${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" - <<'PY'
import unitree_actuator_sdk as sdk

expected_api = 3
expected_commit = '5b79a42d81cd69adac1367db79efdb972a898fd1'
actual_api = getattr(sdk, '__quadruped_binding_api__', None)
actual_commit = getattr(sdk, '__quadruped_sdk_commit__', None)
if actual_api != expected_api or actual_commit != expected_commit:
    raise RuntimeError(
        'Unexpected Unitree binding identity: '
        f'API={actual_api!r}, commit={actual_commit!r}'
    )

motor_type = sdk.MotorType.GO_M8010_6
gear_ratio = float(sdk.queryGearRatio(motor_type))
if gear_ratio <= 0.0:
    raise RuntimeError(f'Invalid GO-M8010-6 gear ratio: {gear_ratio}')

print(f'Unitree Actuator SDK module: {sdk.__file__}')
print(f'Quadruped binding API: {actual_api}')
print(f'Unitree SDK commit: {actual_commit}')
print(f'GO-M8010-6 gear ratio: {gear_ratio:.6f}')
PY
