import hashlib
import json
import os
import urllib.request
import zipfile
from pathlib import Path

from sdks.novavision.src.helper.package import PackageHelper

STORAGE_ROOT = Path("/storage")
PACKAGE_ROOT = STORAGE_ROOT / "FewShotInference"


def unwrap_storage_id(storage_value):
    if isinstance(storage_value, list):
        if not storage_value:
            return None
        storage_value = storage_value[0]

    if isinstance(storage_value, dict):
        for key in ("id_storage", "id", "value"):
            if storage_value.get(key) not in (None, ""):
                storage_value = storage_value[key]
                break

    if storage_value in (None, ""):
        return None

    if isinstance(storage_value, int):
        return storage_value

    if isinstance(storage_value, str):
        text = storage_value.strip()
        if text.isdigit():
            return int(text)
        try:
            return unwrap_storage_id(json.loads(text))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None

    return None


def md5_hash(file_path, chunk_size=8192):
    digest = hashlib.md5()
    with open(file_path, "rb") as file_obj:
        while True:
            data = file_obj.read(chunk_size)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def download_storage_file(storage_id):
    storage_id = unwrap_storage_id(storage_id)
    if storage_id is None:
        raise ValueError("A valid storage file id is required.")

    STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    result = PackageHelper.get_storage_details(storage_id)
    data = result["data"]
    url_path = result["data_url"]
    name = data.get("name") or f"storage_{storage_id}"
    hash_file = data.get("hash_file")
    file_path = STORAGE_ROOT / name

    should_download = True
    if file_path.exists() and hash_file:
        try:
            should_download = md5_hash(file_path) != hash_file
        except OSError:
            should_download = True
    elif file_path.exists() and not hash_file:
        should_download = False

    if should_download:
        urllib.request.urlretrieve(url_path, file_path)

    return file_path, data


def _is_within_directory(directory, target):
    directory = os.path.abspath(directory)
    target = os.path.abspath(target)
    return os.path.commonpath([directory]) == os.path.commonpath([directory, target])


def extract_storage_zip(storage_id):
    zip_path, storage_data = download_storage_file(storage_id)
    folder_name = (
        storage_data.get("hash_file") or f"id_{storage_data.get('id') or zip_path.stem}"
    )
    extract_dir = PACKAGE_ROOT / "galleries" / str(folder_name)
    marker = extract_dir / ".ready"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                member_path = extract_dir / member.filename
                if not _is_within_directory(extract_dir, member_path):
                    raise ValueError(f"Unsafe zip path: {member.filename}")
            zip_ref.extractall(extract_dir)
        marker.write_text(str(zip_path), encoding="utf-8")
    return extract_dir


def resolve_device(device_config):
    token = str(device_config or "CPU").strip().lower()
    if token not in {"gpu", "cuda", "0", "cuda:0"}:
        return "cpu"

    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
