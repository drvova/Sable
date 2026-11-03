"""Locations of the external inputs of the scripts, from environment variables.

  SABLE_HF_SNAPSHOT  directory of the Hugging Face checkpoint of SABLE_ARCH (config, tokenizer,
                     *.safetensors).  If unset, the snapshot is looked up in the local Hugging Face cache
                     (HF_HOME / HF_HUB_CACHE) without downloading; fetch it once with
                     `hf download <SABLE_ARCH>`.
  SABLE_ARCH         the model, as its Hugging Face repository (default Qwen/Qwen2.5-32B-Instruct; also
                     Qwen/Qwen3-32B and Qwen/Qwen3-30B-A3B); selects the F_O architecture in zk.model.ARCHS.
  SABLE_FO_MODEL     the F_O model file written by `python -m scripts.quantize` (about 33 GB for the 32B models).
  SABLE_DATA         optional cache root for the evaluation datasets (WikiText-2, MMLU, GSM8K): dataset
                     NAME is cached in $SABLE_DATA/datasets_NAME.  If unset, the `datasets` default cache.

The F_O calibration corpus is docs/spec/corrected.tex (the quantizer uses its first 1,024 tokens)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_CORPUS = os.path.join(ROOT, "docs", "spec", "corrected.tex")
# SABLE_ARCH: the Hugging Face repository of the model (one of zk.model.ARCHS); Qwen2.5-32B-Instruct by default
HF_REPO = os.environ.get("SABLE_ARCH", "Qwen/Qwen2.5-32B-Instruct")


def hf_snapshot():
    """Directory of the Hugging Face checkpoint."""
    p = os.environ.get("SABLE_HF_SNAPSHOT")
    if p:
        if not os.path.isfile(os.path.join(p, "config.json")):
            raise RuntimeError(f"SABLE_HF_SNAPSHOT={p}: no config.json there (expected a checkpoint snapshot)")
        return p
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(HF_REPO, local_files_only=True)
    except Exception as e:                                  # noqa: BLE001
        raise RuntimeError(f"checkpoint {HF_REPO} not found: set SABLE_HF_SNAPSHOT to its directory, or "
                           f"download it into the Hugging Face cache (`hf download {HF_REPO}`)") from e


def fo_model(must_exist=True):
    """Path of the F_O model file (SABLE_FO_MODEL)."""
    p = os.environ.get("SABLE_FO_MODEL")
    if not p:
        raise RuntimeError("SABLE_FO_MODEL is not set: point it at the F_O model file "
                           "(written by `python -m scripts.quantize`)")
    if must_exist and not os.path.isfile(p):
        raise RuntimeError(f"SABLE_FO_MODEL={p}: no such file (create it with `python -m scripts.quantize`)")
    return p


def datasets_cache(name):
    """cache_dir for datasets.load_dataset of dataset `name` (None: the library's default cache)."""
    root = os.environ.get("SABLE_DATA")
    return os.path.join(root, f"datasets_{name}") if root else None


def wikitext2(split):
    """The WikiText-2 (raw) split as one string."""
    import datasets
    d = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=datasets_cache("wiki"))
    return "".join(d[split]["text"])
