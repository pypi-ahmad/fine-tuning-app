# Disclaimer

Please read this before training on, exporting, or publishing any dataset or model with Fine-Tuning Studio.

## You run this entirely on your own machine, with your own hardware and credentials

Fine-Tuning Studio is a local-first tool. There is no hosted version, no backend server operated by the author, and no account system. Training runs on your own CPU/GPU, using whichever datasets and base models you provide. Hugging Face credentials (`HF_TOKEN` or the legacy `HUGGING_FACE_HUB_TOKEN`) are read only from your environment and are never displayed, logged, or written into the project — see [README.md](README.md#3-configure-hugging-face-access-optional) and [SECURITY.md](SECURITY.md).

## You are responsible for the data and models you process

**You, and only you, are responsible for:**

- Having the rights or permission to use every dataset and base model you load, whether from the Hugging Face Hub, an upload, or elsewhere, and complying with its license, terms of use, and any privacy or contractual obligations that apply to it.
- Deciding whether a dataset or model may be pushed to the Hugging Face Hub or any other destination. Fine-Tuning Studio only uploads when you explicitly choose to (adapter export, merged-model push, or Ollama import) — nothing is uploaded automatically.
- Reviewing any model repository or custom reward module before enabling `trust_remote_code` or running it. Both execute arbitrary Python code on your machine when enabled, and the UI's typed confirmation is a safeguard, not a guarantee — see [SECURITY.md](SECURITY.md).
- Any Hugging Face Hub costs, rate limits, or terms associated with your account and token.
- Verifying that a trained or exported model behaves as intended before relying on it, deploying it, or sharing it further.

## No warranty, no liability

This software is provided "as is," without warranty of any kind, as stated in the [MIT License](LICENSE). The author is not liable for any damage, data loss, GPU hardware issues, unintended disclosure, Hub costs, or other consequences arising from your use of this tool. Use it at your own risk.

## No financial support wanted

This project is free, open-source, and does not want or accept donations, sponsorships, or any other form of financial contribution — see [SUPPORT.md](SUPPORT.md).
