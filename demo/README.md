# Frozen academic demo

`deporank_demo_run.zip` contains the completed academic run, not uploaded customer data. It preserves the original dataset/config hashes, selection lock, all five model families, predictions, and evaluation files byte for byte. Restore it with `python scripts/restore_demo.py`.

`manifest.json` pins the archive checksum. The restore script validates the archive and refuses to overwrite a different existing artifact. The app then checks model files, selection provenance, and environment versions before loading a model.

**Security:** joblib/pickle models can execute code when loaded. Only restore the trusted archive shipped with this repository. A checksum detects changes; it does not make an untrusted model safe. Never accept model uploads from users.

Historical experiment metadata is immutable evidence. Some original run notes remain in their original language; English explanations are supplied by the report and project documentation. Changing locked files would invalidate the experiment.
