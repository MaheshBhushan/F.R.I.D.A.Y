# Dataset manifests

Raw data lives in ignored subdirectories. Each JSONL manifest row must contain:

```json
{"path":"raw/user/session-001/0001.wav","sha256":"...","samples":24000,"sample_rate":16000,"duration_seconds":1.5,"label":"positive","split":"training","source":"primary-user","session":"session-001","license":"private-user-data","phrase":"Friday","condition":"normal-close"}
```

Splits are assigned by original recording/session before augmentation. Derived
siblings must never cross split boundaries. Durations come from decoded audio
sample counts, never feature-row counts.
