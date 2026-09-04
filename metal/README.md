# GPT-OSS Metal workspace

This directory is reserved for the official OpenAI GPT-OSS Metal reference
implementation. It is intentionally separate from the llama.cpp models.

Expected layout:

```text
metal/
  gpt-oss/                                  # cloned openai/gpt-oss source
  models/gpt-oss-20b/metal/model.bin        # converted Metal checkpoint
  .venv/                                     # created by ImageVideoStudio
```

In the ImageVideoStudio language-model screen, choose **Initialize Metal**
after placing the official source. Initialization creates `.venv` and installs
`gpt-oss[metal]` into it when the source is present. It never downloads the
large model checkpoint. The Metal implementation is an official reference
backend and may be experimental.
