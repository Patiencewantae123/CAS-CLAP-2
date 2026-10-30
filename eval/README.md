# Smoke-test set

Five images (car, cat, bike, a second cat photo, a plain blue square) for a
quick check that the pipeline runs end to end. No ground truth; not used for
any reported result.

```
py -m scripts.test --manifest eval/smoke_manifest.json --run-name smoke --no-generate
```

The evaluation data for reported results is built by `scripts/download_data.py`;
see the main README.
