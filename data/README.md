# Data

```
data/
  vocab/                          label lists                                  tracked
    categories_places365.txt        365 Places365 scene classes
    IO_places365.txt                official indoor (1) / outdoor (2) label per class
    imagenet_simple_labels.json     1000 ImageNet class names
  audiocaps/
    splits/                       AudioCaps captions + YouTube ids             tracked
      train.csv                     sound-candidate pool (45,713 unique captions)
      val.csv                       used only for tuning
      test.csv                      used only for evaluation
    pairs/                        image + ground-truth audio pairs              git-ignored
      val/  test/                   frames/<ytid>.jpg, audio/<ytid>.wav (16 kHz mono),
                                    manifest.json, failed.json (unavailable videos)
  places365/                                                                    git-ignored
    filelists/                      official split lists (places365_val.txt, ...)
    val_256/                        36,500 validation images (256 px)
  cache/                          CLIP text-embedding cache, safe to delete     git-ignored
```

## Sources and how each is used

| Dataset | Source | Used for | Contains audio? |
|---|---|---|---|
| Places365 | [places2.csail.mit.edu](http://places2.csail.mit.edu/) — `val_256.tar`, `filelist_places365-standard.tar`; `IO_places365.txt` from [CSAILVision/places365](https://github.com/CSAILVision/places365) | scene vocabulary; indoor/outdoor prior; recognition accuracy on val | no |
| ImageNet-1k | class names from [anishathalye/imagenet-simple-labels](https://github.com/anishathalye/imagenet-simple-labels) | object vocabulary (names only, no images) | no |
| AudioCaps | captions from [cdjkim/audiocaps](https://github.com/cdjkim/audiocaps); media from YouTube | train captions = sound candidates; val = tuning; test = evaluation | yes (via YouTube) |

## Rebuilding

```
# AudioCaps pairs (needs Node.js for yt-dlp); deterministic sample with --seed 0
py -m scripts.fetch_audiocaps --split test --n 100
py -m scripts.fetch_audiocaps --split val --n 100

# Places365 val
curl -L -o val_256.tar http://data.csail.mit.edu/places/places365/val_256.tar
tar -xf val_256.tar -C data/places365
curl -L -o filelist.tar http://data.csail.mit.edu/places/places365/filelist_places365-standard.tar
tar -xf filelist.tar -C data/places365/filelists
```

YouTube videos disappear over time, so a re-fetch can return a slightly
different set; `failed.json` lists the ids that were unavailable.

## Terms

The downloaded media is not redistributed in this repository. Places365 images
are provided for non-commercial research and education; AudioCaps audio and
video belong to the original YouTube uploaders. Check each dataset's terms
before sharing derived files.
