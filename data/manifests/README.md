# Split manifest

The data-preparation command writes a full file manifest here only if the output path is changed. Keep FER2013 images out of Git. Share the generated JSON manifest through the team's approved private storage when every member needs the same recorded split.

The default generated manifest is under the ignored `outputs/data/` folder. It records each relative file path, transformed-image hash, split, class, exclusion reason, source counts, and split hash.
