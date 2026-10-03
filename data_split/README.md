# Split files

For maximum transparency, the small split-list files used in the experiments can be committed here before publication.
The current loaders also support the original layout where these lists are stored inside each downloaded dataset directory.

Recommended files to mirror from the dataset ZIP archives:

```text
data_split/
├── ACDC/
│   ├── train_slices.list
│   ├── val.list
│   └── test.list
└── Synapse/
    ├── train_slices.txt
    └── val.txt
```

Do **not** invent or regenerate these files if the experiments were run with an existing public split. Copy the exact lists used for the reported results.
