from pathlib import Path
import h5py

from .common import ArraySliceDataset, read_list, ordered_patients


class ACDCDataModule:
    def __init__(self, root, labeled_num, image_size=256):
        self.root = Path(root)
        self.labeled_num = int(labeled_num)
        self.image_size = int(image_size)

        self.train_ids = read_list(self.root / "train_slices.list")
        self.val_ids = read_list(self.root / "val.list")
        self.test_ids = read_list(self.root / "test.list")

        patients = ordered_patients(self.train_ids)
        if self.labeled_num > len(patients):
            raise ValueError("labeled_num exceeds number of ACDC training patients")
        self.labeled_patients = set(patients[: self.labeled_num])
        self.unlabeled_patients = set(patients[self.labeled_num :])

        labeled_ids = [x for x in self.train_ids if x.split("_")[0] in self.labeled_patients]
        unlabeled_ids = [x for x in self.train_ids if x.split("_")[0] in self.unlabeled_patients]

        self.labeled_dataset = ArraySliceDataset(
            labeled_ids, self._load_train_slice, labeled=True, image_size=image_size
        )
        self.unlabeled_dataset = ArraySliceDataset(
            unlabeled_ids, self._load_train_slice, labeled=False, image_size=image_size
        )

    def _load_train_slice(self, case_id):
        path = self.root / "data" / "slices" / f"{case_id}.h5"
        with h5py.File(path, "r") as f:
            return f["image"][:], f["label"][:]

    def load_volume(self, case_id):
        path = self.root / "data" / f"{case_id}.h5"
        with h5py.File(path, "r") as f:
            return f["image"][:], f["label"][:]

    def validation_cases(self):
        # Match the original notebook exactly: `test.list` is used for
        # epoch-wise evaluation/model selection. `val.list` is loaded for
        # dataset compatibility but is not used by the original training loop.
        return list(self.test_ids)

    def test_cases(self):
        return list(self.test_ids)
