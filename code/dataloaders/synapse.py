from pathlib import Path
import h5py
import numpy as np

from .common import ArraySliceDataset, read_list, ordered_patients


class SynapseDataModule:
    def __init__(self, root, labeled_num, image_size=256):
        self.root = Path(root)
        self.labeled_num = int(labeled_num)
        self.image_size = int(image_size)

        self.train_ids = read_list(self.root / "train_slices.txt")
        self.test_ids = read_list(self.root / "val.txt")

        patients = ordered_patients(self.train_ids)
        if self.labeled_num > len(patients):
            raise ValueError("labeled_num exceeds number of Synapse training patients")
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
        path = self.root / "train_npz" / f"{case_id}.npz"
        data = np.load(path)
        return data["image"], data["label"]

    def load_volume(self, case_id):
        path = self.root / "test_vol_h5" / f"{case_id}.npy.h5"
        with h5py.File(path, "r") as f:
            return f["image"][:], f["label"][:]

    def validation_cases(self):
        # The common Synapse split names the held-out evaluation list `val.txt`.
        return list(self.test_ids)

    def test_cases(self):
        return list(self.test_ids)
