import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader

global_seed = 0

class PatientDataGenerator(Dataset):
    def __init__(self, cfp_folder, oct_folder, batch_size=2,train=False,
                 root_path ='/mnt/c/Users/Desktop',transform = None):
        self.cfp_folder = cfp_folder
        self.oct_folder = oct_folder
        self.batch_size = batch_size
        self.train = train
        self.transform = transform

        self.root_path = root_path
        self.filenames = sorted(
                    [os.path.splitext(f)[0] for f in os.listdir(cfp_folder) if
                     os.path.isfile(os.path.join(cfp_folder, f))],
                    key=lambda x: int(x)
                )

    def __getitem__(self, item):
        case = self.filenames[item]
        cfp_path = os.path.join(self.cfp_folder, case + '.png')
        oct_path = os.path.join(self.oct_folder, case + '.png')

        cfp = Image.open(cfp_path)
        oct = Image.open(oct_path)

        if self.transform is not None:
            cfp = self.transform(cfp)
            oct = self.transform(oct)

        return cfp, oct, case  #

    def __len__(self):
        return len(self.filenames)


