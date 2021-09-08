import pytorch_lightning as pl
import torch 

class QADataModule(pl.LightningDataModule): 
    def __init__(self, processed_train_dataset, processed_valid_dataset):
        self.processed_train_dataset = processed_train_dataset
        self.processed_valid_dataset = processed_valid_dataset
        self.num_workers = 4
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = True
                
    def train_dataloader(self): 
        return torch.utils.data.DataLoader(
            self.processed_train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, drop_last=True,  
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers,
        )
        
    def val_dataloader(self): 
        return torch.utils.data.DataLoader(
            self.processed_valid_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, drop_last=False,  
            num_workers=self.num_workers, 
            pin_memory=self.pin_memory, 
            persistent_workers=self.persistent_workers,
        )