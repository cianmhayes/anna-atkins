from conv_autoencoder import ConvAutoEncoder
from blob_storage_image_dataset import BlobStorageImageDataset
import torch
import torch.onnx
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import math
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Trainer(object):
    def __init__(
        self,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        output_root: str,
        output_channels: int,
        kernel_size: int,
        learning_rate: float
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = 100
        self.model = ConvAutoEncoder(output_channels, kernel_size)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = self.configure_optimizers()
        self.output_root = output_root
        self.hyper_param_string = "cout{}_k{}_lr{}".format(
            output_channels, kernel_size, learning_rate
        )
        self.tb_writer = SummaryWriter(
            log_dir=self.output_root, comment=self.hyper_param_string
        )
        self.lowest_training_loss = None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch):
        z, x_hat = self.model(train_batch)
        loss = F.mse_loss(x_hat, train_batch)
        return loss

    def validation_step(self, val_batch):
        z, x_hat = self.model(val_batch)
        loss = F.mse_loss(x_hat, val_batch)
        return loss

    def training_phase(self, epoch_num: int) -> None:
        running_loss = 0.0
        batch_count = 0
        for _, data in enumerate(self.training_dataloader):
            self.optimizer.zero_grad()
            loss = self.training_step(data)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            if batch_count % 10 == 0:
                print("    Train - batch {} epoch {} loss: {}".format(batch_count, epoch_num, running_loss / batch_count))
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            print("Train - epoch {} loss: {}".format(epoch_num, avg_loss))
            self.tb_writer.add_scalar("Loss/train", avg_loss, epoch_num)

    def validation_phase(self, epoch_num: int) -> None:
        running_loss = 0.0
        batch_count = 0
        for i, data in enumerate(self.validation_dataloader):
            loss = self.validation_step(data)
            running_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            print("Test - epoch {} loss: {}".format(epoch_num, avg_loss))
            self.tb_writer.add_scalar("Loss/test", avg_loss, epoch_num)
        if (
            self.lowest_training_loss == None
            or running_loss < self.lowest_training_loss
        ):
            self.lowest_training_loss = running_loss
            self.export_model(epoch_num)

    def export_model(self, epoch_num: int) -> None:
        epoch_folder = os.path.join(self.output_root, str(epoch_num))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)
        pt_model_path = os.path.join(epoch_folder, "model.pt")
        torch.save(self.model.state_dict(), pt_model_path)
        sample_input = torch.zeros(1, 3, 1024, 1024)
        onnx_path = os.path.join(epoch_folder, "model.onnx")
        torch.onnx.export(
            self.model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["reconstructed"],
            dynamic_axes={
                "image": {0: "batch", 2: "h", 3: "w"},
                "reconstructed": {0: "batch", 2: "h", 3: "w"},
            },
        )

    def run(self):
        for epoch in range(self.epochs):
            self.training_phase(epoch)
            self.validation_phase(epoch)


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    # data
    storage_connection_string = os.environ.get("STORAGE_CONNECTION_STRING")
    storage_account_container = "anna-atkins"

    script_folder = os.path.dirname(__file__)
    cache_dir = os.path.join(script_folder, "cache", "data", "unprocessed_1000px")
    dataset = BlobStorageImageDataset(
        storage_connection_string,
        storage_account_container,
        "unprocessed_1000px/",
        404,
        cache_dir,
    )
    dataset_size = len(dataset)
    train_size = math.floor(dataset_size * 0.8)
    train, val = random_split(dataset, [train_size, dataset_size - train_size])

    train_loader = DataLoader(train, batch_size=8)
    val_loader = DataLoader(val, batch_size=8)

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = os.path.join(script_folder, "output", datetime_str)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    output_channel_sizes = [16, 24, 32, 64]
    kernel_sizes = [8, 12, 16, 24]
    learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    for ocs in output_channel_sizes:
        for ks in kernel_sizes:
            for lr in learning_rates:
                print("Training - Output channel: {}, Kernel: {}, Learning Rate: {}".format(ocs, ks, lr))
                print("====================================================================================================")
                trainer = Trainer(train_loader, val_loader, output_root, ocs, ks, lr)
                trainer.run()