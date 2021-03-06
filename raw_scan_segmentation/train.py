from conv_autoencoder import ConvAutoEncoder, PatchConvAutoEncoderParameters
from blob_storage_image_dataset import BlobStorageImageDataset
import torch
import torch.onnx
from torch import device, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
import math
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from dotenv import load_dotenv
from model_store import ModelStore

class Trainer(object):
    def __init__(
        self,
        device: torch.device,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        model_store: ModelStore,
        output_root: str,
        autoencoder_parameters: PatchConvAutoEncoderParameters,
        learning_rate: float,
    ) -> None:
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = 1000
        self.model_store = model_store
        self.model = ConvAutoEncoder(autoencoder_parameters)
        self.model.to(self.device)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = self.configure_optimizers()
        self.model_metadata = autoencoder_parameters._asdict()
        self.lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.65,
            min_lr=1e-7,
            threshold=5e-3,
            threshold_mode="rel",
        )
        self.output_root = output_root
        self.hyper_param_string = "k{}_h{}_ic{}_fc{}_lr{}".format(
            autoencoder_parameters.kernel_size,
            autoencoder_parameters.hidden_layers,
            autoencoder_parameters.initial_channels,
            autoencoder_parameters.final_channels,
            learning_rate,
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
            data = data.to(self.device)
            if len(data.size()) == 5:
                bs, ncrops, c, h, w = data.size()
                data = data.view(-1, c, h, w)
            self.optimizer.zero_grad()
            loss = self.training_step(data)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batch_count += 1
            if batch_count % 5 == 0:
                print(
                    "    Train - batch {} epoch {} loss: {:.6f}".format(
                        batch_count, epoch_num, running_loss / batch_count
                    )
                )
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            self.tb_writer.add_scalar("Loss/train", avg_loss, epoch_num)
            print("Train - epoch {} loss: {:.6f}".format(epoch_num, avg_loss))

    def validation_phase(self, epoch_num: int) -> None:
        running_loss = 0.0
        batch_count = 0
        with torch.no_grad():
            for i, data in enumerate(self.validation_dataloader):
                data = data.to(self.device)
                if len(data.size()) == 5:
                    bs, ncrops, c, h, w = data.size()
                    data = data.view(-1, c, h, w)
                loss = self.validation_step(data)
                running_loss += loss.item()
                batch_count += 1
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            self.lr_sched.step(avg_loss)
            self.tb_writer.add_scalar("Loss/test", avg_loss, epoch_num)
            self.tb_writer.add_scalar(
                "LearningRate", self.optimizer.param_groups[0]["lr"], epoch_num
            )
            print(
                "Test - epoch {} loss: {:.6f} learning rate: {:.6f}".format(
                    epoch_num, avg_loss, self.optimizer.param_groups[0]["lr"]
                )
            )
            if (
                self.lowest_training_loss == None
                or running_loss < self.lowest_training_loss
            ):
                self.lowest_training_loss = running_loss
                self.export_model(epoch_num, avg_loss)

    def export_model(self, epoch_num: int, avg_loss: float) -> None:
        epoch_folder = os.path.join(self.output_root, str(epoch_num))
        if not os.path.exists(epoch_folder):
            os.makedirs(epoch_folder)
        pt_model_path = os.path.join(epoch_folder, "model.pt")
        torch.save(self.model.state_dict(), pt_model_path)
        sample_input = torch.zeros(1, 3, 1024, 1024, device=self.device)
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
        self.model_metadata["last_epoch"] = epoch_num
        self.model_metadata["last_test_loss"] = avg_loss
        self.model_store.publish(pt_model_path, onnx_path, "PatchAutoencoder", self.hyper_param_string, self.model_metadata)

    def run(self):
        for epoch in range(self.epochs):
            self.training_phase(epoch)
            self.validation_phase(epoch)


# kernel to input dim map
KERNEL_TO_INPUT_DIM = {
    8: 400,
    12: 396,
    16: 400,
    20: 400,
    24: 396,
    32: 400,
}


def main():
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
        800,
        cache_dir,
    )
    dataset_size = len(dataset)
    train_size = math.floor(dataset_size * 0.65)
    train, val = random_split(dataset, [train_size, dataset_size - train_size])

    model_store = ModelStore(storage_connection_string, "model-store", os.path.join(script_folder, "cache", "model_store"))

    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=True)

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = os.path.join(script_folder, "output", datetime_str)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        print("Running on CPU")
    autoencoder_parameters = PatchConvAutoEncoderParameters(32, 2, 128, 32)
    trainer = Trainer(
        device, train_loader, val_loader, model_store, output_root, autoencoder_parameters, 0.003
    )
    trainer.run()


if __name__ == "__main__":
    load_dotenv()
    main()
