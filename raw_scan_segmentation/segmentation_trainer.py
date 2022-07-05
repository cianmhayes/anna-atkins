from conv_autoencoder import *
from annotated_image_dataset import AnnotatedImageDataset
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
from segmenter import Segmenter, SegmenterParameters
from torchmetrics import Accuracy
from model_store import ModelStore

class SegmentationTrainer(object):
    def __init__(
        self,
        device: torch.device,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        model_store: ModelStore,
        output_root: str,
        tensorboard_log_dir: str,
        pretrained_encoder: ConvEncoder,
        model_parameters: SegmenterParameters,
        learning_rate: float,
    ) -> None:
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = 1000
        self.model_store = model_store
        pretrained_encoder.requires_grad_ = False
        self.model = Segmenter(pretrained_encoder, model_parameters)
        self.model.to(self.device)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = self.configure_optimizers()
        self.lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=0.5, min_lr=1e-8
        )
        self.output_root = output_root
        self.hyper_param_string = "Segmentation_c{}".format(model_parameters.output_classes)
        self.tb_writer = SummaryWriter(
            log_dir=tensorboard_log_dir, comment=self.hyper_param_string
        )
        self.lowest_training_loss = None
        self.model_metadata = model_parameters._asdict()
        self.accuracy_metric = Accuracy(mdmc_average="samplewise")

    def _update_metrics(self, preds:torch.Tensor, target:torch.Tensor) -> None:
        self.accuracy_metric.update(preds, target)

    def _reset_metrics(self) -> None:
        self.accuracy_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer

    def train_step(self, input, target):
        prediction = self.model(input)
        loss = F.cross_entropy(prediction, target, label_smoothing=0.1)
        return loss

    def validation_step(self, input, target):
        prediction = self.model(input)
        loss = F.cross_entropy(prediction, target)
        self._update_metrics(prediction, target)
        return loss

    def training_phase(self, epoch_num: int) -> None:
        running_loss = 0.0
        batch_count = 0
        for _, (data, target) in enumerate(self.training_dataloader):
            data = data.to(self.device)
            if len(data.size()) == 5:
                bs, ncrops, c, h, w = data.size()
                data = data.view(-1, c, h, w)
            if len(target.size()) == 4:
                bs, ncrops, h, w = target.size()
                target = target.view(-1, h, w)
            self.optimizer.zero_grad()
            loss = self.train_step(data, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            self.tb_writer.add_scalar("Loss/train", avg_loss, epoch_num)
            print("Train - epoch {} loss: {:.6f}".format(epoch_num, avg_loss))

    def validation_phase(self, epoch_num: int) -> None:
        running_loss = 0.0
        batch_count = 0
        self._reset_metrics()
        with torch.no_grad():
            for _, (data, target) in enumerate(self.validation_dataloader):
                data = data.to(self.device)
                if len(data.size()) == 5:
                    bs, ncrops, c, h, w = data.size()
                    data = data.view(-1, c, h, w)
                if len(target.size()) == 4:
                    bs, ncrops, h, w = target.size()
                    target = target.view(-1, h, w)
                loss = self.validation_step(data, target)
                running_loss += loss.item()
                batch_count += 1
        if batch_count > 0:
            avg_loss = running_loss / batch_count
            accuracy = self.accuracy_metric.compute()
            self.lr_sched.step(avg_loss)
            self.tb_writer.add_scalar("Loss/test", avg_loss, epoch_num)
            self.tb_writer.add_scalar("Quality/Accuracy", accuracy, epoch_num)
            self.tb_writer.add_scalar(
                "LearningRate", self.optimizer.param_groups[0]["lr"], epoch_num
            )
            print(
                "Test  - epoch {} loss: {:.6f} accuracy: {:.3f} learning rate: {:.6f}".format(
                    epoch_num, avg_loss, accuracy, self.optimizer.param_groups[0]["lr"]
                )
            )
            if (
                self.lowest_training_loss == None
                or running_loss < self.lowest_training_loss
            ):
                self.lowest_training_loss = running_loss
                self.export_model(epoch_num, avg_loss, accuracy)

    def export_model(self, epoch_num: int, avg_loss: float, accuracy: float) -> None:
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
        self.model_metadata["last_test_accuracy"] = accuracy
        self.model_store.publish(pt_model_path, onnx_path, "RawScanSegmenter", self.hyper_param_string, self.model_metadata)

    def run(self):
        for epoch in range(self.epochs):
            self.training_phase(epoch)
            self.validation_phase(epoch)


def main():
    torch.manual_seed(0)
    random.seed(0)
    # data
    storage_connection_string = os.environ.get("STORAGE_CONNECTION_STRING")
    storage_account_container = "anna-atkins"
    annotation_api_host = os.getenv("ANNOTATION_API_HOST")
    annotaiton_api_key = os.getenv("ANNOTATION_API_KEY")

    script_folder = os.path.dirname(__file__)
    cache_dir = os.path.join(script_folder, "cache", "data", "unprocessed_1000px")
    dataset = AnnotatedImageDataset(
        storage_connection_string,
        storage_account_container,
        annotation_api_host,
        annotaiton_api_key,
        "unprocessed_1000px/",
        800,
        cache_dir,
    )
    dataset_size = len(dataset)
    train_size = math.floor(dataset_size * 0.65)
    train, val = random_split(dataset, [train_size, dataset_size - train_size])

    train_loader = DataLoader(train, batch_size=16)
    val_loader = DataLoader(val, batch_size=16)

    # TODO how should I identify autoencoder outputs and segmenter outputs
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = os.path.join(script_folder, "output", datetime_str)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    tensorboard_root = os.path.join(script_folder, "output", "tensorboard")
    if not os.path.exists(tensorboard_root):
        os.makedirs(tensorboard_root)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        print("Running on CPU")
    autoencoder_parameters = PatchConvAutoEncoderParameters(32, 2, 128, 32)
    model_store = ModelStore(storage_connection_string, "model-store", os.path.join(script_folder, "cache", "model_store"))
    pretrained_patch_autoencoder_path = model_store.get_checkpoint("PatchAutoencoder", "k32_h2_ic128_fc32_lr0.003")
    pretrained_patch_autorncoder = load_pretrained_autoencoder_from_checkpoint(
        autoencoder_parameters, pretrained_patch_autoencoder_path
    )
    pretrained_encoder = pretrained_patch_autorncoder.encoder
    segmenter_parameters = SegmenterParameters(
        autoencoder_parameters.kernel_size,
        autoencoder_parameters.final_channels,
        autoencoder_parameters.final_channels,
        dataset.get_annotation_count(),
    )
    trainer = SegmentationTrainer(
        device,
        train_loader,
        val_loader,
        model_store,
        output_root,
        tensorboard_root,
        pretrained_encoder,
        segmenter_parameters,
        0.003,
    )
    trainer.run()


if __name__ == "__main__":
    load_dotenv()
    main()
