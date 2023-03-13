from torch import nn
import torchvision
from config import CFG
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification


class BaseModel(nn.Module):
    def __init__(self, num_classes=CFG.n_classes, fc_type='shallow', binary=True):
        super(BaseModel, self).__init__()
        self.fc_type = fc_type
        self.num_classes = num_classes

        # get backbone
        # self.backbone = r2plus1d_18(pretrained=True)
        self.backbone = getattr(torchvision.models.video,
                                CFG.model_name)(pretrained=True)
        self.backbone.fc = self.get_fc()
        self.binary = binary

    def get_fc(self):
        if self.fc_type == 'deep':
            fc = nn.Sequential(nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features//2),
                               nn.BatchNorm1d(
                                   self.backbone.fc.in_features//2,  momentum=0.1),
                               nn.ReLU(),
                               nn.Linear(self.backbone.fc.in_features //
                                         2, self.num_classes)
                               )

        elif self.fc_type == 'shallow':
            fc = nn.Linear(self.backbone.fc.in_features, self.num_classes)
        else:
            raise ValueError(f"Wrong fc-type input {self.fc_type}")
        return fc

    def forward(self, x):
        x = self.backbone(x)

        return x


class VideoMAE(nn.Module):
    def __init__(self, num_classes=CFG.n_classes, fc_type='shallow', binary=True):
        super(BaseModel, self).__init__()
        self.fc_type = fc_type
        self.num_classes = num_classes

        # get backbone
        # self.backbone = r2plus1d_18(pretrained=True)
        self.backbone = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-large-finetuned-kinetics")
        self.backbone.classifier = self.get_fc()
        self.binary = binary

    def get_fc(self):
        if self.fc_type == 'deep':
            fc = nn.Sequential(nn.Linear(self.backbone.fc.in_features, self.backbone.fc.in_features//2),
                               nn.BatchNorm1d(
                                   self.backbone.fc.in_features//2,  momentum=0.1),
                               nn.ReLU(),
                               nn.Linear(self.backbone.fc.in_features //
                                         2, self.num_classes)
                               )

        elif self.fc_type == 'shallow':
            fc = nn.Linear(self.backbone.classifier.in_features,
                           self.num_classes)
        else:
            raise ValueError(f"Wrong fc-type input {self.fc_type}")
        return fc

    def forward(self, x):
        x = self.backbone(x)

        return x
