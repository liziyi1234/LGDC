LGDC: Omnidirectional Image Quality Assessment


Introduction

LGDC is an omnidirectional image quality assessment method that takes a complete ERP image as input and predicts its quality score.
Key Innovations:
GPEC: Compensates for ERP projection distortion using latitude-longitude information
MQCA: Multi-scale quality feature extraction
ACDP: Adaptively captures distortions in different spatial regions
CGDC: Fuses features from MQCA and ACDC


Quick Start

Installation
git clone https://github.com/liziyi1234/LGDC.git
cd LGDC
pip install torch torchvision einops timm scipy pandas pillow tqdm

Data Preparation

/path/to/dataset/
├── images/
│   ├── 1.jpg
│   └── ...
└── labels.csv    # Columns: dis, mos

Training
python train.py

Inference Example
from LGDC import LGDC
from torchvision import transforms
from PIL import Image

model = LGDC().cuda()
model.load_state_dict(torch.load('best.pth')['model_state_dict'])
model.eval()

img = Image.open('test.jpg').convert('RGB')
img = transforms.Compose([
    transforms.Resize((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])(img).unsqueeze(0).cuda()

score = model(img)  # Output quality score
print(f'MOS: {score.item():.4f}')

File Description
LGDC.py	        Main model (MQCA + ACDC + CGDC)
GPEC.py	        Geometry compensation module
train.py	      Training entry point
config.py	      Configuration parameters
MyDataset.py	  Data loader
utils.py        Utility functions
cyclic_shift.py	Data augmentation

Citation
@article{yan2026lgdc,
  title={Viewport-Unaware Blind Omnidirectional Image Quality Assessment: Learning from Geometry-Deformed Content},
  author={Yan, Jiebin and Li, Ziyi and Wu, Kangcheng and Chen, Pengfei and Zuo, Yifan and Chen, Junjie and Fang, Yuming},
  year={2026}
}
