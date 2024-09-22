
# SAM Model UI Interface - Enhancing Efficiency in Semantic Segmentation Labeling    
 This project utilizes the SAM (Segment-Anything Model) from Facebook Research's [segment-anything](https://github.com/facebookresearch/segment-anything) repository. It aims to provide an intuitive and efficient user interface (UI) to assist users in generating and editing semantic segmentation labels.    
    
## Project Background    
 Semantic segmentation is a crucial task in computer vision, aiming to assign semantic category labels to each pixel in an image, such as humans, vehicles, roads, etc. However, manual annotation of large-scale datasets is time-consuming and tedious. This project leverages the powerful SAM model and combines it with an intuitive UI to offer a more efficient way to create and edit semantic segmentation labels.    
    
## Key Features    
 - **Semantic Segmentation with SAM Model**    
- Utilize the pretrained SAM model for image semantic segmentation, generating initial label outputs.    
      
- **Interactive UI**    
- Provide a user-friendly interface that allows users to edit and refine generated labels to ensure accuracy and completeness.    
      
- **Real-time Preview and Saving**    
- Instantly preview the effects of edited labels and support saving edited results in standard formats such as PNG, JSON or XML.    
    
## How to Use    
 ### Environment Setup
 ```bash
pip install -r requirements.txt
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118  
```

`Notice:` SAM2 requires Torch version 2.3.0 or higher, and you need to download the version compatible with your current CUDA environment.  
To check your CUDA version, run the `nvidia-smi` command in the terminal.

`For example:` if you're using CUDA 11.8, you should download Torch version 2.3.1.

###  Install SAM 2  
```bash  
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2 & pip install -e .
```  
  
### Setting Parameter  
```json  
{    
  "version": 2,
  "Model_Url":"https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", 
  "sam_checkpoint": "checkpoints/sam2_hiera_base_plus.pt",    
  "classes": ["Card", "Truck", "Dog", "Bicycle"]
 }  
```  
 - version: Set Use SAM1 or SAM2. use SAM1 set 1, use SAM2 set 2.
 - Model_Url: if you don't have SAM model, you need set url, it can auto download model.
 - sam_checkpoint: Set Model weight path  
 - classes: Model Label setting. ex: ["dog", "cat", "car", ....]
###  Running the UI
```bash 
 python SAM_Main.py  
``` 

<p float="left">    
    <img src="assets/UI_interfaceV2_1.JPG?raw=true" width="49.7%" />    
    <img src="assets/UI_interfaceV2_2.JPG?raw=true" width="49.7%" />    
    <img src="assets/UI_interfaceV2_3.JPG?raw=true" width="49.7%" />    
    <img src="assets/UI_interfaceV2_4.JPG?raw=true" width="49.7%" />    
</p>    
    
### User Interface Buttons    
 In this project, QPushButton widgets are utilized for various functionalities in the graphical user interface (GUI). Each button is connected to a corresponding method to handle user interactions effectively:    
    
- **Load Folder Button**: This button (`Load Folder`) is used to trigger the loading of a folder or directory.    
    
- **Predict Mask Button**: The `Predict Mask` button initiates the process of predicting a mask or segmentation for the loaded data.    
    
- **Save Mask Button**: Clicking the `Save Mask` button saves the generated mask or segmentation results to a file. Handle the saving process and ensure the output is stored in a specified format (e.g., PNG, JSON).    
    
- **Clear Points Button**: The `Clear Points` button allows users to reset or clear any marked points or annotations on the interface.    
    
## TODO    
 &#9744; Save the coordinates in YOLO-seg format.      
&#9744; Added the coco and Pascal VOC format.    
    
## License The model is licensed under the [Apache 2.0 license](LICENSE).  
  
## Reference  
https://github.com/facebookresearch/segment-anything-2/tree/main  
https://github.com/MikeHuang0618/SAM-UI/tree/feature_SAM2