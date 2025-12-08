# ai_inference.py
import torch
import numpy as np
import segmentation_models_pytorch as smp
import os
import cv2

class MineSegmenter:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MineSegmenter, cls).__new__(cls)
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cls._load_model()
        return cls._instance

    @classmethod
    def _load_model(cls):
        # UPDATE THIS PATH if your model is in a subfolder
        model_path = "mineguard_unet.pth"
        
        print(f"🔄 Loading AI Model from {model_path}...")
        
        # Define architecture (Must match training)
        cls._model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        )
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=cls.device)
                cls._model.load_state_dict(state_dict)
                print("✅ AI Model Loaded Successfully")
            except Exception as e:
                print(f"❌ Error Loading Model: {e}")
        else:
            print(f"⚠️ WARNING: {model_path} not found. AI predictions will be skipped.")
        
        cls._model.to(cls.device)
        cls._model.eval()

    def predict(self, image_numpy):
        """
        Input: Numpy array (Height, Width, 3) - RGB
        Output: Numpy array (Height, Width) - Binary Mask (0 or 255)
        """
        if self._model is None: 
            return np.zeros((image_numpy.shape[0], image_numpy.shape[1]), dtype=np.uint8)

        # 1. Resize to 512x512 (Model Requirement)
        orig_h, orig_w = image_numpy.shape[:2]
        img_resized = cv2.resize(image_numpy, (512, 512))

        # 2. Normalize & Transpose
        img = img_resized / 255.0
        img = np.transpose(img, (2, 0, 1)).astype('float32')
        img_tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)

        # 3. Inference
        with torch.no_grad():
            output = self._model(img_tensor)
            pred_mask = (output > 0.5).float().squeeze().cpu().numpy()

        # 4. Resize mask back to original size
        pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h))
        
        # Return as 0 (Background) or 255 (Mine)
        return (pred_mask_resized > 0.5).astype('uint8') * 255