import argparse
import os
import sys
import cv2
import numpy as np
import onnxruntime as ort

class SAM2OpenCVDemo:
    def __init__(self, encoder_path, decoder_path):
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        
        # Verify model files exist
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder ONNX model not found: {self.encoder_path}")
        if not os.path.exists(self.decoder_path):
            raise FileNotFoundError(f"Decoder ONNX model not found: {self.decoder_path}")
            
        print("[INFO] Initializing ONNX Runtime sessions...")
        # Enable available providers (CUDA if available, otherwise CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            self.encoder_session = ort.InferenceSession(self.encoder_path, providers=providers)
            self.decoder_session = ort.InferenceSession(self.decoder_path, providers=providers)
        except Exception as e:
            print(f"[ERROR] Failed to load ONNX models. Ensure .data files are in the same directory.")
            raise e
            
        print("[INFO] Models loaded successfully!")
        
        # Internal state
        self.orig_image = None
        self.disp_image = None
        self.input_tensor = None
        self.img_h = 0
        self.img_w = 0
        
        # Prompt points list: store tuples of (x, y, label)
        # label: 1.0 for positive, 0.0 for negative
        self.points = []

    def create_sample_image(self):
        """Generates a beautiful synthetic image with clear shapes if no image is provided."""
        print("[INFO] Generating sample test image with distinct colored shapes...")
        img = np.ones((600, 800, 3), dtype=np.uint8) * 240
        
        # Draw some distinct objects to segment
        # Red circle
        cv2.circle(img, (200, 200), 80, (0, 0, 220), -1)
        cv2.circle(img, (200, 200), 80, (0, 0, 100), 3)
        
        # Green rectangle
        cv2.rectangle(img, (450, 100), (650, 350), (40, 200, 40), -1)
        cv2.rectangle(img, (450, 100), (650, 350), (20, 100, 20), 3)
        
        # Blue ellipse
        cv2.ellipse(img, (350, 450), (120, 60), 30, 0, 360, (220, 100, 40), -1)
        cv2.ellipse(img, (350, 450), (120, 60), 30, 0, 360, (120, 40, 20), 3)
        
        # Add some text instructions on the image
        cv2.putText(img, "Left Click: Positive Prompt", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.putText(img, "Right Click: Negative Prompt", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.putText(img, "Press 'C': Clear points", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.putText(img, "Press 'ESC' or 'Q': Quit", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        
        return img

    def set_image(self, image, update_gui=True):
        """Preprocesses the image for model inference."""
        self.orig_image = image.copy()
        self.disp_image = image.copy()
        self.img_h, self.img_w = image.shape[:2]
        
        # Aspect-ratio-preserving resize to max side 1024
        self.scale = 1024.0 / max(self.img_h, self.img_w)
        self.new_h = int(self.img_h * self.scale + 0.5)
        self.new_w = int(self.img_w * self.scale + 0.5)
        
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if len(image.shape) == 3 else cv2.COLOR_GRAY2RGB)
        resized_img = cv2.resize(rgb_img, (self.new_w, self.new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to 1024x1024 with zeros
        padded_img = np.zeros((1024, 1024, 3), dtype=np.float32)
        padded_img[:self.new_h, :self.new_w, :] = resized_img.astype(np.float32) / 255.0
        
        # Transpose to CHW
        chw_img = np.transpose(padded_img, (2, 0, 1))
        
        # Add batch dimension
        self.input_tensor = np.expand_dims(chw_img, axis=0)
        self.points = []
        if update_gui:
            self.update_display()

    def run_inference(self):
        """Runs the encoder and decoder to generate a segmentation mask based on current prompts."""
        if not self.points:
            # If no points, just show original image
            self.disp_image = self.orig_image.copy()
            return
            
        # Prepare prompt coordinates and labels
        # Model expects shape [1, 2, 2] for unnorm_coords and [1, 2] for labels
        # Map points to the scaled/padded 1024x1024 space
        mapped_coords = []
        mapped_labels = []
        
        for pt in self.points[-2:]:  # Take up to last 2 points to fit shape [1, 2, 2]
            mapped_coords.append([float(pt[0] * self.scale), float(pt[1] * self.scale)])
            mapped_labels.append(pt[2])
            
        # Pad to exactly 2 points if only 1 point is available
        while len(mapped_coords) < 2:
            mapped_coords.append([0.0, 0.0])
            mapped_labels.append(-1.0)  # -1.0 indicates padding/dummy point
            
        unnorm_coords = np.array([mapped_coords], dtype=np.float32)
        labels = np.array([mapped_labels], dtype=np.float32)
        
        # 1. Run Encoder
        encoder_inputs = {
            "image": self.input_tensor,
            "unnorm_coords": unnorm_coords,
            "labels": labels
        }
        
        encoder_outputs = self.encoder_session.run(None, encoder_inputs)
        # Outputs: image_embeddings, high_res_features1, high_res_features2, sparse_embedding
        img_embeds, hr_feat1, hr_feat2, sparse_embed = encoder_outputs
        
        # 2. Run Decoder
        decoder_inputs = {
            "image_embeddings": img_embeds,
            "high_res_features1": hr_feat1,
            "high_res_features2": hr_feat2,
            "sparse_embedding": sparse_embed
        }
        
        decoder_outputs = self.decoder_session.run(None, decoder_inputs)
        # Outputs: masks [1, 1, 256, 256], scores [1, 1]
        masks, scores = decoder_outputs
        
        # Postprocess Mask
        mask = masks[0, 0]  # shape (256, 256)
        score = scores[0, 0]
        
        # Convert logits to binary mask (< 0.0 maps to foreground object)
        binary_mask = (mask < 0.0).astype(np.uint8)
        
        # Resize mask to 1024x1024 padded space
        full_padded_mask = cv2.resize(binary_mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        # Crop out the valid unpadded region
        valid_mask = full_padded_mask[:self.new_h, :self.new_w]
        
        # Resize valid mask back to original image size
        full_mask = cv2.resize(valid_mask, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        
        # Create visual overlay
        overlay = self.orig_image.copy()
        
        # Gorgeous dynamic semi-transparent overlay (Cyan/Blue tint for mask)
        color = np.array([255, 144, 30], dtype=np.uint8)  # BGR for elegant custom blue
        overlay[full_mask == 1] = overlay[full_mask == 1] * 0.4 + color * 0.6
        
        # Add mask contours for premium finish
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
        
        # Draw confidence score
        cv2.putText(overlay, f"Score: {score:.3f}", (self.img_w - 180, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if score > 0.7 else (0, 165, 255), 2)
                    
        self.disp_image = overlay

    def update_display(self):
        """Redraws the output window with current masks and prompt points."""
        self.run_inference()
        
        # Draw points over the display image
        canvas = self.disp_image.copy()
        for pt in self.points:
            x, y, label = pt[:3]
            if label == 1.0:
                # Green circle with white border for positive point
                cv2.circle(canvas, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(canvas, (x, y), 4, (0, 220, 0), -1)
            else:
                # Red circle with white border for negative point
                cv2.circle(canvas, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(canvas, (x, y), 4, (0, 0, 220), -1)
                
        cv2.imshow("SAM2 Interactive Demo (OpenCV)", canvas)

    def mouse_callback(self, event, x, y, flags, param):
        """Handles mouse click events for selecting positive and negative prompt points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"[ACTION] Positive point added at ({x}, {y})")
            self.points.append((x, y, 1.0))
            if len(self.points) > 2:
                self.points.pop(0)
            self.update_display()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"[ACTION] Negative point added at ({x}, {y})")
            self.points.append((x, y, 0.0))
            if len(self.points) > 2:
                self.points.pop(0)
            self.update_display()

def main():
    parser = argparse.ArgumentParser(description="Run lightweight SAM2 (EdgeTAM) with OpenCV and ONNX Runtime.")
    parser.add_argument("--encoder", type=str, default="encoder.onnx", help="Path to encoder.onnx")
    parser.add_argument("--decoder", type=str, default="decoder.onnx", help="Path to decoder.onnx")
    parser.add_argument("--image", type=str, default="WIN_20260512_12_12_18_Pro.jpg", help="Path to input image.")
    parser.add_argument("--output", type=str, default="", help="Path to save output image directly without launching GUI.")
    args = parser.parse_args()
    
    # Initialize demo app
    try:
        demo = SAM2OpenCVDemo(args.encoder, args.decoder)
    except Exception as e:
        sys.exit(1)
        
    # Input selection
    if not args.output:
        print("\n" + "="*30)
        print(" SELECT INPUT SOURCE:")
        print(" [I] Image (Default: WIN_20260512_12_12_18_Pro.jpg)")
        print(" [W] Webcam")
        print("="*30)
        choice = input("Enter choice (I/W): ").strip().lower()
    else:
        choice = 'i'

    img = None
    if choice == 'w':
        print("[INFO] Starting webcam... Press SPACE to capture frame, or 'Q' to quit.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam.")
            choice = 'i'
        else:
            while True:
                ret, frame = cap.read()
                if not ret: break
                cv2.imshow("Webcam Live Feed - Press SPACE to Capture", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    img = frame
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
            cap.release()
            cv2.destroyWindow("Webcam Live Feed - Press SPACE to Capture")

    if img is None:
        # Load image
        img_path = args.image if os.path.exists(args.image) else "WIN_20260512_12_12_18_Pro.jpg"
        if not os.path.exists(img_path):
            print(f"[WARNING] Image {img_path} not found. Generating sample image.")
            img = demo.create_sample_image()
        else:
            print(f"[INFO] Loading image: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Failed to read image: {img_path}")
                img = demo.create_sample_image()
        
    if args.output:
        # Non-interactive / verification mode
        print(f"[INFO] Running in non-interactive mode. Segmenting object at center...")
        demo.set_image(img, update_gui=False)
        # Add a positive prompt point at the center of the image
        demo.points.append((img.shape[1]//2, img.shape[0]//2, 1.0))
        demo.run_inference()
        cv2.imwrite(args.output, demo.disp_image)
        print(f"[INFO] Output image saved successfully to {args.output}")
        return

    # Setup window and callbacks for interactive mode
    cv2.namedWindow("SAM2 Interactive Demo (OpenCV)", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("SAM2 Interactive Demo (OpenCV)", demo.mouse_callback)
    
    demo.set_image(img)
    
    print("\n" + "="*50)
    print(" Interactive Controls:")
    print(" - Left Click  : Positive Prompt (Include area)")
    print(" - Right Click : Negative Prompt (Exclude area)")
    print(" - Press 'C'   : Clear all points")
    print(" - Press 'ESC' : Exit application")
    print("="*50 + "\n")
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break
        elif key == ord('c') or key == ord('C'):
            print("[ACTION] Cleared prompt points.")
            demo.points = []
            demo.update_display()
            
    cv2.destroyAllWindows()
    print("[INFO] Application exited successfully.")

if __name__ == "__main__":
    main()
