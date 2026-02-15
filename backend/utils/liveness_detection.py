import torch
import cv2
import numpy as np
from PIL import Image
import os


# Load liveness detection model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "liveness_model.pth")

# Eye cascade for fast eye detection
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


# Simple CNN model definition (must match training model)
class SimpleLivenessNet(torch.nn.Module):
    """Simple CNN for liveness detection."""
    
    def __init__(self):
        super(SimpleLivenessNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 2)  # 2 classes: fake (0), live (1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LivenessDetector:
    def __init__(self, model_path=MODEL_PATH, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = None
        self.use_simple_method = True  # Use simple OpenCV-based liveness detection
        self.load_model(model_path)
        
        # Motion tracking for blink/movement detection
        self.prev_frame_gray = None
        self.motion_history = []
        self.max_motion_history = 10  # Track last 10 frames
        self.blink_threshold = 30.0  # threshold for motion variance
        self.required_blinks = 2  # require at least 2 motion events
        
        # Eye blink detection
        self.blink_history = []  # Track eye closures
        self.max_blink_history = 20  # Track last 20 frames
        self.eye_closed_frames = 0  # Consecutive frames with closed eyes
        self.blinks_detected = 0  # Total blinks in this session
        self.required_blinks_for_liveness = 3  # Need 3+ eye blinks to confirm liveness

    def load_model(self, model_path):
        """Try to load the liveness detection model; fall back to simple method."""
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Liveness model not found at {model_path}; using simple liveness detection")
                return False
            # Recreate model and load state dict
            self.model = SimpleLivenessNet().to(self.device)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.use_simple_method = False
            print(f"✅ Liveness model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load PyTorch model ({e}); using simple OpenCV-based liveness detection")
            self.use_simple_method = True
            return False

    def _simple_liveness_check(self, frame):
        """
        Simple liveness detection using Laplacian variance.
        Still images have lower frequency content than real faces (eyes blinking, skin texture).
        Returns (is_live, confidence)
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Compute Laplacian variance (high = more detailed = live face)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # THRESHOLD for liveness: tune based on your setup
            # Typical values: photos ~100-500, live faces ~500-5000+
            LIVENESS_THRESHOLD = 300.0
            is_live = laplacian_var > LIVENESS_THRESHOLD
            # normalize confidence to [0, 1]
            confidence = min(1.0, laplacian_var / (LIVENESS_THRESHOLD * 2))
            
            return is_live, confidence, laplacian_var
        except Exception as e:
            print(f"⚠️ Simple liveness check error: {e}")
            return True, 0.5, 0

    def _detect_motion(self, frame):
        """
        Detect motion (blinking, head movement) between frames.
        Phone screens and static photos have minimal motion.
        Returns motion magnitude (higher = more motion).
        """
        try:
            # Normalize to fixed size for consistent motion detection
            MOTION_FRAME_SIZE = (64, 64)
            
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Resize to standard size to handle variable ROI dimensions
            gray = cv2.resize(gray, MOTION_FRAME_SIZE)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            
            if not hasattr(self, '_motion_prev_frame') or self._motion_prev_frame is None:
                self._motion_prev_frame = gray.copy()
                return 0.0
            
            # Now both frames are guaranteed to be same size
            # Compute frame difference (optical flow approximation)
            frame_diff = cv2.absdiff(self._motion_prev_frame, gray)
            motion_magnitude = np.mean(frame_diff)
            
            # Update motion history (for motion-based liveness)
            self.motion_history.append(motion_magnitude)
            if len(self.motion_history) > self.max_motion_history:
                self.motion_history.pop(0)
            
            self._motion_prev_frame = gray.copy()
            return motion_magnitude
        except Exception as e:
            print(f"⚠️ Motion detection error: {e}")
            return 0.0

    def _has_sufficient_motion(self):
        """Check if recent frames show sufficient motion (blinks/movement)."""
        if len(self.motion_history) < 5:
            return False  # Need at least 5 frames to detect motion
        
        # Count frames with significant motion
        motion_events = sum(1 for m in self.motion_history if m > self.blink_threshold)
        return motion_events >= self.required_blinks

    def _detect_eye_blinks(self, frame):
        """
        Detect eye blinks by analyzing motion spikes in the face region.
        Natural eye blinking causes motion peaks.
        """
        try:
            # This is called standalone or after _detect_motion
            # Use the motion history already being tracked
            if len(self.motion_history) < 3:
                return True  # Assume open on first few frames
            
            # Analyze recent motion spikes
            recent_motion = self.motion_history[-3:] if len(self.motion_history) >= 3 else self.motion_history
            avg_motion = np.mean(recent_motion)
            
            # Detect motion peaks (spikes) that indicate blinks
            if len(self.motion_history) >= 2:
                recent_avg = np.mean(self.motion_history[:-1]) if len(self.motion_history) > 1 else avg_motion
                current_motion = self.motion_history[-1]
                
                # Spike detection: current motion > 1.3x average = potential blink
                if recent_avg > 0 and current_motion > recent_avg * 1.3 and current_motion > 5:
                    self.blink_history.append(True)  # High motion = blink event
                    
                    # Count distinct blinks: transitions from low to high motion
                    if len(self.blink_history) >= 2 and not self.blink_history[-2]:
                        self.blinks_detected += 1
                else:
                    self.blink_history.append(False)  # Normal motion
            else:
                self.blink_history.append(False)
            
            # Keep history size manageable
            if len(self.blink_history) > self.max_blink_history:
                self.blink_history.pop(0)
            
            return True  # Eyes considered open for liveness check
        except Exception as e:
            return True  # Assume open on error
    
    def has_confirmed_blinks(self):
        """Check if enough blinks have been detected to confirm liveness."""
        return self.blinks_detected >= self.required_blinks_for_liveness

    def is_live(self, frame, face_location=None):
        """
        Detect if a face in the frame is live (not a photo/spoof).
        PRIMARY: Eye blink detection (most reliable)
        SECONDARY: Laplacian variance + PyTorch model
        
        Args:
            frame: BGR image (numpy array)
            face_location: optional (top, right, bottom, left) tuple; if None, process whole frame
        
        Returns:
            (is_live: bool, confidence: float)
        """
        # Extract face ROI if location provided
        if face_location is not None:
            top, right, bottom, left = face_location
            face_roi = frame[max(0, top):min(frame.shape[0], bottom), max(0, left):min(frame.shape[1], right)]
        else:
            face_roi = frame
        
        if face_roi.size == 0:
            return True, 0.5  # empty ROI, assume live
        
        # PRIMARY CHECK: Eye blink detection (most reliable for liveness)
        eyes_visible = self._detect_eye_blinks(face_roi)
        has_confirmed_blinks = self.has_confirmed_blinks()
        
        # If we've already confirmed 3+ blinks, this is definitely a live face
        if has_confirmed_blinks:
            return True, 0.95  # High confidence after confirmed blinks
        
        # Check motion for additional confirmation
        motion = self._detect_motion(face_roi)
        has_motion = self._has_sufficient_motion()
        
        # Run simple liveness check as fallback
        simple_is_live, simple_conf, _ = self._simple_liveness_check(face_roi)
        
        # EARLY FRAMES (first 10 frames): Accept texture-based liveness while waiting for blinks
        if len(self.motion_history) < 10:
            if eyes_visible:
                return True, 0.7  # Eyes visible = probably live
            if simple_is_live and simple_conf > 0.3:
                return True, max(simple_conf, 0.6)
            # Try PyTorch if texture method fails
            if self.model is not None and not self.use_simple_method:
                try:
                    img_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    img_tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                    img_tensor = img_tensor.to(self.device)
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        pytorch_conf = torch.softmax(output, dim=1)[0, 1].item()
                        if pytorch_conf >= 0.4:
                            return True, pytorch_conf
                except:
                    pass
            return simple_is_live, simple_conf
        
        # LATER FRAMES (>= 10 frames): Require eye blinks OR motion + texture
        # This prevents static phone screens from being accepted
        
        # Check for eye blinks in progress
        if eyes_visible and len(self.blink_history) > 0:
            # Eyes visible with motion = likely real
            if has_motion or simple_is_live:
                return True, 0.8
        
        # Check for motion + texture combination
        if has_motion and simple_is_live and simple_conf > 0.3:
            return True, max(simple_conf, 0.7)
        
        # Try PyTorch model as final check
        if self.model is not None and not self.use_simple_method:
            try:
                img_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                img_tensor = torch.from_numpy(np.array(img_pil)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                img_tensor = torch.nn.functional.interpolate(img_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                img_tensor = img_tensor.to(self.device)
                with torch.no_grad():
                    output = self.model(img_tensor)
                    pytorch_conf = torch.softmax(output, dim=1)[0, 1].item()
                    if has_motion and pytorch_conf >= 0.4:
                        return True, pytorch_conf
            except:
                pass
        
        # No sufficient evidence after 10+ frames → likely spoof
        return False, 0.2


# Singleton instance
_detector = None


def get_detector(threshold=0.5):
    """Get or create a singleton liveness detector."""
    global _detector
    if _detector is None:
        _detector = LivenessDetector(threshold=threshold)
    return _detector
