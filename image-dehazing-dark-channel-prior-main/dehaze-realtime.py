import cv2
import numpy as np
from numba import jit, prange
import threading
import time

# Optimize the dark channel calculation with Numba
@jit(nopython=True, parallel=True)
def fast_dark_channel(im, patch_size):
    h, w = im.shape[:2]
    pad = patch_size // 2
    dark_channel = np.zeros((h, w), dtype=np.float32)
    
    for i in prange(h):
        for j in range(w):
            min_val = 1.0
            for k in range(3):  # RGB channels
                for di in range(max(0, i-pad), min(h, i+pad+1)):
                    for dj in range(max(0, j-pad), min(w, j+pad+1)):
                        min_val = min(min_val, im[di, dj, k])
            dark_channel[i, j] = min_val
    
    return dark_channel

# Optimize atmosphere light estimation
@jit(nopython=True)
def fast_atmosphere_light(im, dark):
    h, w = im.shape[:2]
    num_pixels = int(h * w * 0.001)
    
    # Flatten arrays
    dark_vec = dark.flatten()
    indices = np.argsort(dark_vec)[-num_pixels:]
    
    # Initialize for brightness values
    brightest_vals = np.zeros(3, dtype=np.float32)
    
    # Use brightest pixels in dark channel to estimate atmosphere light
    for idx in indices:
        i, j = idx // w, idx % w
        for k in range(3):
            brightest_vals[k] += im[i, j, k]
    
    return brightest_vals / num_pixels

# Optimize transmission estimation
@jit(nopython=True, parallel=True)
def fast_transmission_estimate(im, A, patch_size, omega=0.95):
    h, w = im.shape[:2]
    im_normalized = np.empty_like(im)
    
    # Normalize by atmospheric light
    for k in range(3):
        if A[k] > 0.001:  # Avoid division by zero
            im_normalized[:, :, k] = im[:, :, k] / A[k]
        else:
            im_normalized[:, :, k] = im[:, :, k]
    
    # Calculate dark channel of normalized image
    dark = fast_dark_channel(im_normalized, patch_size)
    
    # Calculate transmission
    return 1.0 - omega * dark

# Simplified guided filter for transmission refinement
def simplified_guided_filter(guide, src, radius, eps):
    mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_p = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))
    
    return mean_a * guide + mean_b

# Optimize image recovery
@jit(nopython=True, parallel=True)
def fast_recover(im, t, A, tx=0.1):
    h, w = im.shape[:2]
    res = np.empty_like(im)
    
    for i in prange(h):
        for j in range(w):
            t_val = max(t[i, j], tx)
            for k in range(3):
                res[i, j, k] = (im[i, j, k] - A[k]) / t_val + A[k]
                # Clipping values
                if res[i, j, k] > 1.0:
                    res[i, j, k] = 1.0
                elif res[i, j, k] < 0.0:
                    res[i, j, k] = 0.0
    
    return res

class ProcessingThread(threading.Thread):
    def __init__(self, frame):
        threading.Thread.__init__(self)
        self.frame = frame.copy()
        self.result = None
        self.completed = False
        
    def run(self):
        try:
            # Resize for faster processing
            small_frame = cv2.resize(self.frame, (426, 240))
            
            # Process at lower resolution
            Id = small_frame.astype('float32') / 255.0
            
            # Calculate dark channel
            dark = fast_dark_channel(Id, 5)  # Reduced patch size
            
            # Calculate atmosphere light
            A = fast_atmosphere_light(Id, dark)
            
            # Estimate transmission
            te = fast_transmission_estimate(Id, A, 10)  # Reduced patch size
            
            # Convert to grayscale for guided filter
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY).astype('float32') / 255.0
            
            # Refine transmission
            t = simplified_guided_filter(gray, te, 5, 0.001)  # Reduced radius
            
            # Recover the scene
            J = fast_recover(Id, t, A, 0.1)
            
            # Convert back to uint8
            result = (J * 255).astype(np.uint8)
            
            # Resize back to display resolution
            self.result = cv2.resize(result, (848, 480))
            self.completed = True
        except Exception as e:
            print(f"Error in processing thread: {e}")
            self.completed = True

def main():
    # Camera setup
    ip_camera_url = "http://192.0.0.4:8080/video"  # Replace with your actual IP Webcam URL
    cap = cv2.VideoCapture(ip_camera_url)
    
    if not cap.isOpened():
        print("Error: Could not connect to IP Webcam.")
        return
    
    # Set lower resolution for faster acquisition
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize variables
    processing_thread = None
    last_frame = None
    display_frame = None
    skip_count = 0
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    print("Starting dehazing process. Press 'q' to quit.")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            break
        
        # Store the last successfully captured frame
        last_frame = frame.copy()
        
        # Process every other frame to improve performance
        skip_count += 1
        if skip_count % 2 == 0:
            continue
        
        # Start a new processing thread if the previous one is done
        if processing_thread is None or processing_thread.completed:
            if processing_thread is not None and processing_thread.result is not None:
                display_frame = processing_thread.result
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 10:
                    current_time = time.time()
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_start_time = current_time
                    fps_counter = 0
            
            # Start new thread
            processing_thread = ProcessingThread(last_frame)
            processing_thread.start()
        
        # Display the most recent completed frame
        if display_frame is not None:
            # Add FPS counter
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Dehazed Video', display_frame)
        else:
            # If no processed frame is available yet, show original
            cv2.putText(last_frame, "Processing...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Dehazed Video', last_frame)
        
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    if processing_thread is not None and processing_thread.is_alive():
        processing_thread.join(timeout=1.0)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()