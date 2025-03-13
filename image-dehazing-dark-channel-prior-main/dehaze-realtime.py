import cv2
import numpy as np

def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    min_img = np.minimum(np.minimum(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.erode(min_img, kernel)

def AtmosphereLight(im, dark):
    h, w = im.shape[:2]
    num_pixels = int(h * w * 0.001)
    dark_vec = dark.reshape(h * w)
    img_vec = im.reshape(h * w, 3)
    indices = np.argsort(dark_vec)[-num_pixels:]
    return np.mean(img_vec[indices], axis=0).reshape((1, 3))

def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    return 1 - omega * DarkChannel(im3, sz)

def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return cv2.boxFilter(a, cv2.CV_64F, (r, r)) * im + cv2.boxFilter(b, cv2.CV_64F, (r, r))

def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255
    return Guidedfilter(gray, et, 10, 0.0001)

def Recover(im, t, A, tx=0.1):
    t = np.maximum(t, tx)
    res = np.empty(im.shape, im.dtype)
    for ind in range(3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return np.clip(res, 0, 1)

def dehaze_frame(frame):
    Id = frame.astype('float64') / 255
    dark = DarkChannel(Id, 10)
    A = AtmosphereLight(Id, dark)
    te = TransmissionEstimate(Id, A, 20)
    t = TransmissionRefine(frame, te)
    J = Recover(Id, t, A, 0.1)
    return (J * 255).astype(np.uint8)

def main():
    ip_camera_url = "http://192.0.0.4:8080/video"  # Replace with your actual IP Webcam URL
    cap = cv2.VideoCapture(ip_camera_url)
    
    if not cap.isOpened():
        print("Error: Could not connect to IP Webcam.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            break
        
        dehazed_frame = dehaze_frame(cv2.resize(frame, (840, 640)))
        cv2.imshow('Dehazed Video', cv2.resize(dehazed_frame, (848, 480)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
