import cv2
import time

def test_camera():
    print("Testing primary camera (index 0)...")
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    for backend in backends:
        print(f"Trying backend {backend}...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Success with backend {backend}! Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print(f"❌ Opened but could not read frame with backend {backend}")
        else:
            print(f"❌ Could not open camera with backend {backend}")
        if cap:
            cap.release()
            
    print("\nTesting index 1...")
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Success with index 1! Frame shape: {frame.shape}")
            cap.release()
            return True
    
    print("\n❌ All tests failed.")
    return False

if __name__ == "__main__":
    test_camera()
