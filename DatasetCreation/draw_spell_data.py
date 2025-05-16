import cv2
import numpy as np
import os

# === Configuration ===
SAVE_DIR = "spells_dataset"         # Directory where spell drawings will be saved
LABEL = "close"                     # Label for the current set of drawings ("open" or "close")
IMG_SIZE = 28                       # Target image size (28x28 pixels)

# Create the save directory if it doesn't already exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize a black canvas for drawing (300x300 pixels, grayscale)
canvas = np.zeros((300, 300), dtype=np.uint8)

# Variables for drawing state
drawing = False                     # True when mouse is held down
ix, iy = -1, -1                     # Last cursor position
counter = 0                         # Counts and indexes saved images

# === Mouse Event Callback Function ===
def draw(event, x, y, flags, param):
    global drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing when mouse button is pressed
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw white line (thickness 7) as mouse moves
            cv2.line(canvas, (ix, iy), (x, y), 255, 7)
            ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        # Stop drawing when mouse button is released
        drawing = False

# === Setup Drawing Window ===
cv2.namedWindow("Draw Spell")
cv2.setMouseCallback("Draw Spell", draw)

print("[SPACE] = Save drawing, [C] = Clear, [ESC] = Quit")

# === Main Loop ===
while True:
    cv2.imshow("Draw Spell", canvas)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC key: Exit the loop
        break
    elif key == ord('c'):
        # 'C' key: Clear the canvas
        canvas[:] = 0
    elif key == ord(' '):
        # SPACE key: Save the current drawing
        # Resize to 28x28 and save as PNG with label and counter
        resized = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        filename = f"{LABEL}_{counter}.png"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, resized)
        print(f"Saved: {filename}")
        counter += 1
        canvas[:] = 0  # Clear after saving

# Cleanup OpenCV windows
cv2.destroyAllWindows()
