import os
import sys
import numpy as np
from PIL import Image

# Import the existing visualization script
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from visualize_evaluation import run_visualization

def generate_mock_image(output_path):
    # Create a 256x256 image with a gray background
    img_array = np.ones((256, 256, 3), dtype=np.uint8) * 128
    
    # Draw some "road lines"
    img_array[:, 120:128] = [255, 255, 255]
    img_array[:, 128:136] = [255, 200, 0]
    
    # Draw some "cars"
    img_array[50:90, 80:110] = [200, 50, 50]   # Red car
    img_array[150:190, 150:180] = [50, 50, 200] # Blue car
    
    img = Image.fromarray(img_array)
    img.save(output_path)
    print(f"Mock image saved to {output_path}")

if __name__ == '__main__':
    mock_img_path = os.path.join(os.path.dirname(__file__), 'mock_bev_input.png')
    generate_mock_image(mock_img_path)
    
    output_vis_path = os.path.join(os.path.dirname(__file__), '..', 'manuscript', 'figures', 'crafx_visualization.png')
    print(f"Running visual evaluation with output: {output_vis_path}")
    
    run_visualization(mock_img_path, output_vis_path)
    print("Mock visualization pipeline completed.")
