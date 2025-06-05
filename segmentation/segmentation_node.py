#!/usr/bin/env python3

import os
import random
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from PIL import Image as PILImage
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms

# Risk mapping definitions
risk_map = {
    0: 3, 1: 0, 2: 1, 3: 0, 4: 0, 5: 3, 6: 2, 7: 3, 8: 2, 9: 2,
    10: 2, 11: 2, 12: 2, 13: 3, 14: 3, 15: 4, 16: 4, 17: 1,
    18: 3, 19: 3, 20: 3, 21: 1, 22: 3, 23: 3,
}
risk_color_map = {
    0: (0, 100, 0),       # Very Safe
    1: (144, 238, 144),   # Safe
    2: (255, 255, 0),     # Moderate
    3: (255, 165, 0),     # Risky
    4: (255, 0, 0),       # Dangerous
}

class ImagePublisherSubscriber(Node):
    def __init__(self):
        super().__init__('image_publisher_subscriber')

        self.subscription = self.create_subscription(
            String, 'trigger_topic', self.listener_callback, 10)
        self.image_publisher = self.create_publisher(Image, 'image_topic', 10)
        self.segmentation_publisher = self.create_publisher(Image, 'segmentation_topic', 10)
        self.overlay_publisher = self.create_publisher(Image, 'overlay_topic', 10)
        self.bridge = CvBridge()

        self.get_logger().info('Node has started.')

        # Paths
        self.image_directory = '/path/to/pictures'
        self.model_path = '/path/to/the/model.pt'

        self.image_files = [os.path.join(self.image_directory, f)
                            for f in os.listdir(self.image_directory)
                            if f.lower().endswith(('.jpg', '.png'))]

        if not self.image_files:
            self.get_logger().warn(f"No images found in {self.image_directory}.")

        self.model = torch.load(self.model_path, map_location=torch.device('cpu'))
        self.model.eval()
        self.get_logger().info('Model loaded successfully.')

        self.preprocess = transforms.Compose([
            transforms.Resize([520, 520]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def listener_callback(self, msg):
        self.get_logger().info(f'Received trigger: "{msg.data}"')

        if not self.image_files:
            self.get_logger().error('No images to process.')
            return

        image_path = random.choice(self.image_files)
        cv_image = cv2.imread(image_path)

        if cv_image is None:
            self.get_logger().error(f'Failed to read image from {image_path}')
            return

        try:
            img_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            img_t = self.preprocess(img_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(img_t)
                output_tensor = outputs['out']

            segmented_image = draw_segmentation_map(output_tensor)
            risk_map_gray = generate_risk_map_from_labels(output_tensor)
            overlayed_image = image_overlay(cv_image, segmented_image)
            overlayed_with_circle = highlight_low_risk_zone(risk_map_gray, overlayed_image)

            # Convert to ROS messages
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            ros_segmented = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
            ros_overlayed = self.bridge.cv2_to_imgmsg(overlayed_with_circle, "bgr8")

            # Publish all images
            self.image_publisher.publish(ros_image)
            self.segmentation_publisher.publish(ros_segmented)
            self.overlay_publisher.publish(ros_overlayed)

            self.get_logger().info('Images published successfully.')

        except Exception as e:
            self.get_logger().error(f'Error during image processing: {e}')



def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).numpy()
    red_map, green_map, blue_map = (np.zeros_like(labels, dtype=np.uint8) for _ in range(3))

    for label_num, risk_level in risk_map.items():
        idx = labels == label_num
        b, g, r = risk_color_map[risk_level]
        red_map[idx], green_map[idx], blue_map[idx] = r, g, b

    return np.stack([red_map, green_map, blue_map], axis=2)

def generate_risk_map_from_labels(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).numpy()
    risk_map_gray = np.zeros_like(labels, dtype=np.uint8)
    for class_id, risk_level in risk_map.items():
        risk_map_gray[labels == class_id] = risk_level
    return risk_map_gray

def image_overlay(image, segmented_image, alpha=1.0, beta=0.8, gamma=0):
    if image.shape[:2] != segmented_image.shape[:2]:
        segmented_image = cv2.resize(segmented_image, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(image, alpha, segmented_image, beta, gamma)

def highlight_low_risk_zone(risk_map_gray, original_image, block_size=30, margin=100):
    h_risk, w_risk = risk_map_gray.shape
    h_img, w_img, _ = original_image.shape

    min_risk = float('inf')
    best_center = None

    for y in range(0, h_risk - block_size, block_size):
        for x in range(0, w_risk - block_size, block_size):
            y0, y1 = max(0, y - margin), min(h_risk, y + block_size + margin)
            x0, x1 = max(0, x - margin), min(w_risk, x + block_size + margin)
            window = risk_map_gray[y0:y1, x0:x1]
            avg_risk = np.mean(window)
            if avg_risk < min_risk:
                min_risk = avg_risk
                best_center = (x + block_size // 2, y + block_size // 2)

    if best_center:
        real_x = int(best_center[0] * (w_img / w_risk))
        real_y = int(best_center[1] * (h_img / h_risk))
        cv2.circle(original_image, (real_x, real_y), 500, (255, 0, 0), 10)
        cv2.circle(original_image, (real_x, real_y), 5, (255, 0, 0), -1)
    return original_image


def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

