#!/usr/bin/env python3

import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from PIL import Image as PILImage
from cv_bridge import CvBridge
import cv2
import torch
from torchvision import transforms
from ament_index_python.packages import get_package_share_directory

# Risk
risk_map = {
    0: 3, 1: 0, 2: 1, 3: 0, 4: 0, 5: 3, 6: 2, 7: 3, 8: 2, 9: 2, 10: 2,
    11: 2, 12: 2, 13: 3, 14: 3, 15: 4, 16: 4, 17: 1, 18: 3, 19: 3,
    20: 3, 21: 1, 22: 3, 23: 3,
}

risk_color_map = {
    0: (0, 100, 0),
    1: (144, 238, 144),
    2: (255, 255, 0),
    3: (255, 165, 0),
    4: (255, 0, 0),
}

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        self.image_subscription = self.create_subscription(
            Image, '/world/risk/model/x500_depth_0/link/camera_link/sensor/IMX214/image', self.image_callback, 10)
        self.image_publisher = self.create_publisher(Image, 'image_topic', 10)
        self.segmentation_publisher = self.create_publisher(Image, 'segmentation_topic', 10)
        self.overlay_publisher = self.create_publisher(Image, 'overlay_topic', 10)
        self.bridge = CvBridge()

        self.get_logger().info('Node has started.')

        #self.package_path = get_package_share_directory('image')
        model_path = '/root/uav_ws/src/segmentation-node/models/best_model_MobileNet.pt'

        try:
            if torch.cuda.is_available():
                self.exec_type = 'cuda'
            else: 
                self.exec_type = 'cpu'
            self.model = torch.load(model_path, map_location=torch.device('{}'.format(self.exec_type)), weights_only=False)
            self.model.eval()
            self.model_loaded = True
            self.get_logger().info('Model loaded successfully.')
        except Exception as e:
            self.model_loaded = False
            self.get_logger().error(f"Failed to load model: {e}")

        self.preprocess = transforms.Compose([
            transforms.Resize([520, 520]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def image_callback(self, msg):
        if not self.model_loaded:
            self.get_logger().warn("Model not loaded. Skipping image.")
            return

        self.get_logger().info('Received image.')

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_pil = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            if self.exec_type == 'cuda':
                img_t = self.preprocess(img_pil).unsqueeze(0).cuda()
            else:
                img_t = self.preprocess(img_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(img_t)
                output_tensor = outputs['out']

            segmented_image = draw_segmentation_map(output_tensor)
            risk_map_gray = generate_risk_map_from_labels(output_tensor)
            overlayed_image = image_overlay(cv_image, segmented_image)
            overlayed_with_circle, best_center_scaled, best_center_unscaled = highlight_low_risk_zone(risk_map_gray, overlayed_image)

            if best_center_scaled and best_center_unscaled:
                self.get_logger().info(f"Best center (risk map): {best_center_unscaled}")
                self.get_logger().info(f"Best center (scaled to image): {best_center_scaled}")

            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.segmentation_publisher.publish(self.bridge.cv2_to_imgmsg(segmented_image, "bgr8"))
            self.overlay_publisher.publish(self.bridge.cv2_to_imgmsg(overlayed_with_circle, "bgr8"))

            self.get_logger().info("Published segmented images.")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")


def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze().cpu(), dim=0).numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num, risk_level in risk_map.items():
        idx = labels == label_num
        b, g, r = risk_color_map[risk_level]
        red_map[idx], green_map[idx], blue_map[idx] = r, g, b

    return np.stack([red_map, green_map, blue_map], axis=2)

def generate_risk_map_from_labels(outputs):
    labels = torch.argmax(outputs.squeeze().cpu(), dim=0).numpy()
    risk_map_gray = np.zeros_like(labels).astype(np.uint8)
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
        return original_image, (real_x, real_y), best_center
    return original_image, None, None

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


