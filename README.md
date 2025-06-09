# seg_ros2_wrapper

This ROS 2 package is used to preform semantic segmentation on the ROS 2 camera stream. 

## Functions


Loads a PyTorch model from disk.
Subscribes to the image topic.  
Runs semantic segmentation. 

Generates:

The original image.
A segmentation map (color-coded by risk).
An overlay with segmentation + a circle over the safest region.

Publishes those 3 images to ROS 2 topics:
* `/image_topic`
* `/segmentation_topic`
* `/overlay_topic`
