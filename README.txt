This ROS 2 node performs semantic segmentation on static images located in a local folder. It:

    Loads a PyTorch model from disk.

    Waits for a trigger (std_msgs/String on /trigger_topic).

    When triggered:

        Picks a random image from a specified folder.

        Runs segmentation to classify each pixel into a class.

        Converts class labels to risk levels.

        Generates:

            The original image.

            A segmentation map (color-coded by risk).

            An overlay with segmentation + a circle over the safest region.

        Publishes those 3 images to ROS 2 topics:

            /image_topic

            /segmentation_topic

            /overlay_topic
