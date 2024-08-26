# First import library
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
import shutil

def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        os.stat(path)
    except (OSError, ValueError):
        return False
    return True

def make_clean_folder(path_folder):
    if not exists(path_folder):
        os.makedirs(path_folder)
    else:
        user_input = input("%s not empty. Overwrite? (y/n) : " % path_folder)
        if user_input.lower() == 'y':
            shutil.rmtree(path_folder)
            os.makedirs(path_folder)
        else:
            exit()

if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                    Remember to change the stream fps and format to match the recorded.")
    # Add argument which takes path to a bag file as an input
    parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
    parser.add_argument("-o", "--output", type=str, help="Path to output images directory")
    # Parse the command line arguments to an object
    args = parser.parse_args()

    # Safety if no parameter have been given
    if not args.input:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    # Check if the given file have bag extension
    if os.path.splitext(args.input)[1] != ".bag":
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()

    if not args.output:
        print("No output paramater have been given.")
        print("For help type --help")
        exit()
    else:
        path_output = args.output
        path_depth = os.path.join(args.output, "depth")
        path_color = os.path.join(args.output, "rgb")

        os.makedirs(path_output, exist_ok=True)
        os.makedirs(path_depth, exist_ok=True)
        os.makedirs(path_color, exist_ok=True)

    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, args.input, repeat_playback=False)

        # Start streaming from file
        pipeline.start(config)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        align = rs.align(align_to)

        # Streaming loop
        frame_idx = 0
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            if frame_idx == 0:
                intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                cam_K = np.eye(3)
                cam_K[0, 0] = intrinsics.fx
                cam_K[1, 1] = intrinsics.fy
                cam_K[0, 2] = intrinsics.ppx
                cam_K[1, 2] = intrinsics.ppy
                np.savetxt(os.path.join(args.output, "cam_K.txt"), cam_K)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key == 27:
                cv2.destroyAllWindows()
                break
            
            cv2.imwrite(os.path.join(path_color, f"{frame_idx:05d}.png"), color_image)
            cv2.imwrite(os.path.join(path_depth, f"{frame_idx:05d}.png"), depth_image)
            frame_idx += 1

    finally:
        # Stop streaming
        pipeline.stop()