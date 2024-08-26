from estimater import *
from datareader import *
import argparse
import pyrealsense2 as rs

from segment_anything import SamPredictor, sam_model_registry

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/mesh/rsbox/rsd435ibox.obj')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--track_refine_iter', type=int, default=2)
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # Obtain both depth and colro streams
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)

    # Start streaming from file
    pipeline.start(config)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    mesh = trimesh.load(args.mesh_file)

    debug = args.debug
    debug_dir = args.debug_dir
    os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # Initialize SAM
    sam = sam_model_registry["vit_h"](checkpoint="../segment-anything/models/sam_vit_h_4b8939.pth")
    sam_predictor = SamPredictor(sam)

    cam_K = np.eye(3)

    try:
        # Streaming loop
        frame_idx = 0
        pose_initialized = False
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

            # Convert images to numpy arrays
            color_image_bgr = np.asanyarray(color_frame.get_data())
            color_image_rgb = cv2.cvtColor(color_image_bgr, cv2.COLOR_BGR2RGB)
            color_H, color_W = color_image_bgr.shape[:2]

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth = cv2.resize(depth_image/1e3, (color_W, color_H), interpolation=cv2.INTER_NEAREST)
            depth[(depth<0.001) | (depth>=np.inf)] = 0

            if frame_idx == 0:
                intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
                cam_K[0, 0] = intrinsics.fx
                cam_K[1, 1] = intrinsics.fy
                cam_K[0, 2] = intrinsics.ppx
                cam_K[1, 2] = intrinsics.ppy
            
            # Initialize the pose first by selecting the object
            if not pose_initialized:
                rx, ry, rw, rh = cv2.selectROI('RealSense', color_image_bgr, False, False, True)
                cv2.destroyAllWindows()
                if rw == 0 or rh == 0:
                    print('Selection canceled')
                    continue
                
                print('Generating mask, please wait')
                roi_box = np.array([int(rx), int(ry), int(rx+rw), int(ry+rh)])
                sam_predictor.set_image(color_image_rgb)
                masks, _, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=roi_box[None, :],
                    multimask_output=False)
                
                h, w = masks[0].shape[-2:]
                mask_image = (255 * masks[0].reshape(h, w)).astype(np.uint8)
                seg_mask = cv2.bitwise_and(color_image_bgr, np.full_like(color_image_bgr, 255), mask=mask_image)
                
                cv2.imshow('Mask', np.hstack((color_image_bgr, seg_mask)))
                print('Press any key to accept, ESC to cancel')
                key = cv2.waitKey(0)
                if key & 0xFF != 27:
                    # Register object to pose estimator
                    print("Registering initial pose")
                    pose = est.register(K=cam_K, rgb=color_image_rgb, depth=depth, ob_mask=masks[0], iteration=args.est_refine_iter)
                    pose_initialized = True
                cv2.destroyAllWindows()
                continue
            
            # Compute traked pose
            pose = est.track_one(rgb=color_image_rgb, depth=depth, K=cam_K, iteration=args.track_refine_iter)

            # Visualize pose
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(cam_K, img=color_image_rgb, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_image_rgb, ob_in_cam=center_pose, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)

            # Show images
            cv2.imshow('RealSense', vis[...,::-1])
            key = cv2.waitKey(1)
            # if pressed escape exit program
            if key & 0xFF == 27:
                cv2.destroyAllWindows()
                break
            
            frame_idx += 1

    finally:
        # Stop streaming
        pipeline.stop()