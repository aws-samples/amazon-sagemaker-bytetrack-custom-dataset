import argparse
import os
import os.path as osp
import time
import cv2
import torch
import subprocess
import sys
sys.path.append('/ByteTrack')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

import glob

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "--demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.7, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--num_classes", default=1, type=int, help="num of classes")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    
    parser.add_argument("--test_h", type=int, default=800, help="height for test")
    parser.add_argument("--test_w", type=int, default=1440, help="width for test")
    
    parser.add_argument("--interval", type=int, default=2, help="interval for saving frame")
    
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    return parser

class Predictor(object):
    def __init__(
        self,
        model,
        args,
        test_size,
        trt_file=None,
        decoder=None
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = args.num_classes
        self.confthre = args.conf
        self.nmsthre = args.nms
        self.test_size = test_size
        self.device = args.device
        self.fp16 = args.fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, test_size[0], test_size[1]), device=self.device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def image_demo(predictor, vis_folder, current_time, args, data_dir, save_mot=True):
    files = get_image_list(data_dir)
    files.sort()
    
    tracker = BYTETracker(
        frame_rate=30,
        track_thresh=0.5,
        track_buffer=30,
        mot20=False,
        match_thresh=0.8
    )
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], predictor.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                print(tlwh)
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if True:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if save_mot:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def imageflow_demo(predictor, vis_folder, current_time, args, video_path, save_mscoco=True):
    cap = cv2.VideoCapture(video_path)
    
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, video_path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
        
    
    video_name = video_path.split("/")[-1]
        
    # Directory setting as MOT challenge format
    mot_dir = osp.join(save_folder, 'mot', video_name)
        
    mot_gt_dir = osp.join(mot_dir, 'gt')
    os.makedirs(mot_gt_dir, exist_ok=True)
        
    mot_det_dir = osp.join(mot_dir, 'det')
    os.makedirs(mot_det_dir, exist_ok=True)
        
    img_dir = osp.join(mot_dir, 'img1')
    os.makedirs(img_dir, exist_ok=True)
    
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(
        frame_rate=30,
        track_thresh=0.5,
        track_buffer=30,
        mot20=False,
        match_thresh=0.8
    )
    
    timer = Timer()
    frame_id = 0
    results_gt = []
    results_det = []
    
    frame_gt_no = 0
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(
                    outputs[0],
                    [img_info['height'], img_info['width']],
                    predictor.test_size
                )
                
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        if frame_id % args.interval == 0:
                            results_gt.append(
                                f"{frame_gt_no},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},1,-1\n"
                            )
                            results_det.append(
                                f"{frame_gt_no},-1,{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},1,-1\n"
                            )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            vid_writer.write(online_im)
        else:
            break
        # Save frames
        if frame_id % args.interval == 0:
            img_path = osp.join(img_dir, f'frame_{frame_gt_no:04}.jpeg')
            cv2.imwrite(img_path, frame)
            frame_gt_no += 1
        frame_id += 1

    if save_mscoco:
        res_file = osp.join(mot_gt_dir, "gt.txt")
        with open(res_file, 'w') as f:
            f.writelines(results_gt)
        logger.info(f"save gt to {res_file}")

        res_file = osp.join(mot_det_dir, "det.txt")
        with open(res_file, 'w') as f:
            f.writelines(results_det)
        logger.info(f"save det to {res_file}")
        
    
def load_model(model_dir, device):
    try:
        model = torch.load(f"{model_dir}/model.pth", map_location=device)
    except Execept as e:
        print(str(e))
    model.eval()
    return model

def main(args):
    
    output_dir = "/opt/ml/processing/output"
    #output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)
    
    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))
    
    test_size = None
    if args.test_h is not None:
        test_size = (args.test_h, args.test_w)
    
    model_dir = "/opt/ml/processing/model"
    
    subprocess.run(f"tar -xf {model_dir}/model.tar.gz -C {model_dir}", shell=True)
    model = load_model(model_dir, args.device)

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    
    predictor = Predictor(model, args, test_size, decoder)
    
    current_time = time.localtime()
    
    input_dir = "/opt/ml/processing/input"
    #input_dir = "./input"
    if args.demo == "image":
        seq_list = os.listdir(input_dir)
        seq_list = [f for f in seq_list if os.path.isdir(os.path.join(input_dir, f))]
        for seq_name in seq_list:
            seq_dir = os.path.join(input_dir, seq_name)
            image_demo(predictor, vis_folder, current_time, args, seq_dir)
    elif args.demo == "video" or args.demo == "webcam":
        data_search_path = os.path.join(input_dir, "*.mp4")
        video_list = glob.glob(data_search_path)
        for video_path in video_list:
            print(f'video_path: {video_path}')
            imageflow_demo(predictor, vis_folder, current_time, args, video_path)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)