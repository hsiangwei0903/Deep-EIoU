import argparse
import os
import sys
sys.path.append('.')
import numpy as np
from loguru import logger
from tracker.tracking_utils.timer import Timer
from tracker.Deep_EIoU import Deep_EIoU

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("DeepEIoU For Evaluation!")

    parser.add_argument("--root_path", default="/home/hsiangwei/Desktop/sportsmot/", type=str)
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=60, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--nms_thres", type=float, default=0.7, help='nms threshold')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=True, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    return parser

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def image_track(detections, embeddings, sct_output_path, args):

    # Tracker
    tracker = Deep_EIoU(args, frame_rate=30)
    
    results = []
    
    num_frames = len(detections)

    scale = min(1440/1280, 800/720)

    for frame_id,det in enumerate(detections,1):

        det /= scale

        embs = embeddings[frame_id-1]
        
        if det is not None:

            embs = [e[0] for e in embs]
            embs = np.array(embs)

            trackerTimer.tic()
            online_targets = tracker.update(det, embs)
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.last_tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                vertical = False
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()

        else:
            timer.toc()
        if frame_id % 100 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    with open(sct_output_path, 'w') as f:
        f.writelines(results)
    logger.info(f"save SCT results to {sct_output_path}")
    
def main():
    
    args = make_parser().parse_args()
    data_path = args.root_path
    seq_path =  os.path.join(data_path,'detection/')
    os.makedirs(os.path.join(data_path,'SCT'), exist_ok=True)
    
    seqs = os.listdir(seq_path)
    seqs = [path.replace('.npy','') for path in seqs if path.endswith('.npy')]
    seqs.sort()
        
    for seq in seqs:
            
        logger.info('Processing seq {}'.format(seq))
        
        if not os.path.exists(os.path.join(data_path,'detection/','{}.npy'.format(seq))):
            continue

        detections = np.load(os.path.join(data_path,'detection/','{}.npy'.format(seq)),allow_pickle=True)
        
        embeddings = np.load(os.path.join(data_path,'embedding/','{}.npy'.format(seq)),allow_pickle=True)
            
        sct_output_path = os.path.join(data_path,'SCT','{}.txt'.format(seq))
        
        # SCT tracking
        image_track(detections, embeddings, sct_output_path, args)
            
if __name__ == "__main__":
    main()
