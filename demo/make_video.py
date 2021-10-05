import cv2
import os
import argparse

# References: https://blog.csdn.net/bryant_meng/article/details/110079285
# Check test_video.sh in the main directory
# Step1: Run video_to_frame to convert low light video to images
# Step2: Run test.py to convert low light images to enhanced images
# Step3: Run image_to_video to convert enhanced images to enhanced videos


def parse_args():
    parser = argparse.ArgumentParser(description="Video_Demo")
    parser.add_argument('--video_path', type=str, default='demo/Movie/Ari.mp4')
    parser.add_argument("--image_lowlight_folder", type=str, default='data/test_data/Frames/%d.jpg')
    parser.add_argument('--image_folder', type=str, default='data/test_output/Frames/')
    parser.add_argument('--save_path', type=str, default='demo/Movie/Res.mp4')
    parser.add_argument('--choice', type=str, required=True, choices = ['V2I', 'I2V'], default='V2I')

    args = parser.parse_args()
    return args


def cal_frame(args):
    video_cap = cv2.VideoCapture(args.video_path)
    frame_count = 0
    while True:
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1

    print(frame_count)


def cal_fps(args):
    video = cv2.VideoCapture(args.video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    video.release()
    return fps


def video_to_frame(args):
    vidcap = cv2.VideoCapture(args.video_path) # TODO: replace with url like youtube link
    success, image = vidcap.read()

    count = 0
    while success:
        cv2.imwrite(args.image_lowlight_folder % count, image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        print("We have %2d images" % count)


def image_to_video(args):
    img = cv2.imread(args.image_folder + '0.jpg')
    fps = cal_fps(args)
    size = (img.shape[1], img.shape[0])
    print(size)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # For mp4 only
    videoWrite = cv2.VideoWriter(args.save_path, fourcc, fps, size)

    files = os.listdir(args.image_folder)
    out_num = len(files)

    for i in range(0, out_num):
        fileName = args.image_folder + str(i) + '.jpg'
        img = cv2.imread(fileName)
        videoWrite.write(img)

def main(args):
    if args.choice == 'V2I':
        video_to_frame(args)
    elif args.choice == 'I2V':
        image_to_video(args)
    else:
        raise TypeError

if __name__ == "__main__":
    args = parse_args()
    main(args)


