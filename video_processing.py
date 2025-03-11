import cv2
import numpy as np
import os

class VideoProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.video_capture = None
        self.video_writer = None

    def open_video(self):
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"The input video file does not exist: {self.input_path}")
        self.video_capture = cv2.VideoCapture(self.input_path)

    def create_output_writer(self):
        frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))

        four_cc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.output_path, four_cc, fps, (frame_width, frame_height))

    def apply_filter(self, frame):
        # Simple Gaussian blur filter
        return cv2.GaussianBlur(frame, (15, 15), 0)

    def process_video(self):
        self.open_video()
        self.create_output_writer()

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            filtered_frame = self.apply_filter(frame)
            self.video_writer.write(filtered_frame)

        self.release_resources()

    def release_resources(self):
        if self.video_capture:
            self.video_capture.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()

def main():
    input_video_path = 'input_video.mp4'
    output_video_path = 'output_video.avi'

    video_processor = VideoProcessor(input_video_path, output_video_path)
    video_processor.process_video()
    print("Video processing complete!")

if __name__ == '__main__':
    main()

def create_directory_structure(base_dir='video_processing'):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'output'), exist_ok=True)

def copy_video_to_input(source, destination):
    import shutil
    destination_path = os.path.join(destination, os.path.basename(source))
    shutil.copy(source, destination_path)
    return destination_path

def generate_report(input_video_path, output_video_path):
    import datetime
    now = datetime.datetime.now()
    report = f"Video Processing Report\n"
    report += f"Input Video: {input_video_path}\n"
    report += f"Output Video: {output_video_path}\n"
    report += f"Processing Date: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"

    with open('processing_report.txt', 'w') as report_file:
        report_file.write(report)

def add_logging():
    import logging
    logging.basicConfig(filename='video_processing.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def main_with_logging():
    add_logging()
    logging.info('Starting video processing...')
    
    create_directory_structure()
    input_video_path = 'input/input_video.mp4'
    output_video_path = 'output/output_video.avi'
    
    try:
        video_processor = VideoProcessor(input_video_path, output_video_path)
        video_processor.process_video()
        generate_report(input_video_path, output_video_path)
        logging.info('Video processing complete!')
    except Exception as e:
        logging.error(f'Error during video processing: {e}')

if __name__ == '__main__':
    main_with_logging()

def perform_edge_detection(frame):
    return cv2.Canny(frame, 100, 200)

def save_frames_as_images(video_path, output_folder):
    video_capture = cv2.VideoCapture(video_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count:04d}.jpg'), frame)
        frame_count += 1

    video_capture.release()

def play_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

def resize_video(input_video_path, output_video_path, scale_percent=50):
    video_capture = cv2.VideoCapture(input_video_path)
    
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(output_video_path, four_cc, fps, (new_width, new_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (new_width, new_height))
        video_writer.write(resized_frame)

    video_capture.release()
    video_writer.release()

def merge_videos(video_list, output_video_path):
    four_cc = cv2.VideoWriter_fourcc(*'XVID')
    video_writers = []
    video_captures = []
    
    for video_path in video_list:
        video_capture = cv2.VideoCapture(video_path)
        video_captures.append(video_capture)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        writer = cv2.VideoWriter(output_video_path, four_cc, fps, (frame_width, frame_height))
        video_writers.append(writer)

    while True:
        frames = []
        for capture in video_captures:
            ret, frame = capture.read()
            frames.append(frame if ret else None)

        if all(frame is None for frame in frames):
            break

        for i, frame in enumerate(frames):
            if frame is not None:
                video_writers[i].write(frame)

    for capture, writer in zip(video_captures, video_writers):
        capture.release()
        writer.release()

def main_video_utilities():
    # Copy video to input directory
    source_video = 'source_video.mp4'
    input_directory = 'video_processing/input'
    video_path = copy_video_to_input(source_video, input_directory)

    # Process the video by applying a filter
    output_processed_video = 'video_processing/output/processed_video.avi'
    video_processor = VideoProcessor(video_path, output_processed_video)
    video_processor.process_video()

    # Save frames as images
    save_frames_as_images(video_path, 'video_processing/output/frames')

    # Play processed video
    play_video(output_processed_video)

    # Resize video
    resized_video_path = 'video_processing/output/resized_video.avi'
    resize_video(video_path, resized_video_path, scale_percent=50)

    # Merge videos example
    merge_video_list = [output_processed_video, resized_video_path]
    merged_video_path = 'video_processing/output/merged_video.avi'
    merge_videos(merge_video_list, merged_video_path)

if __name__ == '__main__':
    main_video_utilities()