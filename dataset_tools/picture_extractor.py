import cv2
import os

DATASET_PATH = "dataset/data"
FRAMES_PER_VIDEO = 20


def extract_even_frames(video_path, output_folder, num_frames, prefix):
    """
    Extract evenly spaced frames from a video file.

    Args:
        video_path (str): Path to input video file
        output_folder (str): Folder where extracted images will be saved
        num_frames (int): Number of frames to extract
        prefix (str): Prefix for output filenames
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Error: Cannot read video {video_path}")
        return

    step = max(1, total_frames // num_frames)
    saved = 0

    for i in range(num_frames):
        frame_id = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()
        if not ret:
            continue

        filename = os.path.join(output_folder, f"{prefix}_{saved}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    cap.release()
    print(f"{video_path} -> {saved} images extracted")


def process_all_breeds():
    """
    Processes all breed folders inside dataset directory.

    Expected structure:
    data/
        breed_name/
            video/
                *.mp4, *.avi, ...
            photo/
    """

    for breed in os.listdir(DATASET_PATH):
        breed_path = os.path.join(DATASET_PATH, breed)

        video_folder = os.path.join(breed_path, "video")
        photo_folder = os.path.join(breed_path, "photo")

        if not os.path.isdir(video_folder):
            continue

        for file in os.listdir(video_folder):
            if file.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(video_folder, file)
                prefix = os.path.splitext(file)[0]

                extract_even_frames(
                    video_path,
                    photo_folder,
                    FRAMES_PER_VIDEO,
                    prefix
                )


if __name__ == "__main__":
    process_all_breeds()