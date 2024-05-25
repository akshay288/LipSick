import subprocess
import os
import warnings
from compute_crop_radius import calculate_crop_radius_statistics
import runpod 
import uuid
import requests

# Suppress specific warnings about video conversion
warnings.filterwarnings("ignore", message="Video does not have browser-compatible container or codec. Converting to mp4")

def get_versioned_filename(filepath):
    base, ext = os.path.splitext(filepath)
    counter = 1
    versioned_filepath = filepath
    while os.path.exists(versioned_filepath):
        versioned_filepath = f"{base}({counter}){ext}"
        counter += 1
    return versioned_filepath

def compute_crop_radius_stats(video_file):
    if video_file is None:
        return "Please upload a video file first."
    print("Computing Crop Radius...")
    _, _, _, most_common = calculate_crop_radius_statistics(video_file.name)
    print(f"Done: Crop radius = {most_common}")
    return most_common

def process_files(source_video, driving_audio, custom_crop_radius=None, auto_mask=True, ref_index_1=None, ref_index_2=None, ref_index_3=None, ref_index_4=None, ref_index_5=None, activate_custom_frames=False):
    ref_indices = [index for index in [ref_index_1, ref_index_2, ref_index_3, ref_index_4, ref_index_5] if index is not None]
    if custom_crop_radius is None or custom_crop_radius == 0:
        ref_indices = [index + 5 for index in ref_indices]
    ref_indices_str = ','.join(map(str, ref_indices)) if len(ref_indices) == 5 else ""

    if not driving_audio:
        print("Please upload audio first.")
        return "", "Error: Audio file is required."

    pretrained_model_path = "./asserts/pretrained_lipsick.pth"
    deepspeech_model_path = "./asserts/output_graph.pb"
    res_video_dir = "./asserts/inference_result"

    base_name = os.path.splitext(os.path.basename(source_video))[0]
    output_video_name = f"{base_name}_lipsick.mp4" if not auto_mask else "LipSick_Blend.mp4"
    output_video_path = os.path.join(res_video_dir, output_video_name)
    output_video_path = get_versioned_filename(output_video_path)

    cmd = [
        'python3', 'inference.py',
        '--source_video_path', source_video,
        '--driving_audio_path', driving_audio,
        '--pretrained_lipsick_path', pretrained_model_path,
        '--deepspeech_model_path', deepspeech_model_path,
        '--res_video_dir', res_video_dir,
        '--custom_reference_frames', ref_indices_str
    ]

    if custom_crop_radius is not None:
        cmd.extend(['--custom_crop_radius', str(custom_crop_radius)])

    if activate_custom_frames:
        cmd.append('--activate_custom_frames')

    if auto_mask:
        cmd.append('--auto_mask')

    try:
        subprocess.run(cmd, check=True)
        print("Complete")
        print("Result saved in ./asserts/inference_results")
        print("Thank you for using LipSick for your Lip-Sync needs.")
        
        # Return the path of the processed video
        return "", output_video_path
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return "An error occurred during processing. Please check the console log.", ""


def upload_file_to_s3(presigned_url, file_path):
    with open(file_path, 'rb') as file:
        response = requests.put(presigned_url, data=file)
        if response.status_code == 200:
            print("File uploaded successfully.")
        else:
            print(f"Failed to upload file. Status code: {response.status_code}, Response: {response.text}")


UPLOAD_FOLDER = '/tmp/vidai-files'
CONFIG_FOLDER = '/tmp/vidai-configs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONFIG_FOLDER, exist_ok=True)


def download_file(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print(f"Video downloaded successfully and saved to {save_path}")
    else:
        raise Exception(f"Failed to download video. HTTP status code: {response.status_code}")


def run_script(job_id, upload_url, video_path, audio_path, avatar_id):
    os.makedirs(f"{UPLOAD_FOLDER}/{job_id}", exist_ok=True)
    os.makedirs(f"{UPLOAD_FOLDER}/avatars/{avatar_id}", exist_ok=True)
    avatar_local_path = f"{UPLOAD_FOLDER}/avatars/{avatar_id}/video.mp4"
    audio_local_path = f"{UPLOAD_FOLDER}/{job_id}/audio.mp3"

    download_file(audio_path, audio_local_path)
    if not os.path.exists(avatar_local_path):
        download_file(video_path, avatar_local_path)

    status, output_path = process_files(avatar_local_path, audio_local_path, 0, True, 0, 0, 0, 0, 0, False)
    print("!!!!!!!!!!")
    print(status)
    print(output_path)
    print("!!!!!!!!!!")

    if os.path.exists(output_path):
        upload_file_to_s3(upload_url, output_path)
    else:
        print("Could not find output path")


def handler(job):
    data = job["input"]

    upload_url = data.get("uploadUrl")
    video_path = data.get("video")
    audio_path = data.get("audio")
    avatar_id = data.get("avatar")

    if not video_path:
        return { "error": "Must supply video path" }
    if not audio_path:
        return { "error": "Must supply audio path" }
    if not avatar_id:
        return { "error": "Must supply avatar id" }
    if not upload_url:
        return { "error": "Must supply upload url" }

    job_id = str(uuid.uuid4())
    print("--------------------------")
    print(job_id)
    print(upload_url)
    print(video_path)
    print(audio_path)
    print(avatar_id)
    print("--------------------------")
    run_script(job_id, upload_url, video_path, audio_path, avatar_id)

    return { "success": True }


runpod.serverless.start({"handler": handler})