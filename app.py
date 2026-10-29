from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
import uuid
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips

app = FastAPI(title="AI Video Generator API")

# Models for Request
class VideoRequest(BaseModel):
    prompt: str
    aspect_ratio: str
    style: str

# Helper: Extract Token
def get_hf_client(hf_token: str):
    if not hf_token:
        raise HTTPException(status_code=401, detail="Hugging Face Token is missing")
    return InferenceClient(token=hf_token)

# 1. AI Prompt Enhancer
def enhance_prompt(client: InferenceClient, prompt: str) -> str:
    try:
        # Using Zephyr for prompt enhancement
        response = client.text_generation(
            f"<|system|>You are a cinematic video script writer. Make this prompt highly descriptive for an AI video generator.<|user|>{prompt}<|assistant|>",
            model="HuggingFaceH4/zephyr-7b-beta",
            max_new_tokens=100
        )
        return response.strip()
    except Exception as e:
        print(f"Prompt enhancement failed: {e}")
        return prompt # Fallback to original

# 2 & 3. Text to Video Generation (Mocked for Inference API)
def generate_video_clip(client: InferenceClient, scene_prompt: str, clip_path: str):
    try:
        # Note: Free tier video models on Inference API change frequently.
        # Example using a placeholder model name (Zeroscope/similar).
        video_bytes = client.text_to_image(scene_prompt, model="cerspense/zeroscope_v2_576w") 
        # In a real scenario with Inference API supporting video, this returns video bytes.
        with open(clip_path, "wb") as f:
            f.write(video_bytes)
        return clip_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e}")

# 5. Audio Generation
def generate_audio(client: InferenceClient, script: str, audio_path: str):
    try:
        # Using a TTS model like Bark or Suno
        audio_bytes = client.text_to_speech(script, model="suno/bark")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        return audio_path
    except Exception:
        return None # Proceed without audio if it fails

# 6. Cloud Stitching with MoviePy (FFmpeg wrapper)
def stitch_video_and_audio(video_clips: list, audio_path: str, output_path: str):
    try:
        clips = [VideoFileClip(c) for c in video_clips]
        final_video = concatenate_videoclips(clips, method="compose")
        
        if audio_path:
            audio = AudioFileClip(audio_path)
            final_video = final_video.set_audio(audio)
            
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac", fps=24)
        return output_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stitching failed: {e}")

@app.post("/generate")
async def generate_video_endpoint(
    request: VideoRequest, 
    x_hf_token: str = Header(None)
):
    client = get_hf_client(x_hf_token)
    session_id = str(uuid.uuid4())
    os.makedirs(f"/tmp/{session_id}", exist_ok=True)
    
    # Step 1: Enhance Prompt
    enhanced_script = enhance_prompt(client, request.prompt)
    
    # Step 2 & 3: Generate Clips (Simplified to 1 clip for latency safety on free tier)
    clip_path = f"/tmp/{session_id}/clip1.mp4"
    generate_video_clip(client, enhanced_script, clip_path)
    
    # Step 5: Generate Audio
    audio_path = f"/tmp/{session_id}/audio.wav"
    generate_audio(client, enhanced_script, audio_path)
    
    # Step 6: Stitch
    final_output = f"/tmp/{session_id}/final_video.mp4"
    stitch_video_and_audio([clip_path], audio_path, final_output)
    
    # Return file (In production, upload to a bucket or serve static URL)
    return FileResponse(final_output, media_type="video/mp4", filename="ai_video.mp4")

          
