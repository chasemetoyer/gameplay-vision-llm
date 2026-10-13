import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from realtime_inference import process_video, answer_query
from agent_core.qwen_reasoning_core import ProjectorBank
from config.presets import load_preset

VIDEO = "data/videos/clip.mp4"
QUESTION = "When does the phase transition begin?"
TRAINED_WEIGHTS = "outputs/projector_weights.pt"

preset = load_preset("standard")
loop = process_video(
    video_path=VIDEO,
    device="cuda",
    fps=preset.inference.fps,
    use_sam=preset.perception.use_sam,
    projector_weights=TRAINED_WEIGHTS,
    lora_path="outputs/lora_adapter",
    use_cache=True,
    cache_dir="data/cache",
    preset_config=preset,
)
loop.reasoning_core.enable_web_search = False

def ask(loop, question, seed=0):
    loop.reasoning_core.clear_conversation()
    torch.manual_seed(seed)
    return answer_query(loop, question)

answer_trained = ask(loop, QUESTION)

loop.projectors = ProjectorBank(loop.config).to("cuda")
answer_random = ask(loop, QUESTION)
answer_other = ask(loop, "What does the boss look like")

print("TRAINED: ",answer_trained)
print("RANDOM: ", answer_random)
print("OTHER: ", answer_other)

print("is answer trained and answer random the same?", answer_other == answer_random)