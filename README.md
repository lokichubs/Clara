# Clara
AI tool to improve physical therapy by using computer vision to provide personalized instructions and connecting patients with the physical therapist.

## Pitch Deck and Demos
- Pitch Deck: [Google Slides](https://docs.google.com/presentation/d/1jK0xl5fyFLbfb_fgUN-AP216uVQqTRukC3Tl_47UAl8/edit?usp=sharing)
- Current stage and demos are available in the `assets/` folder.

## Setup

### 1. Create the Environment
Make sure you have Conda installed.

```bash
conda env create -f env.yaml
conda activate clara
```

### 2. LLM 
Install ollama from [here](https://ollama.com) and setup following instructions and make sure to run it in the background when testing code 

### 3. Run the leg extension code
```bash
cd leg_extension
python main.py
conda activate clara
```