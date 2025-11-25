# natural-turn-transcription
A companion repo to Cooney & Reece (2025): NaturalTurn: a method to segment speech into psychologically meaningful conversational turns 
https://doi.org/10.1038/s41598-025-24381-1

Codebase for transforming speech-to-text API JSON into naturalistic transcripts.  

Code and data are provided by [BetterUp](https://betterup.com). All code files copyright of BetterUp, Inc., 2024.  

This code is written with Python and Jupyter Notebooks.  

**Recommended setup using Conda**
1. Create a new conda environment with Python 3.11  
`conda create -n your_env_name python=3.11`

2. Activate the new environment  
`conda activate your_env_name`

3. Install pip if it's not already included  
`conda install pip`

4. Install packages from the requirements.txt file. Run this command from within this repo's home directory. For example, if you cloned this repo into `~/` on a Mac, do `cd ~/natural-turn-transcription` first before running the below command.  
`pip install -r requirements.txt`

5. Install the companion Spacy model   
`python -m spacy download en_core_web_sm`

6. Open `json-to-natural` Jupyter notebook    
`jupyter notebook json-to-natural.ipynb`  

7. Choose directories to pull JSON data from (`data_dir`) and to save transcripts into (`save_dir`). Set these when you initialize a `TranscriptFormatter()` instance in the code block marked `"Step 3"`.  
    * Note: If you don't make any changes to this repo, and if you are interested in using the CANDOR corpus to test out the NaturalTurn algorithm, then the default settings will work out of the box. You will, however, need to unzip `data/candor_metadata_files.zip` (the resulting folder should remain nested under `data/`). You will also need  [`aws-2024-candor-raw-api.zip`](https://osf.io/nv2ar). Save and unzip this .zip file in the repo's `data/` directory. The code provided requires both the .zip file and the unzipped directory to be in `data/`.

8. Now you can run `json-to-natural.ipynb`. It will convert AWS Transcribe output JSON to transcripts formatted using the NaturalTurn algorithm. You have the option to either save each transcript individually, or as a single concatenated CSV file. 
      

**Notes**
* This code is presented as-is for use under the attached MIT license.  
* This code comes with no guarantees of quality, performance, accuracy. 
* This repo is provided without any commitment to support, troubleshoot, or otherwise assist in the understanding, execution, or modification of the code and data within.  
* Any inquiries may be addressed to Gus Cooney (gus.cooney@gmail.com) or Andrew Reece (andrew.reece@betterup.com). 
