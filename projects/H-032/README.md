# Bio STEMGPT, part of the Mic-Hackathon

## Team
- Bridget Vincent @BridgetSquidget, Ph.D. candidate Eco, evo, marine bio, UC Santa Barbara
- Arielle Rothman, M.S. Bio(?) (Freshly defended, woohoo!!!), University of Toronto
- Saurin Rajesh Savla, M.S. Computer Science, The Pennsylvania State University

Note: This project is being developed as part of the AI and ML for Microscopy Hackathon.

## Overview

This project aims to create a chatbot to assist biologists in segmenting and exploring their TEM data. 

## Usage

### Create and activate Virtual Environment
**Windows:**
```
python -m venv venv
venv\Scripts\activate
```
**macOS/ Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

### Install and Load the Ollama Model

Pull the required instruction tuned model:

```
ollama pull qwen2.5:3b-instruct
```

### Install required Python dependencies
```
pip install -r requirements.txt
```

### Run Flask app
```
python app.py
```
  
