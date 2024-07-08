---
title: RVC V2
emoji: 💻
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 3.42.0
app_file: app.py
pinned: false
license: lgpl-3.0
---

## 🔧 Pre-requisites

Before running the project, you must have the following tool installed on your machine: 
* [Python v3.8.0](https://www.python.org/downloads/release/python-380/)

Also, you will need to clone the repository:

```bash
# Clone the repository
git clone https://huggingface.co/spaces/mateuseap/magic-vocals/
# Enter in the root directory
cd magic-vocals
```

## 🚀 How to run

After you've cloned the repository and entered in the root directory, run the following commands:

```bash
# Create and activate a Virtual Environment (make sure you're using Python v3.8.0 to do it)
python -m venv venv
. venv/bin/activate

# Change mode and execute a shell script to configure and run the application
chmod +x run.sh
./run.sh
```

After the shell script executes everything, the application will be running at http://127.0.0.1:7860! Open up the link in a browser to use the app:

![Magic Vocals](https://i.imgur.com/V55oKv8.png)

**You only need to execute the `run.sh` one time**, once you've executed it one time, you just need to activate the virtual environment and run the command below to start the app again:

```bash
python app.py
```

**THE `run.sh` IS SUPPORTED BY THE FOLLOWING OPERATING SYSTEMS:**


| OS        | Supported |
|-----------|:---------:|
| `Windows` |     ❌    |
| `Ubuntu`  |     ✅    |