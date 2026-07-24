# ML Engineer Interview - CBraMod Homework

Position: ML Engineer
Topic: EEG Foundation Models
Reference Paper: CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding (ICLR 2025)

## Overview
This homework evaluates your ability to understand research papers, analyze code
quality, implement alternative architectures, and design data pipelines for large-scale pretraining.
## Resources:
- Paper: https://arxiv.org/pdf/2412.07236
- Repository: https://github.com/wjq-learning/CBraMod
- Alternative architecture: https://github.com/elouayas/EEGSimpleConv

## Part 1: Code Review & Reproducibility
Goal: Analyze the CBraMod repository, reproduce results, and suggest
improvements.

### Task A: Code Review
To begin with, please review the code provided by the authors of CBraMod and comment the pros and cons of the data loading pipeline, and preprocessing code. Comment the overall code quality and the scalability concerns. As a Machine Learning engineer, what would you implement to correct the potential drawbacks?

#### Expected outcome:
It could go from a simple text with specific bullet points to a few slides.

### Task B: Reproduce Results on SHU-MI Dataset
Using the [SHU-MI dataset](https://figshare.com/articles/code/shu_dataset/19228725), reproduce the paper’s results.
You’ll need a password to access the dataset :  shu-bci2022


#### Requirements:
- Use the pretrained weights from HuggingFace
- Follow the paper’s train/val/test split (subjects 1-15 / 16-20 / 21-25)
- Document any issues encountered during reproduction

#### Expected outcome:
We expect the candidate to provide the code used to replicate the results: either a small repository or a notebook.

### GPU:
If the candidate has no GPU available, that is not necessarily a problem. The idea here is not to have the best performing model but just to make sure the code
compiles and works. If it is really an issue, we will still be able to run the provided code on our GPUs internally.

### Task C: Alternative Architecture Comparison
Goal: Implement and compare EEGSimpleConv against CBraMod.
Using the [EEGSimpleConv repository](https://github.com/elouayas/EEGSimpleConv) as reference, adapt it to run on the same SHU-MI dataset as before.
- Document the architectural and performance differences.
- Analysis: When would you prefer one over the other?

#### Expected outcome:
We expect the candidate
- To provide the code used to replicate the results: either a small repository or a notebook. Both Task B and C could be done in the same repo, of course.
- A proper comparison of both SimpleConv and CBramod is expected.

## Part 2: Data Harmonization for Large-Scale Pretraining (Design)
**Goal:** Design a strategy to unify heterogeneous EEG data sources for foundation model pretraining.
**Context:** Building a large-scale EEG foundation model requires aggregating data from multiple sources. In practice, EEG datasets vary significantly in format, channel
configuration, sampling rate, and metadata richness. Your task is to design a pretraining pipeline that could scale to terabytes of EEG data from diverse origins.




### Expected outcome:
The idea here is to give time to the candidate to allow her/him to prepare for the discussion during the debrief session.
No specific output is expected for this exercise: no need for slides or text.
### Data Sources to Consider

**Source A** CBraMod Format (SHU-MI dataset)
*   Format: Preprocessed .mdb files

**Source B** HBN Dataset (BIDS Format)
*   Reference: https://neuromechanist.github.io/data/hbn/
*   Format: BIDS-compliant (SET/BDF files)
*   Scale: 3,000+ subjects, ~1.9TB across 11 releases
*   Sampling: 500Hz (raw), 100Hz (mini datasets)
*   Channels: 128-channel EGI system
*   Rich metadata

### Design Questions
*   Data Architecture: How do you format data coming from both sources?
*   Processing Pipeline: How do you preprocess several TB of data in a way that’s reproducible and scalable, in a reasonable amount of time?
*   Data streaming: with a large training cluster, your models can ingest a lot of data (1GB/s). How do you design a data pipeline that can stream data fast enough?
*   Don’t hesitate to suggest new questions that might arise when developing such a pipeline.


## Timeline
*   Homework submission: send the code 2 days before interview
*   Debrief session: 90 minutes

Good luck!
