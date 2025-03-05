# Evaluation of Current LLM-based Machine Translation Capabilities: Russian Morphology

This repository contains code for running and evaluation of GPT4-o and DeepL models on the task of translation from Russian into English. All code was written for Python 3.13.1.

## Files in the repository

"GPT.py" , "DeepL.py" – scripts for running each model;
    
"rlc_test.csv" – Russian L2 Learners’ dataset used for evaluation;
    
"Evaluations.py" – scripts containing all evaluation metrics used;
    
"requirements.txt" – versions of libraries used in the scripts;
    
"original_src.txt" , "correct_src.txt" – text files with original and corrected source sentences;
    
"original_target_gpt.txt" , "correct_target_gpt.txt" – text files with translations of original and corrected sentences produced by GPT4-o model;
    
"original_target_deepl.txt" , "correct_target_deepl.txt" – text files with translations of original and corrected sentences produced by DeepL model.
