# HackTheHill2024-Scribble-Transformer
Project repo for Hack The Hill 2024, Sep 26-28

Team Members: Ciaran McDonald-Jensen, Camila Restrepo, Tess Le Blanc

Description: Scribble Transformer is a program about self-improvement. It is a program to help you 'improve' your writting. It will take in an image of your messy handwritting, try to interpret the letters that are written, perform a spell/grammar check and possibly fix your grammar, then transform it towards slightly neater writting with better grammar/spelling. It then tries to interpret the only slightly messy writting again, performing the loop over and over again, each time getting to slightly nicer writting, and possibly changing your sentence completely towards what the program thinks you should have written. You will end up with a series of images of the steps you should take to improve your hand writting :)

NOTE: This is an unfinished project! We spent the hackathon learning a ton about AI and it was an incredible experience, but this is not a functioning project

Contents of Project: This project involved three components, the first two involve using AI models that someone else made and understanding what they did to use it for our project:

Image_Transformer - UNFINISHED. Takes input of messy writting and what it should be transformed towards, then makes messy writting closer towards what it should be. Used open source Stable Diffusion program as a template for AI image generation, we reverse engineered how diffusion process works to improve images and got the relevant code we needed, we now just need to figure out the inputs and modify it a bit to work with the other components.

Identifier - FINISHED. Takes input of messy writting and identifies what text it says. Used untrained Ocular AI model, trained the model ourselves.

Spell_Check - NOT STARTED. Takes input of text, fixes typos and grammar. This should be easy with a python library, just focused on other aspects of the project.












Dependances:
torch - https://pytorch.org/get-started/locally/
Other things probably, we didn't get that far