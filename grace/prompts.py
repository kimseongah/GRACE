STRUCTURED_SYS = (
    "You judge audio-visual clips. Reason briefly, then answer in EXACTLY this format and NOTHING ELSE after it:\n"
    "<think> one or two sentences localizing the evidence </think>\n"
    "<answer>yes|no</answer>\n"
    "<conf>0.0-1.0</conf>\n\n"
    "Example:\n"
    "<think> A dog is clearly visible and barking audio matches it. </think>\n"
    "<answer>yes</answer>\n"
    "<conf>0.92</conf>"
)

NATIVE_SYS = "Answer the question with a single word: yes or no."
