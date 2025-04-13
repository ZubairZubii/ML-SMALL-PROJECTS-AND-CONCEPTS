from torch.utils.data import dataloader
from transformers import pipeline
"""
classifier = pipeline("sentiment-analysis")
text = "YOUR ARE FUCKING MAN"
print(classifier(text))
"""

"""translator = pipeline(task="translation", model="facebook/nllb-200-distilled-600M")
text = "My puppy is adorable Your kitten is cute. Her panda is friendly. His llama is thoughtful. We all have nice pets"
text_translated = translator(text,
                             src_lang="eng_Latn",
                             tgt_lang="urd_Arab")
print(text_translated)"""

'''
from datasets import load_dataset
from IPython.display import Audio as IPythonAudio

# Load dataset
data_load = load_dataset("ashraq/esc50", split="train[0:10]")

# Extract sample data
sample_data = data_load[0]

# Print audio data information
print("Sample Data:", sample_data)

# Extract audio array and sampling rate
audio_array = sample_data['audio']['array']
sampling_rate = sample_data['audio']['sampling_rate']

# Print audio array and sampling rate
print("Audio Array:", audio_array)
print("Sampling Rate:", sampling_rate)

# Display and listen to the audio
IPythonAudio(audio_array, rate=sampling_rate)

zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="laion/clap-htsat-unfused")


print(zero_shot_classifier.feature_extractor.sampling_rate)



from datasets import Audio

data_load = data_load.cast_column("audio", Audio(sampling_rate=4800))

sample_data = data_load[0]

candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]

print(zero_shot_classifier(sample_data["audio"]["array"], candidate_labels = candidate_labels))
'''






from datasets import load_dataset
from transformers import pipeline

# Load the example audio data
dataset = load_dataset("librispeech_asr", split="train.clean.100", streaming=True, trust_remote_code=True)
example = next(iter(dataset))
#print(next(iter(dataset)))
#dataload = dataset.take(5)
#print(list(dataload)[2])

# Display the audio
from IPython.display import Audio as IPythonAudio
print(IPythonAudio(example["audio"]["array"], rate=example["audio"]["sampling_rate"]))

# Create the ASR pipeline with language specified
asr = pipeline(
    task="automatic-speech-recognition",
    model="openai/whisper-large-v3",

)

print(asr(example["audio"]["array"]))




