from openai import OpenAI
client = OpenAI(api_key = '')

audio_file= open("2nd Avenue 2.m4a", "rb")
transcription = client.audio.transcriptions.create(
  model="whisper-1", 
  file=audio_file
)
print(transcription.text)