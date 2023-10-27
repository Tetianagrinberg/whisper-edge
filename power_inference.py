import whisper
import numpy as np
import pandas as pd
import time
import threading
import os
import torch

def record_power_consumption(arr, interval=0.1, duration=30):
    start_time = time.time()

    while time.time() - start_time < duration:
        try:
            with open('/jetson-inference/power/in_current0_input') as file:
              current = float(file.read().strip())  # Ensure it's a float
              arr.append(current)

        except IOError as e:
          print("Error reading file: ", e)
    time.sleep(interval)


def save_csv(my_array, file_name):
    df = pd.DataFrame(np.asarray(my_array))
    df.to_csv(file_name)

def run_whisper_inference(latencies, whisper_model):
    speech_ids=[1,2,3,4,5]
    speech_index = 0
    for speech_id in speech_ids:
        inference_start = time.time()
        result = whisper.transcribe(model=whisper_model, audio=f"speech{speech_id}.mp3")
        inference_stop = time.time()
        latencies.append( inference_stop - inference_start )
        text = result['text'].strip()
        print(text)
        speech_index += 1


# Main script
if __name__ == "__main__":
    # Define the array size: duration / interval
    print("Loading model...")
    model = whisper.load_model(name='tiny.en')
    
    print("Initializing model buffer...")
    dummy_recording_size = 30 * 16000 #seconds times sample rate
    whisper.transcribe(model=model,
                       audio=np.zeros(dummy_recording_size, dtype=np.float32))


    power_readings = []
    latencies = []

    thread_power = threading.Thread(target=record_power_consumption, args=(power_readings,))
    thread_inference = threading.Thread(target=run_whisper_inference, args=(latencies, model))

    # Start the threads
    print("starting inference thread...")
    thread_inference.start()
    print("starting power thread...")
    thread_power.start()

    # Wait for the thread to complete
    thread_power.join()
    thread_inference.join()

    print("all threads complete")


    # Now power_readings contains the power consumption data
    print("Power consumption readings:")
    print(power_readings)

    print("latencies: ")
    print(latencies)

    save_csv(power_readings, "power_readings.csv")
    save_csv(latencies, "latencies.csv")
