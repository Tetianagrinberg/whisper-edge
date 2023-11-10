from absl import app
from absl import flags
from absl import logging
from functools import partial
from functools import wraps
import numpy as np
import queue
import sounddevice as sd
import time
import whisper
import threading 

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'tiny.en',
                    'The version of the OpenAI Whisper model to use.')
flags.DEFINE_string('language', 'en',
                    'The language to use or empty to auto-detect.')
flags.DEFINE_string('input_device', 'plughw:2,0',
                    'The input device used to record audio.')
flags.DEFINE_integer('sample_rate', 16000,
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('num_channels', 1,
                     'The number of channels of the recorded audio.')
flags.DEFINE_integer('channel_index', 0,
                     'The index of the channel to use for transcription.')
flags.DEFINE_integer('chunk_seconds', 5,
                     'The length in seconds of each recorded chunk of audio.')
flags.DEFINE_string('latency', 'low', 'The latency of the recording stream.')


# A decorator to log the timing of performance-critical functions.
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        stop = time.time()
        logging.debug(f'{func.__name__} took {stop-start:.3f}s')
        print(f'{func.__name__} took {stop-start:.3f}s')
        return result
    return wrapper


@timed
def transcribe(model, audio):
    # Run the Whisper model to transcribe the audio chunk.
    result = whisper.transcribe(model=model, audio=audio)

    # Use the transcribed text.
    text = result['text'].strip()
    # text = result
    logging.info(text)


@timed
def stream_callback(indata, frames, time, status, audio_queue):
    if status:
        logging.warning(f'Stream callback status: {status}')

    # Add this chunk of audio to the queue.
    audio = indata[:, FLAGS.channel_index].copy()
    audio_queue.put_nowait(audio)


# @timed
def process_audio(audio_queue, model):
    # Block until the next chunk of audio is available on the queue.
    if not audio_queue.empty():
      # start_q = time.time()
      audio = audio_queue.get_nowait()
      # stop_q = time.time()
      # print(f"pulling from queue took took {stop_q-start_q:.3f}s")
      # Transcribe the latest audio chunk.
      transcribe(model=model, audio=audio)

    else:
      pass
    
def record_power_consumption(arr, interval=0.1, duration=180):
    start_time = time.time()
    index = 0
    while time.time() - start_time < duration:
        try:
            with open('/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power1_input') as file:
                current = float(file.read().strip())  # Ensure it's a float
                arr[index] = current
                index += 1
        except IOError as e:
            print("Error reading file: ", e)
        time.sleep(interval)

def main(argv):
    # Define the array size: duration / interval
    array_size = int(180 / 0.1)
    power_readings = np.zeros(array_size)


    # Load the Whisper model into memory, downloading first if necessary.
    logging.info(f'Loading model "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)
    
    # The first run of the model is slow (buffer init), so run it once empty.
    logging.info('Warming model up...')
    block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
    whisper.transcribe(model=model,
                       audio=np.zeros(block_size, dtype=np.float32))

    # Stream audio chunks into a queue and process them from there. The
    # callback is running on a separate thread.
    logging.info('Starting stream...')
    audio_queue = queue.Queue()
    callback = partial(stream_callback, audio_queue=audio_queue)
    # Start the thread
    thread = threading.Thread(target=record_power_consumption, args=(power_readings,))
    thread.start()
    duration=180
    stream_start=time.time()
    with sd.InputStream(samplerate=FLAGS.sample_rate,
                        blocksize=block_size,
                        device=FLAGS.input_device,
                        channels=FLAGS.num_channels,
                        dtype=np.float32,
                        latency=FLAGS.latency,
                        callback=callback):
        while time.time()-stream_start < duration:
            # Process chunks of audio from the queue.
            process_audio(audio_queue, model)
    
    # Wait for the thread to complete
    thread.join()
    print(power_readings)
    np.save("/jetson-inference/speeches/power_readings30s.npy", power_readings)

if __name__ == '__main__':
    app.run(main)
