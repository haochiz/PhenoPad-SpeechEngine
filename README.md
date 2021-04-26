PhenoPad-SpeechEngine
=====================
A real-time speech recognition/diarization server for [PhenoPad](https://github.com/jixuan-wang/PhenoPad-UWP) based on [Kaldi GStreamer sever](https://github.com/alumae/kaldi-gstreamer-server).

Installation
------------
### Requirements
#### Python 2.7 with the following:
* Tornado (need >= 4.2 for tornado.locks)
* ws4py (0.3.2) (see https://github.com/Lawouach/WebSocket-for-Python/issues/152 for more details)
* YAML
* JSON
* Python 2.x bindings for gobject-introspection libraries, provided by the `python-gi`
package on Debian and Ubuntu
* psutil
* concurrent.future (https://pypi.org/project/futures/)
* numpy
* webrtcvad
* torch
* pyannote.core 2.0.2

#### Yaafe
You can install Yaafe with [conda](https://docs.conda.io/en/latest/) or Docker, or build it from sources (https://github.com/Yaafe/Yaafe).

#### Kaldi
Download and compile Kaldi (http://kaldi.sourceforge.net). Also compile the online extensions (`make ext`)
and the Kaldi GStreamer plugin (see `README` in Kaldi's `src/gst-plugin` directory).

#### Compile the Kaldi nnet2 GStreammer plugin
Clone it from https://github.com/alumae/gst-kaldi-nnet2-online. Follow the instuctions and compile it. This should result in a file `/path/to/gst-kaldi-nnet2-online/src/libgstkaldinnet2onlinedecoder.so`. 

Make sure that the GST plugin path includes the path where the `libgstkaldinnet2onlinedecoder.so` library you compiled earlier
resides, something like:

    export GST_PLUGIN_PATH=/path/to/gst-kaldi-nnet2-online/src

Test if it worked:

    gst-inspect-1.0 kaldinnet2onlinedecoder

The latter should print out information about the new Kaldi's GStreamer plugin.

#### Download the acoustic and language models
You can download the DNN-based models for English, trained on the TEDLIUM speech corpus and combined with a generic English language model
provided by Cantab Research by running the `download-tedlium-nnet2.sh` under `test/models` to download the models (attention, 1.5 GB):

    cd test/models 
    ./download-tedlium-nnet2.sh
    cd ../../

The config for using the models is `.../PhenoPad-SpeechEngine/kaldi-gstreamer-server/sample_english_nnet2.yaml`

Running the server
------------------
Go to the root path of SpeechEngine and create the following folders:
```
data
|-- audiofiles
     |-- single_channel
     |-- multi_channel
|-- saved_kaldi_asr_results
```
In the root path of SpeechEngine and run `bash start.sh -y /path/to/your/yaml`. This starts the master server that handles client requests that listens on port 8888 and the worker that performs the recognition task. You can modify the port number in `start.sh`. When you are done, run `stop.sh` to terminate the servers.

If you run into problems when running the servers, check `master.log` and `worker.log` for logging output.
