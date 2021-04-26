import requests, json
import sys

#host_url = "http://54.226.217.30:5000/config/api/worker_info"
host_url = "http://localhost:5000/config/api/worker_info"


def create_worker(pid, num_speakers=2, full_diarization_timestamp=0):

    url = host_url

    worker = {  "worker_pid": pid,
                "num_speakers": num_speakers,
                "full_diarization_timestamp": full_diarization_timestamp
                }
    data = json.dumps(worker)

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=data, headers={"Content-Type": "application/json"})
    print(response.text)


def update_worker(pid, num_speakers=None, full_diarization_timestamp=None):
    url = host_url + '/' + str(pid)
    
    worker = {}

    if num_speakers is not None:
        worker["num_speakers"] = num_speakers
    if full_diarization_timestamp is not None:
        worker["full_diarization_timestamp"] = full_diarization_timestamp
    data = json.dumps(worker)

    headers = {"Content-Type": "application/json"}

    response = requests.put(url, data=data, headers={"Content-Type": "application/json"})
    print(response.text)


def get_worker(pid):

    url = host_url + '/' + str(pid)

    headers = {"Content-Type": "application/json"}

    response = requests.get(url)
    print(response.text)

    worker = json.loads(response.text)

    return worker



def delete_worker(pid):
    url = host_url + '/' + str(pid)

    headers = {"Content-Type": "application/json"}

    response = requests.delete(url)
    print(response.text)


if __name__ == '__main__':
    update_worker(int(sys.argv[1]), int(sys.argv[2]), 2999)

