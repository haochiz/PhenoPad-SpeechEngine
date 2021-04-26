# Adapted from https://gist.github.com/miguelgrinberg/5614326

from flask import Flask, jsonify, abort, request, make_response, url_for
from flask_httpauth  import HTTPBasicAuth
from filelock import FileLock

import atexit
import cPickle as pickle
import os
from multiprocessing import Process
import thread
import pprint as pp

app = Flask(__name__, static_url_path = "")

WORKER_INFO_FILE = os.path.join('/tmp/worker_info.log')
WORKER_INFO_LOCK = os.path.join('/tmp/worker_info.lock')

lock = FileLock(WORKER_INFO_LOCK)
    

worker_info = []


@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)


def get_worker_info():
    global worker_info

    lock.acquire()

    try:
        with open(WORKER_INFO_FILE, 'r') as f:
            worker_info = pickle.load(f)
    except:
        worker_info = []

     # append one with pid = -1 for debugging purpose only :D
    info = {
        'worker_pid': 0,
        'num_speakers': 1,
        'full_diarization_timestamp': 0
    }

    # We do not want to repeately create the same worker :D
    repeated = False
    for i, w in enumerate(worker_info):
        if w['worker_pid'] == info['worker_pid']:
            worker_info[i] = info
            repeated = True
            break
    
    if not repeated:
        worker_info.append(info)

    lock.release()


def persist_worker_info():
    lock.acquire()
    with open(WORKER_INFO_FILE, 'w') as f:
        pickle.dump(worker_info, f)

    lock.release()


def make_public_info(info):
    new_task = {}
    for field in info:
        if field == 'worker_id':
            new_task['uri'] = url_for('get_info', task_id = info['worker_id'], _external = True)
        else:
            new_task[field] = info[field]
    print("New Task: ")
    print(new_task)
    return new_task
    

@app.route('/config/api/worker_info', methods = ['GET'])
def get_info_all():
    get_worker_info()
    return jsonify( { 'info': map(make_public_info, worker_info) } )


@app.route('/config/api/worker_info/<int:worker_pid>', methods = ['GET'])
def get_info(worker_pid):
    get_worker_info()
    info = filter(lambda t: t['worker_pid'] == worker_pid, worker_info)
    if len(info) == 0:
        abort(404)
    
    persist_worker_info()

    print("Getting " + str(worker_pid))

    return jsonify( { 'info': make_public_info(info[0]) } )


@app.route('/config/api/worker_info', methods = ['POST'])
def create_info():
    get_worker_info()
    if not request.json or not 'num_speakers' in request.json:
        abort(400)

    info = {
        'worker_pid': request.json['worker_pid'],
        'num_speakers': request.json['num_speakers'],
        'full_diarization_timestamp': request.json['full_diarization_timestamp']
    }

    # We do not want to repeately create the same worker :D
    repeated = False
    for i, w in enumerate(worker_info):
        if w['worker_pid'] == request.json['worker_pid']:
            worker_info[i] = info
            repeated = True
            break
    
    if not repeated:
        worker_info.append(info)

    persist_worker_info()
    return jsonify( { 'info': make_public_info(info) } ), 201


@app.route('/config/api/worker_info/<int:worker_pid>', methods = ['DELETE'])
def delete_info(worker_pid):
    get_worker_info()

    info = filter(lambda t: t['worker_pid'] == worker_pid, worker_info)
    if len(info) == 0:
        abort(404)
    worker_info.remove(info[0])
    persist_worker_info()
    return jsonify( { 'result': True } )


@app.route('/config/api/worker_info/<int:worker_pid>', methods = ['PUT'])
def update_info(worker_pid):
    get_worker_info()

    info = filter(lambda t: t['worker_pid'] == worker_pid, worker_info)
    if len(info) == 0:
        abort(404)
    if not request.json:
        abort(400)

    info[0]['num_speakers'] = request.json.get('num_speakers', info[0]['num_speakers'])

    print("Updating ID " + str(worker_pid) + " to " + str(info[0]['num_speakers']))

    persist_worker_info()
    return jsonify( { 'info': make_public_info(info[0]) } )


def threaded_start():
    app.run(debug = True, host='0.0.0.0')

# starts rest server, returns the thread that contains REST server
def start_rest_server():
    server_thread = thread.start_new_thread(threaded_start, ())
    return server_thread


def scripted_start():
    os.system('python kaldigstserver/rest_server.py &')


if __name__ == '__main__':
    threaded_start()

