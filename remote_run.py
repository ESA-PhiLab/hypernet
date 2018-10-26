"""
Remote run module

This module enables running tasks in different machines through Jenkins.
"""

import subprocess
import getpass
import argparse
import os
import time
import json
import sys

import jenkinsapi
from jenkinsapi.jenkins import Jenkins
from jenkinsapi.utils.crumb_requester import CrumbRequester


def process_request(
        host_name: str,
        host_port: str,
        job_name: str,
        credentials_id: str,
        repository_url: str,
        artifacts: str,
        file_name: str,
        commit_id: str,
        parameters: str,
        target_labels: str
) -> None:
    """
    Process a request
    :param host_name: (str) String with the name of the remote host
    :param host_port: (int) Port number of the remote host
    :param job_name: (str) Name of a job to run
    :param credentials_id: (str) Jenkins credentials id to be used during repository fetch
    :param repository_url: (str) Repository URL with code to be run remotely
    :param artifacts: (str) Mask of artifacts (calculation result files)
    :param file_name: (str) Name of the file to be run remotely
    :param commit_id: (str) Commit id
    :param parameters: (str) Additional parameters for python script to be run remotely
    :param target_labels: (str) Target labels
    """
    assert host_name is not None, 'Host name is required'
    assert host_port is not None, 'Port number is required'
    assert job_name is not None, 'Job name is required'
    assert credentials_id is not None, 'Credentials ID is required'
    assert repository_url is not None, 'Repository URL is required'
    assert file_name is not None, 'File name is required'
    assert commit_id is not None, 'Commit ID is required'

    if host_port == '80':
        url = 'http://' + host_name
    else:
        url = 'http://' + host_name + ':' + host_port
    server = Jenkins(
        url,
        requester=CrumbRequester(baseurl=url)
    )

    build_params = {
        'file_name': file_name,
        'commit_id': commit_id,
        'parameters': parameters,
        'target_labels': target_labels,
        'credentials_id': credentials_id,
        'repository_url': repository_url,
        "artifacts_path": artifacts
    }

    job = server[job_name]
    job.invoke(build_params=build_params)
    build = job.get_last_build()
    print("Build queued:")
    print(build.get_result_url())

def get_commit_id() -> str:
    """
    Get id of the latest commit
    :return: (str) Commit id of HEAD
    """

    process = subprocess.Popen('git rev-parse HEAD', stdout = subprocess.PIPE)

    commit_id, error = process.communicate()
    assert error is None, 'There was an error when trying to retrieve the latest commit ID'

    return commit_id[:-1].decode(sys.stdout.encoding)

def parse_arguments() -> argparse.ArgumentParser:
    """
    Parse arguments
    :return: (argparse.ArgumentParser) Parser
    """
    parser = argparse.ArgumentParser(description='Script for running remote jobs.')

    parser.add_argument(
        '-s',
        action='store',
        dest='settings',
        default='settings.json',
        help='Path to the settings .json file'
    )
    parser.add_argument(
        '-f',
        action='store',
        dest='file_name',
        help='Relative path to the script that will be run'
    )
    parser.add_argument(
        '-p',
        action='store',
        dest='parameters',
        help='Parameters of the script to be run remotely'
    )
    parser.add_argument(
        '-c',
        action='store',
        dest='commit_id',
        help='Commit id'
    )
    parser.add_argument(
        '-l',
        action='store',
        dest='target_labels',
        help='Target machine labels'
    )
    parser.add_argument(
        '--host',
        action='store',
        dest='remote_host_name',
        help='Jenkins host name'
    )
    parser.add_argument(
        '--port',
        action='store',
        dest='remote_host_port',
        help='Jenkins port number'
    )
    parser.add_argument(
        '-j',
        action='store',
        dest='job_name',
        help='Jenkins job name'
    )
    parser.add_argument(
        '-i',
        action='store',
        dest='credentials_id',
        help='Jenkins credentials ID'
    )
    parser.add_argument(
        '-u',
        action='store',
        dest='url',
        help='URL of git repository to process'
    )
    parser.add_argument(
        '-a',
        action='store',
        dest='artifacts',
        help='Artifacts to store'
    )

    return parser

def main() -> None:
    """
    Entry point
    """
    configuration = {}

    args = parse_arguments().parse_args()

    try:
        with open(args.settings) as config_file:
            configuration = json.load(config_file)
    except FileNotFoundError:
        print('{} not available'.format(args.settings))

    commit_id = args.commit_id or get_commit_id()
    process_request(
        host_name=args.remote_host_name or configuration.get('remote_host_name', None),
        host_port=args.remote_host_port or configuration.get('remote_host_port', None),
        job_name=args.job_name or configuration.get('job_name', None),
        credentials_id=args.credentials_id or configuration.get('credentials_id', None),
        repository_url=args.url or configuration.get('url', None),
        artifacts=args.artifacts or configuration.get('artifacts', None),
        commit_id=commit_id,
        file_name=args.file_name,
        parameters=args.parameters,
        target_labels=args.target_labels
    )

if __name__ == '__main__':
    main()
